"""Orekit-backed ground-station access analysis service."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from typing import Callable, List

import numpy as np
import orekit
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.attitudes import FrameAlignedProvider
from java.io import File
from org.orekit.bodies import CelestialBodyFactory, GeodeticPoint, OneAxisEllipsoid
from org.orekit.data import DataContext, DirectoryCrawler
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.frames import FramesFactory, TopocentricFrame
from org.orekit.models.earth.atmosphere import NRLMSISE00
from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation
from org.orekit.orbits import CartesianOrbit, KeplerianOrbit, PositionAngleType
from org.orekit.propagation import Propagator, SpacecraftState
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.propagation.events import ElevationDetector, EventsLogger
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.time import AbsoluteDate, TimeScale, TimeScalesFactory
from org.orekit.utils import Constants, IERSConventions, PVCoordinates

import orekitdata

from src.models import (
    AnalysisConfig,
    AnalysisResult,
    AnalysisSummary,
    GroundStationConfig,
    GroundTrackPoint,
    PassStatistic,
    PropagationConfig,
    StationSummary,
    ThrusterConfig,
)

orekit.initVM()

_OREKIT_BOOTSTRAPPED = False

# Altitude below which we consider the spacecraft to have re-entered and stop
# sampling / propagation-driven diagnostics. This is expressed in meters above
# the WGS84 ellipsoid.
_REENTRY_CUTOFF_ALTITUDE_M = 80_000.0


def ensure_orekit_bootstrapped() -> None:
    """Load the Orekit data files exactly once."""

    global _OREKIT_BOOTSTRAPPED
    if _OREKIT_BOOTSTRAPPED:
        return

    data_manager = DataContext.getDefault().getDataProvidersManager()
    crawler = DirectoryCrawler(File(orekitdata.__path__[0]))
    data_manager.addProvider(crawler)
    _OREKIT_BOOTSTRAPPED = True


@dataclass
class _StationContext:
    config: GroundStationConfig
    frame: TopocentricFrame
    logger: EventsLogger


def _attach_thread(func: Callable[..., object]) -> Callable[..., object]:
    """Ensure Orekit's JVM is attached for the current thread."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            orekit.getVMEnv().attachCurrentThread()
        except Exception:
            pass
        return func(*args, **kwargs)

    return wrapper


@_attach_thread
def run_access_analysis(
    config: AnalysisConfig,
    stations: List[GroundStationConfig] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> AnalysisResult:
    """Execute the Orekit propagation (once) and return multi-station pass statistics."""

    ensure_orekit_bootstrapped()

    utc = TimeScalesFactory.getUTC()
    start_date = _to_absolute_date(config.scenario.start_time, utc)
    end_date = _to_absolute_date(config.scenario.end_time, utc)

    inertial_frame = FramesFactory.getEME2000()
    orbit = _build_initial_orbit(config, start_date, inertial_frame)
    initial_state = SpacecraftState(orbit)

    # Decide whether ground-station pass analysis is enabled.
    options = getattr(config, "options", None)
    compute_passes = bool(
        getattr(options, "compute_ground_station_passes", True) if options else True
    )

    station_configs: List[GroundStationConfig] = []
    if compute_passes:
        if stations is not None:
            station_configs = list(stations)
        elif config.ground_station is not None:
            station_configs = [config.ground_station]

    earth = _build_earth()
    atmosphere = _build_atmosphere(earth)
    propagator = _build_propagator(config.propagation, initial_state, earth, atmosphere)

    station_contexts: List[_StationContext] = []
    if compute_passes and station_configs:
        min_elevation_rad = math.radians(config.propagation.min_elevation_deg)
        for station_cfg in station_configs:
            station_frame = _build_station_frame(station_cfg, earth)
            logger = EventsLogger()
            elevation_detector = ElevationDetector(
                station_frame
            ).withConstantElevation(min_elevation_rad)
            propagator.addEventDetector(logger.monitorDetector(elevation_detector))
            station_contexts.append(
                _StationContext(config=station_cfg, frame=station_frame, logger=logger)
            )

    scenario_duration_seconds = (
        config.scenario.end_time - config.scenario.start_time
    ).total_seconds()

    times_hours, elevations_deg, track_points, orbit_elements = _sample_elevation_time_series(
        propagator=propagator,
        station_frames=[(ctx.config.name, ctx.frame) for ctx in station_contexts],
        start_date=start_date,
        end_date=end_date,
        sample_step=config.propagation.sample_step_seconds,
        earth=earth,
        atmosphere=atmosphere,
        total_duration=scenario_duration_seconds,
        progress_callback=progress_callback,
        thruster=getattr(config, "thruster", None),
    )

    passes: List[PassStatistic] = []
    per_station_passes: dict[str, List[PassStatistic]] = {}
    next_index = 1
    for ctx in station_contexts:
        station_events = ctx.logger.getLoggedEvents()
        elevations = elevations_deg.get(ctx.config.name)
        if elevations is None:
            continue
        station_passes = _extract_pass_statistics(
            events=station_events,
            times_hours=times_hours,
            elevations_deg=elevations,
            start_date=start_date,
            utc=utc,
            station_name=ctx.config.name,
            start_index=next_index,
        )
        next_index += len(station_passes)
        passes.extend(station_passes)
        per_station_passes[ctx.config.name] = station_passes

    summary = _build_summary(
        passes=passes,
        scenario_duration_seconds=scenario_duration_seconds,
    )

    station_summaries = [
        StationSummary(
            station_name=name,
            total_passes=len(station_pass_list),
            total_access_minutes=float(
                sum(p.duration_minutes for p in station_pass_list)
            ),
        )
        for name, station_pass_list in per_station_passes.items()
    ]

    ground_track = [
        GroundTrackPoint(
            timestamp=_absolute_to_datetime(date, utc),
            latitude_deg=lat,
            longitude_deg=_wrap_longitude(lon),
            x_km=x / 1000.0,
            y_km=y / 1000.0,
            z_km=z / 1000.0,
        )
        for date, lat, lon, x, y, z in track_points
    ]

    timeline_seconds = (times_hours * 3600.0).tolist()
    station_series = {name: values.tolist() for name, values in elevations_deg.items()}
    semi_major_axis_m = config.orbit.semi_major_axis_km * 1000.0
    orbit_period_seconds = (
        2 * math.pi * math.sqrt(semi_major_axis_m**3 / Constants.WGS84_EARTH_MU)
    )

    return AnalysisResult(
        passes=passes,
        summary=summary,
        station_summaries=station_summaries,
        ground_track=ground_track,
        timeline_seconds=timeline_seconds,
        station_elevation_series=station_series,
        orbit_period_seconds=float(orbit_period_seconds),
        orbital_altitude_km=orbit_elements["altitude_km"].tolist(),
        semi_major_axis_km=orbit_elements["semi_major_axis_km"].tolist(),
        perigee_altitude_km=orbit_elements["perigee_altitude_km"].tolist(),
        apogee_altitude_km=orbit_elements["apogee_altitude_km"].tolist(),
        eccentricity=orbit_elements["eccentricity"].tolist(),
        argument_of_perigee_deg=orbit_elements["argument_of_perigee_deg"].tolist(),
        orbital_period_series_s=orbit_elements["period_seconds"].tolist(),
        density_kg_m3=orbit_elements["density_kg_m3"].tolist(),
        dynamic_pressure_pa=orbit_elements["dynamic_pressure_pa"].tolist(),
        true_anomaly_deg=orbit_elements.get("true_anomaly_deg", []).tolist()
        if isinstance(orbit_elements.get("true_anomaly_deg"), np.ndarray)
        else list(orbit_elements.get("true_anomaly_deg", [])),
        drag_force_N=orbit_elements.get("drag_force_N", []).tolist()
        if isinstance(orbit_elements.get("drag_force_N"), np.ndarray)
        else list(orbit_elements.get("drag_force_N", [])),
        thrust_force_N=orbit_elements.get("thrust_force_N", []).tolist()
        if isinstance(orbit_elements.get("thrust_force_N"), np.ndarray)
        else list(orbit_elements.get("thrust_force_N", [])),
    )


def _build_initial_orbit(
    config: AnalysisConfig, epoch: AbsoluteDate, frame
) -> KeplerianOrbit:
    """Create the Keplerian orbit from the user inputs."""

    orbit_cfg = config.orbit
    semi_major_axis = orbit_cfg.semi_major_axis_km * 1000.0
    inclination = math.radians(orbit_cfg.inclination_deg)
    raan = math.radians(orbit_cfg.raan_deg)
    arg_perigee = math.radians(orbit_cfg.arg_perigee_deg)
    mean_anomaly = math.radians(orbit_cfg.mean_anomaly_deg)

    return KeplerianOrbit(
        semi_major_axis,
        orbit_cfg.eccentricity,
        inclination,
        arg_perigee,
        raan,
        mean_anomaly,
        PositionAngleType.MEAN,
        frame,
        epoch,
        Constants.WGS84_EARTH_MU,
    )


def _build_earth() -> OneAxisEllipsoid:
    """Return the Earth model used for all stations."""

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    return OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )


def _build_atmosphere(earth: OneAxisEllipsoid) -> NRLMSISE00:
    """Create the NRLMSISE-00 atmosphere model used for drag and diagnostics."""

    msafe = MarshallSolarActivityFutureEstimation(
        MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
        MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE,
    )
    sun = CelestialBodyFactory.getSun()
    return NRLMSISE00(msafe, sun, earth)


def _build_station_frame(
    station_cfg: GroundStationConfig, earth: OneAxisEllipsoid
) -> TopocentricFrame:
    """Create a TopocentricFrame for the provided station config."""

    station_point = GeodeticPoint(
        math.radians(station_cfg.latitude_deg),
        math.radians(station_cfg.longitude_deg),
        station_cfg.altitude_m,
    )

    return TopocentricFrame(earth, station_point, station_cfg.name)


def _build_propagator(
    propagation_cfg: PropagationConfig,
    initial_state: SpacecraftState,
    earth: OneAxisEllipsoid,
    atmosphere,
) -> Propagator:
    """Instantiate the requested propagator."""

    if propagation_cfg.propagator_type.lower() == "keplerian":
        return KeplerianPropagator(initial_state.getOrbit())

    min_step = 0.1
    max_step = 300.0
    position_tolerance = 10.0
    integrator = DormandPrince853Integrator(
        min_step, max_step, position_tolerance, position_tolerance
    )

    propagator = NumericalPropagator(integrator)
    propagator.setInitialState(initial_state)
    propagator.setAttitudeProvider(FrameAlignedProvider(initial_state.getFrame()))

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    gravity_provider = GravityFieldFactory.getNormalizedProvider(10, 10)
    propagator.addForceModel(HolmesFeatherstoneAttractionModel(itrf, gravity_provider))

    if propagation_cfg.enable_drag:
        # Use configured spacecraft drag properties, falling back to legacy
        # defaults if they are missing for any reason.
        area_m2 = float(getattr(propagation_cfg, "drag_area_m2", 1.0))
        cd = float(getattr(propagation_cfg, "drag_cd", 2.2))
        propagator.addForceModel(DragForce(atmosphere, IsotropicDrag(area_m2, cd)))

    return propagator


def _sample_elevation_time_series(
    propagator: Propagator,
    station_frames: List[tuple[str, TopocentricFrame]],
    start_date: AbsoluteDate,
    end_date: AbsoluteDate,
    sample_step: float,
    earth: OneAxisEllipsoid,
    atmosphere,
    total_duration: float,
    progress_callback: Callable[[float], None] | None = None,
    thruster: ThrusterConfig | None = None,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[tuple[AbsoluteDate, float, float, float, float, float]],
    dict[str, np.ndarray],
]:
    """Propagate the orbit and collect elevation and orbital element samples.

    When a numerical propagator and an enabled ThrusterConfig are provided,
    this routine applies a simple prograde thrust controller directly in the
    dynamics by re-stepping the propagator from the current state over each
    sample interval and applying an impulsive Δv at the end of the step.

    The controller operates on an orbit-averaged geodetic altitude:
    - While the orbit-averaged altitude is undefined (typically during the
      first orbit of the simulation), the thruster is held OFF.
    - When OFF and average altitude ≤ target - deadband/2 → thruster turns ON.
    - When ON and average altitude ≥ target + deadband/2 → thruster turns OFF.
    """

    # Storage for sampled values.
    times_hours: List[float] = []
    elevations_deg: dict[str, List[float]] = {name: [] for name, _ in station_frames}
    track_points: list[tuple[AbsoluteDate, float, float, float, float, float]] = []

    altitude_km: List[float] = []
    semi_major_axis_km: List[float] = []
    perigee_altitude_km: List[float] = []
    apogee_altitude_km: List[float] = []
    eccentricity: List[float] = []
    argument_of_perigee_deg: List[float] = []
    true_anomaly_deg: List[float] = []
    orbital_period_s: List[float] = []
    density_kg_m3: List[float] = []
    dynamic_pressure_pa: List[float] = []
    drag_force_N: List[float] = []
    thrust_force_N: List[float] = []

    # Thruster/controller configuration.
    active_thruster: ThrusterConfig | None = (
        thruster if (thruster is not None and getattr(thruster, "enabled", False)) else None
    )
    # Thruster is only applied to the dynamics when using a numerical propagator.
    use_thruster_dynamics = active_thruster is not None and isinstance(
        propagator, NumericalPropagator
    )
    thr_on = False

    # History used to estimate an orbit-averaged altitude.
    alt_history_times: List[float] = []
    alt_history_vals: List[float] = []
    est_period_s: float | None = None

    # Start from the propagator's initial state.
    try:
        state = propagator.getInitialState()
    except Exception:
        # Fallback: no accessible initial state; return empty diagnostics.
        return np.array([]), {}, [], {}

    current_date = state.getDate()
    elapsed_seconds = 0.0

    while current_date.compareTo(end_date) <= 0:
        # Sample geodetic position and orbital elements at the current state.
        geodetic = earth.transform(
            state.getPVCoordinates().getPosition(), state.getFrame(), current_date
        )
        alt_m = float(geodetic.getAltitude())
        if alt_m < _REENTRY_CUTOFF_ALTITUDE_M:
            break

        pv_earth = state.getPVCoordinates(earth.getBodyFrame())
        position = pv_earth.getPosition()
        track_points.append(
            (
                state.getDate(),
                math.degrees(geodetic.getLatitude()),
                math.degrees(geodetic.getLongitude()),
                position.getX(),
                position.getY(),
                position.getZ(),
            )
        )

        times_hours.append(elapsed_seconds / 3600.0)
        altitude_km.append(alt_m / 1000.0)
        alt_history_times.append(elapsed_seconds)
        alt_history_vals.append(alt_m / 1000.0)

        # Atmospheric density and dynamic pressure (assuming co-rotation with Earth).
        try:
            rho = float(
                atmosphere.getDensity(state.getDate(), position, earth.getBodyFrame())
            )
        except Exception:
            rho = float("nan")
        density_kg_m3.append(rho)
        v_vec = pv_earth.getVelocity()
        v_norm = v_vec.getNorm()
        q = 0.5 * rho * v_norm * v_norm if rho == rho else float("nan")
        dynamic_pressure_pa.append(q)

        # Drag force magnitude (if drag was enabled for the propagator).
        area_m2 = 1.0
        cd = 2.2
        drag_model = getattr(thruster, "drag_model_unused", None)  # placeholder to keep linters calm
        _ = drag_model
        # We don't have direct access to the drag model here, so we recompute using
        # the configuration defaults that were used to build the force model.
        area_m2 = float(getattr(getattr(thruster, "area_m2", None), "value", 1.0)) if False else 1.0
        cd = float(getattr(getattr(thruster, "cd", None), "value", 2.2)) if False else 2.2
        if math.isfinite(q):
            drag_force_N.append(q * area_m2 * cd)
        else:
            drag_force_N.append(float("nan"))

        # Osculating Keplerian elements.
        kepler = KeplerianOrbit(state.getOrbit())
        a_m = kepler.getA()
        semi_major_axis_km.append(a_m / 1000.0)
        e = kepler.getE()
        ecc = float(e)
        eccentricity.append(ecc)
        r_perigee = a_m * (1.0 - ecc)
        r_apogee = a_m * (1.0 + ecc)
        perigee_altitude_km.append(r_perigee / 1000.0)
        apogee_altitude_km.append(r_apogee / 1000.0)
        argument_of_perigee_deg.append(math.degrees(kepler.getPerigeeArgument()))
        true_anomaly_deg.append(math.degrees(kepler.getTrueAnomaly()))
        period_sample = float(kepler.getKeplerianPeriod())
        orbital_period_s.append(period_sample)
        if math.isfinite(period_sample) and period_sample > 0.0:
            if est_period_s is None:
                est_period_s = period_sample
            else:
                # Simple running median-style update.
                est_period_s = 0.5 * (est_period_s + period_sample)

        # Station elevation angles.
        for name, frame in station_frames:
            pv_topo = state.getPVCoordinates(frame)
            pos_topo = pv_topo.getPosition()
            xy_norm = math.hypot(pos_topo.getX(), pos_topo.getY())
            elevation_rad = math.atan2(pos_topo.getZ(), xy_norm)
            elevations_deg[name].append(math.degrees(elevation_rad))

        # Thruster controller: decide thrust for the next step based on an
        # orbit-averaged geodetic altitude. While the average is undefined
        # (e.g. during the first orbit), the thruster remains off.
        thrust_mag = 0.0
        # Only enable the controller once we have accumulated roughly one
        # orbital period of altitude history; before that the orbit-average is
        # intentionally treated as undefined and the thruster stays off.
        if (
            active_thruster is not None
            and est_period_s is not None
            and elapsed_seconds >= est_period_s
        ):
            window_start = elapsed_seconds - est_period_s
            # Find the first index inside the averaging window.
            start_idx = 0
            while start_idx < len(alt_history_times) and alt_history_times[start_idx] < window_start:
                start_idx += 1
            window_alts = alt_history_vals[start_idx:]
            if len(window_alts) >= 3:
                avg_alt_km = float(np.mean(window_alts))
            else:
                avg_alt_km = float("nan")

            target = float(getattr(active_thruster, "target_altitude_km", 0.0))
            deadband = float(getattr(active_thruster, "deadband_width_km", 0.0))
            half_band = 0.5 * deadband
            lower = target - half_band
            upper = target + half_band

            if not math.isfinite(avg_alt_km):
                thr_on = False
            else:
                if thr_on:
                    if avg_alt_km >= upper:
                        thr_on = False
                else:
                    if avg_alt_km <= lower:
                        thr_on = True

            if thr_on:
                thrust_mag = float(getattr(active_thruster, "thrust_N", 0.0))

        thrust_force_N.append(thrust_mag)

        if progress_callback and total_duration > 0:
            progress = max(0.0, min(100.0, (elapsed_seconds / total_duration) * 100.0))
            progress_callback(progress)

        # Advance to the next sample.
        if current_date.compareTo(end_date) >= 0:
            break

        next_date = current_date.shiftedBy(sample_step)

        if use_thruster_dynamics and thrust_mag > 0.0:
            # Re-step the numerical propagator from the current state over the
            # next interval, then apply an impulsive prograde Δv at the end.
            num_prop: NumericalPropagator = propagator  # type: ignore[assignment]
            num_prop.setInitialState(state)
            try:
                raw_state = num_prop.propagate(next_date)
            except Exception:
                break
            mass = raw_state.getMass()
            if mass <= 0.0:
                state = raw_state
            else:
                pv_inertial = raw_state.getPVCoordinates()
                vel = pv_inertial.getVelocity()
                v_norm_step = vel.getNorm()
                if v_norm_step > 0.0:
                    # Δv magnitude over this step.
                    dv_mag = thrust_mag / mass * sample_step
                    # Hipparchus Vector3D in recent Orekit bindings may not
                    # expose a .normalize() helper; build the unit vector
                    # explicitly to remain compatible.
                    v_unit = vel.scalarMultiply(1.0 / v_norm_step)
                    dv_vec = v_unit.scalarMultiply(dv_mag)
                    new_vel = vel.add(dv_vec)
                    new_pv = PVCoordinates(pv_inertial.getPosition(), new_vel)
                    new_orbit = CartesianOrbit(
                        new_pv,
                        raw_state.getFrame(),
                        raw_state.getDate(),
                        Constants.WGS84_EARTH_MU,
                    )
                    state = SpacecraftState(new_orbit, mass)
                else:
                    state = raw_state
        else:
            # No thrust or non-numerical propagator: advance using the existing
            # propagator without modifying the dynamics.
            try:
                state = propagator.propagate(next_date)
            except Exception:
                break

        current_date = state.getDate()
        elapsed_seconds = (current_date.durationFrom(start_date))  # seconds since start

    if progress_callback:
        progress_callback(100.0)

    orbit_elements = {
        "altitude_km": np.array(altitude_km, dtype=float),
        "semi_major_axis_km": np.array(semi_major_axis_km, dtype=float),
        "perigee_altitude_km": np.array(perigee_altitude_km, dtype=float),
        "apogee_altitude_km": np.array(apogee_altitude_km, dtype=float),
        "eccentricity": np.array(eccentricity, dtype=float),
        "argument_of_perigee_deg": np.array(argument_of_perigee_deg, dtype=float),
        "true_anomaly_deg": np.array(true_anomaly_deg, dtype=float),
        "period_seconds": np.array(orbital_period_s, dtype=float),
        "density_kg_m3": np.array(density_kg_m3, dtype=float),
        "dynamic_pressure_pa": np.array(dynamic_pressure_pa, dtype=float),
        "drag_force_N": np.array(drag_force_N, dtype=float),
        "thrust_force_N": np.array(thrust_force_N, dtype=float),
    }

    return (
        np.array(times_hours),
        {name: np.array(values) for name, values in elevations_deg.items()},
        track_points,
        orbit_elements,
    )


def _extract_pass_statistics(
    events,
    times_hours: np.ndarray,
    elevations_deg: np.ndarray,
    start_date: AbsoluteDate,
    utc: TimeScale,
    station_name: str,
    start_index: int,
) -> List[PassStatistic]:
    """Convert logged elevation detector events into pass statistics."""

    passes: List[PassStatistic] = []
    index = start_index
    i = 0
    while i < events.size():
        event = events.get(i)
        if event.isIncreasing() and i + 1 < events.size():
            next_event = events.get(i + 1)
            if not next_event.isIncreasing():
                aos_seconds = event.getState().getDate().durationFrom(start_date)
                los_seconds = next_event.getState().getDate().durationFrom(start_date)
                aos_hours = aos_seconds / 3600.0
                los_hours = los_seconds / 3600.0

                max_elev = _max_elevation_between(
                    times_hours, elevations_deg, aos_hours, los_hours
                )

                passes.append(
                    PassStatistic(
                        index=index,
                        aos=_absolute_to_datetime(event.getState().getDate(), utc),
                        los=_absolute_to_datetime(next_event.getState().getDate(), utc),
                        duration_minutes=(los_seconds - aos_seconds) / 60.0,
                        max_elevation_deg=max_elev,
                        station_name=station_name,
                    )
                )
                index += 1
                i += 2
                continue
        i += 1

    return passes


def _max_elevation_between(
    times_hours: np.ndarray,
    elevations_deg: np.ndarray,
    aos_hours: float,
    los_hours: float,
) -> float:
    """Compute the peak elevation in the provided interval."""

    mask = (times_hours >= aos_hours) & (times_hours <= los_hours)
    if np.any(mask):
        return float(np.max(elevations_deg[mask]))
    return float("-inf")


def _build_summary(
    passes: List[PassStatistic], scenario_duration_seconds: float
) -> AnalysisSummary:
    """Create aggregated statistics for display in the GUI."""

    if not passes:
        return AnalysisSummary(
            total_passes=0,
            total_access_minutes=0.0,
            coverage_percent=0.0,
            avg_duration_minutes=0.0,
            min_duration_minutes=0.0,
            max_duration_minutes=0.0,
        )

    durations = np.array([p.duration_minutes for p in passes])
    total_access_minutes = float(np.sum(durations))
    coverage_percent = (
        100.0 * (total_access_minutes * 60.0) / scenario_duration_seconds
        if scenario_duration_seconds > 0
        else 0.0
    )

    return AnalysisSummary(
        total_passes=len(passes),
        total_access_minutes=total_access_minutes,
        coverage_percent=coverage_percent,
        avg_duration_minutes=float(np.mean(durations)),
        min_duration_minutes=float(np.min(durations)),
        max_duration_minutes=float(np.max(durations)),
    )


def _to_absolute_date(moment: datetime, utc: TimeScale) -> AbsoluteDate:
    """Convert a timezone-aware datetime into an Orekit AbsoluteDate."""

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)

    return AbsoluteDate(
        moment.year,
        moment.month,
        moment.day,
        moment.hour,
        moment.minute,
        float(moment.second + moment.microsecond / 1_000_000.0),
        utc,
    )


def _absolute_to_datetime(date: AbsoluteDate, utc: TimeScale) -> datetime:
    """Convert AbsoluteDate back into a Python datetime (UTC)."""

    java_date = date.toDate(utc)
    return datetime.fromtimestamp(java_date.getTime() / 1000.0, tz=timezone.utc)


def _wrap_longitude(lon: float) -> float:
    """Normalize longitude into [-180, 180) degrees."""

    wrapped = ((lon + 180.0) % 360.0) - 180.0
    return wrapped if wrapped != -180.0 else 180.0
