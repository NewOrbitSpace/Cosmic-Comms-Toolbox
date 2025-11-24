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
from org.orekit.bodies import GeodeticPoint, OneAxisEllipsoid
from org.orekit.data import DataContext, DirectoryCrawler
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.frames import FramesFactory, TopocentricFrame
from org.orekit.models.earth.atmosphere import SimpleExponentialAtmosphere
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.propagation import Propagator, SpacecraftState
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.propagation.events import ElevationDetector, EventsLogger
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.time import AbsoluteDate, TimeScale, TimeScalesFactory
from org.orekit.utils import Constants, IERSConventions

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
)

orekit.initVM()

_OREKIT_BOOTSTRAPPED = False


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

    station_configs = stations or [config.ground_station]
    earth = _build_earth()
    propagator = _build_propagator(config.propagation, initial_state, earth)

    min_elevation_rad = math.radians(config.propagation.min_elevation_deg)
    station_contexts: List[_StationContext] = []
    for station_cfg in station_configs:
        station_frame = _build_station_frame(station_cfg, earth)
        logger = EventsLogger()
        elevation_detector = ElevationDetector(station_frame).withConstantElevation(
            min_elevation_rad
        )
        propagator.addEventDetector(logger.monitorDetector(elevation_detector))
        station_contexts.append(
            _StationContext(config=station_cfg, frame=station_frame, logger=logger)
        )

    scenario_duration_seconds = (
        config.scenario.end_time - config.scenario.start_time
    ).total_seconds()

    times_hours, elevations_deg, track_points = _sample_elevation_time_series(
        propagator=propagator,
        station_frames=[(ctx.config.name, ctx.frame) for ctx in station_contexts],
        start_date=start_date,
        end_date=end_date,
        sample_step=config.propagation.sample_step_seconds,
        earth=earth,
        total_duration=scenario_duration_seconds,
        progress_callback=progress_callback,
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
        atmosphere = SimpleExponentialAtmosphere(earth, 0.0004, 42000.0, 7500.0)
        propagator.addForceModel(DragForce(atmosphere, IsotropicDrag(1.0, 2.2)))

    return propagator


def _sample_elevation_time_series(
    propagator: Propagator,
    station_frames: List[tuple[str, TopocentricFrame]],
    start_date: AbsoluteDate,
    end_date: AbsoluteDate,
    sample_step: float,
    earth: OneAxisEllipsoid,
    total_duration: float,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[tuple[AbsoluteDate, float, float, float, float, float]],
]:
    """Propagate the orbit and collect elevation samples."""

    current_date = start_date
    times_hours: List[float] = []
    elevations_deg: dict[str, List[float]] = {name: [] for name, _ in station_frames}
    track_points: list[tuple[AbsoluteDate, float, float, float, float, float]] = []
    elapsed_seconds = 0.0

    while current_date.compareTo(end_date) <= 0:
        state = propagator.propagate(current_date)
        geodetic = earth.transform(
            state.getPVCoordinates().getPosition(), state.getFrame(), current_date
        )
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
        for name, frame in station_frames:
            pv_topo = state.getPVCoordinates(frame)
            position = pv_topo.getPosition()
            xy_norm = math.hypot(position.getX(), position.getY())
            elevation_rad = math.atan2(position.getZ(), xy_norm)
            elevations_deg[name].append(math.degrees(elevation_rad))

        if progress_callback and total_duration > 0:
            progress = max(0.0, min(100.0, (elapsed_seconds / total_duration) * 100.0))
            progress_callback(progress)

        elapsed_seconds += sample_step
        current_date = start_date.shiftedBy(elapsed_seconds)

    if progress_callback:
        progress_callback(100.0)

    return (
        np.array(times_hours),
        {name: np.array(values) for name, values in elevations_deg.items()},
        track_points,
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
