"""Dataclasses shared between the UI layer and the analysis services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class GroundStationConfig:
    """Represents a user-defined ground station."""

    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float


@dataclass
class OrbitConfig:
    """Stores the classical Keplerian orbital elements."""

    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_perigee_deg: float
    mean_anomaly_deg: float


@dataclass
class PropagationConfig:
    """Holds configuration for the propagator and access detection."""

    propagator_type: str  # e.g. "keplerian" or "numerical"
    min_elevation_deg: float
    sample_step_seconds: float
    enable_drag: bool


@dataclass
class ScenarioConfig:
    """Defines the time span for the analysis."""

    start_time: datetime
    end_time: datetime


@dataclass
class AnalysisConfig:
    """Aggregates all input required to run the Orekit analysis."""

    ground_station: GroundStationConfig
    orbit: OrbitConfig
    propagation: PropagationConfig
    scenario: ScenarioConfig


@dataclass
class PassStatistic:
    """Represents a single access window."""

    index: int
    aos: datetime
    los: datetime
    duration_minutes: float
    max_elevation_deg: float
    station_name: str | None = None


@dataclass
class AnalysisSummary:
    """High-level statistics derived from all access windows."""

    total_passes: int
    total_access_minutes: float
    coverage_percent: float
    avg_duration_minutes: float
    min_duration_minutes: float
    max_duration_minutes: float


@dataclass
class StationSummary:
    """Aggregated statistics for an individual ground station."""

    station_name: str
    total_passes: int
    total_access_minutes: float


@dataclass
class GroundTrackPoint:
    """Represents a sampled point along the satellite trajectory."""

    timestamp: datetime
    latitude_deg: float
    longitude_deg: float
    x_km: float
    y_km: float
    z_km: float


@dataclass
class AnalysisResult:
    """Final payload returned to the UI after running the analysis."""

    passes: List[PassStatistic]
    summary: AnalysisSummary
    station_summaries: List[StationSummary]
    ground_track: List[GroundTrackPoint]
    timeline_seconds: List[float]
    station_elevation_series: dict[str, List[float]]
    orbit_period_seconds: float
