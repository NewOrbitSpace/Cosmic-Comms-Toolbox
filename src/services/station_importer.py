"""Utilities for loading ground-station definitions from CSV/Excel files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from src.models import GroundStationConfig

REQUIRED_COLUMNS = ["name", "latitude", "longitude", "altitude"]


@dataclass
class StationImportError(Exception):
    """Raised when the ground-station file cannot be parsed."""

    message: str

    def __str__(self) -> str:
        return self.message


def load_ground_stations_from_file(path: str | Path) -> List[GroundStationConfig]:
    """Load stations from a CSV or Excel file."""

    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise StationImportError(f"File not found: {file_path}")

    df = _read_dataframe(file_path)
    normalized_columns = {col.lower(): col for col in df.columns}

    missing = [col for col in REQUIRED_COLUMNS if col not in normalized_columns]
    if missing:
        raise StationImportError(
            f"Missing required columns: {', '.join(missing)}. "
            "Expected columns: name, latitude, longitude, altitude."
        )

    stations: List[GroundStationConfig] = []
    for _, row in df.iterrows():
        try:
            stations.append(
                GroundStationConfig(
                    name=str(row[normalized_columns["name"]]).strip(),
                    latitude_deg=float(row[normalized_columns["latitude"]]),
                    longitude_deg=float(row[normalized_columns["longitude"]]),
                    altitude_m=float(row[normalized_columns["altitude"]]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise StationImportError(
                f"Invalid numeric value in row {_ + 1}: {exc}"
            ) from exc

    if not stations:
        raise StationImportError("No rows were found in the provided file.")

    return stations


def _read_dataframe(file_path: Path):
    """Return a pandas DataFrame from CSV or Excel input."""

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    raise StationImportError(
        f"Unsupported file type '{suffix}'. Please select CSV or Excel."
    )
