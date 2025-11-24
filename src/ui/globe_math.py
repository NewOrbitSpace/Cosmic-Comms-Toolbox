"""Small shared helpers for globe math transforms."""

from __future__ import annotations

import math
from typing import Tuple


def rotate_vector_z(
    vector: tuple[float, float, float],
    angle_rad: float,
) -> tuple[float, float, float]:
    """Rotate the provided vector around the +Z axis by the given angle."""
    cos_ang = math.cos(angle_rad)
    sin_ang = math.sin(angle_rad)
    x, y, z = vector
    return (cos_ang * x - sin_ang * y, sin_ang * x + cos_ang * y, z)


__all__: Tuple[str, ...] = ("rotate_vector_z",)

