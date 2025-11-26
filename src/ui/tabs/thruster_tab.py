"""Thruster summary tab mixin for post-simulation burn statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from PySide6.QtWidgets import QFormLayout, QLabel, QVBoxLayout, QWidget

from src.models import AnalysisResult


@dataclass
class _ThrusterRunConfig:
    enabled: bool
    thrust_N: float
    mass_kg: float
    target_altitude_km: float
    deadband_width_km: float


class ThrusterTabMixin:
    """Provides a tab showing deadband-controlled thruster statistics."""

    def _build_thruster_tab(self) -> QWidget:
        """Create the Thruster Summary tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        self.thruster_status_label = QLabel(
            "Run an analysis with the thruster controller enabled to see burn statistics.\n"
            "The controller uses an orbit-averaged geodetic altitude estimate with a deadband "
            "around the target altitude; thrust is off while the average altitude is undefined."
        )
        self.thruster_status_label.setWordWrap(True)
        layout.addWidget(self.thruster_status_label)

        form = QFormLayout()
        self._thruster_config_label = QLabel("—")
        form.addRow("Configuration used:", self._thruster_config_label)

        self._thruster_num_burns_label = QLabel("—")
        form.addRow("Number of burns:", self._thruster_num_burns_label)

        self._thruster_total_burn_time_label = QLabel("—")
        form.addRow("Total burn duration:", self._thruster_total_burn_time_label)

        self._thruster_total_impulse_label = QLabel("—")
        form.addRow("Total impulse:", self._thruster_total_impulse_label)

        self._thruster_delta_v_label = QLabel("—")
        form.addRow("Effective Δv:", self._thruster_delta_v_label)

        layout.addLayout(form)
        layout.addStretch(1)

        return tab

    # ------------------------------------------------------------------
    # Configuration capture
    # ------------------------------------------------------------------

    def _collect_thruster_config_for_run(self) -> Dict | None:
        """Snapshot the current thruster/controller settings for an analysis run."""
        checkbox = getattr(self, "thruster_enable_checkbox", None)
        if checkbox is None or not checkbox.isChecked():
            return None

        thrust_widget = getattr(self, "thruster_thrust_input", None)
        mass_widget = getattr(self, "thruster_mass_input", None)
        target_widget = getattr(self, "thruster_target_altitude_input", None)
        deadband_widget = getattr(self, "thruster_deadband_width_input", None)
        if not all([thrust_widget, mass_widget, target_widget, deadband_widget]):
            return None

        return {
            "enabled": True,
            "thrust_N": float(thrust_widget.value()),
            "mass_kg": float(mass_widget.value()),
            "target_altitude_km": float(target_widget.value()),
            "deadband_width_km": float(deadband_widget.value()),
        }

    # ------------------------------------------------------------------
    # Summary computation and UI update
    # ------------------------------------------------------------------

    def _update_thruster_summary(self, result: AnalysisResult | None) -> None:
        """Recompute thruster statistics from the latest analysis result."""
        # Access the snapshot captured for the completed run.
        cfg_dict = getattr(self, "_last_thruster_config", None)
        if result is None or cfg_dict is None or not cfg_dict.get("enabled", False):
            self._set_thruster_summary_placeholder(
                "Thruster controller was disabled or no run has completed yet."
            )
            return

        config = _ThrusterRunConfig(
            enabled=True,
            thrust_N=float(cfg_dict.get("thrust_N", 0.0)),
            mass_kg=float(cfg_dict.get("mass_kg", 0.0)),
            target_altitude_km=float(cfg_dict.get("target_altitude_km", 0.0)),
            deadband_width_km=float(cfg_dict.get("deadband_width_km", 0.0)),
        )

        summary = self._compute_thruster_statistics(result, config)
        if summary is None:
            self._set_thruster_summary_placeholder(
                "Unable to compute orbit-averaged altitude – thruster controller statistics "
                "are unavailable for this run."
            )
            return

        self._last_thruster_summary = summary  # type: ignore[attr-defined]

        # Human-readable configuration line.
        self._thruster_config_label.setText(
            f"Thrust {config.thrust_N:.3f} N, mass {config.mass_kg:.1f} kg, "
            f"target altitude {config.target_altitude_km:.1f} km, "
            f"deadband width {config.deadband_width_km:.1f} km"
        )

        self.thruster_status_label.setText(
            "Thruster controller analysis based on orbit-averaged geodetic altitude. "
            "Thrust is disabled whenever the orbit-average altitude is undefined "
            "(typically during the first and last orbit of the simulation)."
        )

        self._thruster_num_burns_label.setText(str(summary["num_burns"]))
        self._thruster_total_burn_time_label.setText(
            f"{summary['total_burn_duration_s']:.1f} s"
        )
        self._thruster_total_impulse_label.setText(
            f"{summary['total_impulse_Ns']:.3f} N·s"
        )
        dv = summary.get("delta_v_m_s")
        if dv is not None and np.isfinite(dv):
            self._thruster_delta_v_label.setText(f"{dv:.3f} m/s")
        else:
            self._thruster_delta_v_label.setText("—")

    def _set_thruster_summary_placeholder(self, message: str) -> None:
        """Reset UI elements when no valid thruster statistics are available."""
        if getattr(self, "thruster_status_label", None) is not None:
            self.thruster_status_label.setText(message)

        for label in (
            "_thruster_config_label",
            "_thruster_num_burns_label",
            "_thruster_total_burn_time_label",
            "_thruster_total_impulse_label",
            "_thruster_delta_v_label",
        ):
            widget = getattr(self, label, None)
            if widget is not None:
                widget.setText("—")

    # ------------------------------------------------------------------
    # Core controller / statistics logic
    # ------------------------------------------------------------------

    def _compute_thruster_statistics(
        self, result: AnalysisResult, config: _ThrusterRunConfig
    ) -> Optional[Dict[str, float]]:
        """Compute thruster activity statistics from the analysis result.

        Prefer the simulated thrust force time series when available so that
        statistics match the actual propagated dynamics; fall back to a simple
        post-processed controller approximation otherwise.
        """
        times = np.asarray(getattr(result, "timeline_seconds", []), dtype=float)
        thrust_series = np.asarray(getattr(result, "thrust_force_N", []), dtype=float)

        if times.size == 0 or thrust_series.size != times.size:
            return None

        if not np.all(np.isfinite(times)) or times.size < 2:
            return None

        # Infer on/off pattern directly from the thrust magnitude.
        thrust_mag = float(config.thrust_N)
        if thrust_mag <= 0.0:
            return None
        on_flags = thrust_series > (0.5 * thrust_mag)

        if not np.any(on_flags):
            return {
                "num_burns": 0.0,
                "total_burn_duration_s": 0.0,
                "total_impulse_Ns": 0.0,
                "delta_v_m_s": 0.0,
            }

        # Count distinct burns from rising edges in the on/off sequence.
        num_burns = 0
        for i in range(1, on_flags.size):
            if on_flags[i] and not on_flags[i - 1]:
                num_burns += 1

        # Approximate total burn duration using per-step durations.
        step_dt = np.empty_like(times)
        step_dt[:-1] = np.diff(times)
        step_dt[-1] = step_dt[-2] if times.size > 1 else 0.0
        total_burn_s = float(np.sum(step_dt[on_flags]))

        thrust = max(config.thrust_N, 0.0)
        total_impulse = thrust * total_burn_s
        mass = max(config.mass_kg, 0.0)
        delta_v = total_impulse / mass if mass > 0.0 else None

        return {
            "num_burns": float(num_burns),
            "total_burn_duration_s": total_burn_s,
            "total_impulse_Ns": total_impulse,
            "delta_v_m_s": float(delta_v) if delta_v is not None else float("nan"),
        }


