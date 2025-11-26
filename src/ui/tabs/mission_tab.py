"""Mission analysis tab mixin."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.models import GroundTrackPoint
from src.ui.constants import EARTH_ROTATION_RATE_RAD_PER_SEC
from src.ui.globe_math import rotate_vector_z
from src.ui.tabs.visualization_tab import ecef_to_globe_coords
from src.ui.opengl import GlobeWidget


class MissionTabMixin:
    """Logic for building and updating the mission analysis tab."""

    def _build_mission_analysis_tab(self) -> QWidget:
        """Create the Mission Configuration tab with configuration and mission map."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.addWidget(self._build_orbit_group())
        config_layout.addWidget(self._build_scenario_group())
        config_layout.addWidget(self._build_propagation_group())
        button_row = QHBoxLayout()
        # Stack Run/Stop vertically so the stop button appears directly under Run.
        button_column = QVBoxLayout()
        self.run_button = QPushButton("Run Analysis")
        self._set_run_button_state("dirty")
        self.run_button.clicked.connect(self._handle_run_clicked)  # type: ignore[arg-type]
        button_column.addWidget(self.run_button)
        # Stop button is disabled until a run is in progress.
        from PySide6.QtWidgets import QPushButton as _QPushButtonAlias  # local alias to avoid circular import hints

        self.stop_button = getattr(self, "stop_button", None)
        if self.stop_button is None:
            self.stop_button = _QPushButtonAlias("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._handle_stop_clicked)  # type: ignore[arg-type]
        button_column.addWidget(self.stop_button)
        button_row.addLayout(button_column)
        self.run_progress = QProgressBar()
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(0)
        self.run_progress.setFormat("0%")
        self.run_progress.setTextVisible(True)
        self.run_progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.run_progress.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #1E90FF;
                border-radius: 4px;
                background-color: #0b1f2d;
                color: white;
                padding: 2px;
                min-height: 28px;
            }
            QProgressBar::chunk {
                background-color: #76c7ff;
                border-radius: 4px;
            }
            """
        )
        self.run_progress.hide()
        button_row.addWidget(self.run_progress)
        config_layout.addLayout(button_row)
        config_layout.addStretch(1)
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        # Mission tab now focuses on configuration + orbit overview globe.
        analysis_layout.addWidget(self._build_mission_globe_panel())
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(config_widget)
        splitter.addWidget(analysis_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        tab_layout.addWidget(splitter)
        return tab

    def _build_mission_globe_panel(self) -> QWidget:
        """Create the mission analysis globe with frame toggle and orbit slider."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.mission_globe_widget = GlobeWidget()
        self.mission_globe_widget.reset_camera()
        self.mission_globe_widget.set_day_night_enabled(False)
        self.mission_globe_widget.set_uniform_lighting(True)
        self._update_mission_earth_rotation(None)
        layout.addWidget(self.mission_globe_widget, stretch=1)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Frame:"))
        self.mission_frame_combo = QComboBox()
        self.mission_frame_combo.addItems(["ECI (inertial)", "ECEF (earth-fixed)"])
        self.mission_frame_combo.setCurrentIndex(0)
        self.mission_frame_combo.currentIndexChanged.connect(
            self._handle_mission_frame_changed
        )  # type: ignore[attr-defined]
        controls.addWidget(self.mission_frame_combo)
        controls.addWidget(QLabel("Scenario window:"))
        self.mission_window_slider = QSlider(Qt.Orientation.Horizontal)
        self.mission_window_slider.setRange(0, 1000)
        self.mission_window_slider.setSingleStep(5)
        self.mission_window_slider.setValue(0)
        self.mission_window_slider.valueChanged.connect(
            self._handle_mission_window_changed
        )  # type: ignore[attr-defined]
        controls.addWidget(self.mission_window_slider, stretch=1)
        self.mission_window_label = QLabel("Start → End")
        controls.addWidget(self.mission_window_label)
        layout.addLayout(controls)
        self.mission_globe_status_label = QLabel("Run analysis to view ground track.")
        self.mission_globe_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.mission_globe_status_label)
        self._apply_mission_control_state()
        return tab

    def _build_orbit_group(self) -> QGroupBox:
        """Create widgets for the orbital elements."""
        group = QGroupBox("Initial State")
        form = QFormLayout(group)
        self.sma_input = QDoubleSpinBox()
        self.sma_input.setRange(6378.0, 50000.0)
        self.sma_input.setValue(6628.0)
        self.sma_input.setSuffix(" km")
        self.ecc_input = QDoubleSpinBox()
        self.ecc_input.setRange(0.0, 0.99)
        self.ecc_input.setDecimals(5)
        self.ecc_input.setValue(0.0)
        self.inc_input = QDoubleSpinBox()
        self.inc_input.setRange(0.0, 180.0)
        self.inc_input.setValue(97.4)
        self.inc_input.setSuffix(" °")
        self.raan_input = QDoubleSpinBox()
        self.raan_input.setRange(0.0, 360.0)
        self.raan_input.setValue(0.0)
        self.raan_input.setSuffix(" °")
        self.argp_input = QDoubleSpinBox()
        self.argp_input.setRange(0.0, 360.0)
        self.argp_input.setValue(90.0)
        self.argp_input.setSuffix(" °")
        self.mean_anom_input = QDoubleSpinBox()
        self.mean_anom_input.setRange(0.0, 360.0)
        self.mean_anom_input.setValue(0.0)
        self.mean_anom_input.setSuffix(" °")
        form.addRow("Semi-major axis:", self.sma_input)
        self.sma_input.valueChanged.connect(lambda *_: self._mark_dirty())
        form.addRow("Eccentricity:", self.ecc_input)
        self.ecc_input.valueChanged.connect(lambda *_: self._mark_dirty())
        form.addRow("Inclination:", self.inc_input)
        self.inc_input.valueChanged.connect(lambda *_: self._mark_dirty())
        form.addRow("RAAN:", self.raan_input)
        self.raan_input.valueChanged.connect(lambda *_: self._mark_dirty())
        form.addRow("Arg Perigee:", self.argp_input)
        self.argp_input.valueChanged.connect(lambda *_: self._mark_dirty())
        form.addRow("Mean Anomaly:", self.mean_anom_input)
        self.mean_anom_input.valueChanged.connect(lambda *_: self._mark_dirty())
        return group

    def _build_scenario_group(self) -> QGroupBox:
        """Create date/time controls for the scenario window."""
        group = QGroupBox("Scenario Window")
        layout = QHBoxLayout(group)
        default_start = datetime(2020, 5, 1, 11, 36, 0, tzinfo=timezone.utc)
        from PySide6.QtWidgets import QDateTimeEdit

        self.start_datetime = QDateTimeEdit(default_start)
        self.start_datetime.setDisplayFormat("dd-MMM-yyyy HH:mm:ss")
        self.start_datetime.setCalendarPopup(True)
        self.end_datetime = QDateTimeEdit(default_start + timedelta(days=7))
        self.end_datetime.setDisplayFormat("dd-MMM-yyyy HH:mm:ss")
        self.end_datetime.setCalendarPopup(True)
        layout.addWidget(QLabel("Start (UTC):"))
        layout.addWidget(self.start_datetime)
        self.start_datetime.dateTimeChanged.connect(lambda *_: self._mark_dirty())
        layout.addWidget(QLabel("End (UTC):"))
        layout.addWidget(self.end_datetime)
        self.end_datetime.dateTimeChanged.connect(lambda *_: self._mark_dirty())
        return group

    def _build_propagation_group(self) -> QGroupBox:
        """Create controls for propagation and high-level analysis options."""
        group = QGroupBox("Analysis Configuration")
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self.propagator_combo = QComboBox()
        self.propagator_combo.addItems(["numerical", "keplerian"])
        self.min_elev_input = QDoubleSpinBox()
        self.min_elev_input.setRange(0.0, 90.0)
        self.min_elev_input.setValue(0.5)
        self.sample_step_input = QSpinBox()
        self.sample_step_input.setRange(10, 3600)
        self.sample_step_input.setValue(60)
        form.addRow("Propagator:", self.propagator_combo)
        self.propagator_combo.currentIndexChanged.connect(lambda *_: self._mark_dirty())
        form.addRow("Sample Step (s):", self.sample_step_input)
        self.sample_step_input.valueChanged.connect(lambda *_: self._mark_dirty())
        layout.addLayout(form)

        # Ground-station analysis settings and toggles
        from PySide6.QtWidgets import QCheckBox

        self.gs_access_group = QGroupBox("Ground-station Pass Settings")
        gs_form = QFormLayout(self.gs_access_group)
        self.ground_pass_checkbox = getattr(self, "ground_pass_checkbox", None)
        if self.ground_pass_checkbox is None:
            self.ground_pass_checkbox = QCheckBox("Include ground-station passes")
        # Start unchecked, but keep the group enabled so the user can always
        # re-toggle the option; only the analysis behavior depends on the state.
        self.ground_pass_checkbox.setChecked(False)
        self.ground_pass_checkbox.stateChanged.connect(
            self._handle_analysis_options_changed
        )  # type: ignore[arg-type]
        gs_form.addRow(self.ground_pass_checkbox)
        gs_form.addRow("Min Elevation (deg):", self.min_elev_input)
        self.min_elev_input.valueChanged.connect(lambda *_: self._mark_dirty())
        layout.addWidget(self.gs_access_group)

        # Drag configuration in its own box
        self.drag_checkbox = getattr(self, "drag_checkbox", None)
        if self.drag_checkbox is None:
            from PySide6.QtWidgets import QCheckBox

            # Keep the label simple as requested.
            self.drag_checkbox = QCheckBox("Include drag")
        # Start with drag enabled by default.
        self.drag_checkbox.setChecked(True)
        self.drag_checkbox.stateChanged.connect(self._handle_drag_toggle)  # type: ignore[arg-type]
        self.drag_area_input = QDoubleSpinBox()
        self.drag_area_input.setRange(0.01, 100.0)
        self.drag_area_input.setSingleStep(0.05)
        self.drag_area_input.setValue(0.43)
        self.drag_area_input.setSuffix(" m²")
        self.drag_area_input.valueChanged.connect(
            lambda *_: self._handle_drag_parameters_changed()
        )
        self.drag_cd_input = QDoubleSpinBox()
        self.drag_cd_input.setRange(1.0, 5.0)
        self.drag_cd_input.setSingleStep(0.1)
        self.drag_cd_input.setValue(3.0)
        self.drag_cd_input.valueChanged.connect(
            lambda *_: self._handle_drag_parameters_changed()
        )
        self.drag_cd_area_label = QLabel()
        self.drag_cd_area_label.setStyleSheet("color: #DDD;")
        self.drag_params_group = QGroupBox("Spacecraft Drag Properties")
        params_layout = QFormLayout(self.drag_params_group)
        params_layout.addRow("Cross-sectional area:", self.drag_area_input)
        params_layout.addRow("Drag coefficient:", self.drag_cd_input)
        params_layout.addRow("C_d × A:", self.drag_cd_area_label)
        self.drag_params_group.setEnabled(self.drag_checkbox.isChecked())

        self.drag_group = QGroupBox("Drag Configuration")
        drag_form = QFormLayout(self.drag_group)
        # Just show the checkbox with its label, without an extra \"Enable drag:\"
        # caption row.
        drag_form.addRow(self.drag_checkbox)
        drag_form.addRow(self.drag_params_group)
        layout.addWidget(self.drag_group)

        # Thruster + controller configuration in its own box
        from PySide6.QtWidgets import QCheckBox

        self.thruster_group = QGroupBox("Thruster & Altitude-Hold Controller")
        thruster_form = QFormLayout(self.thruster_group)
        self.thruster_enable_checkbox = getattr(self, "thruster_enable_checkbox", None)
        if self.thruster_enable_checkbox is None:
            self.thruster_enable_checkbox = QCheckBox(
                "Enable prograde thruster to counteract drag"
            )
        # Start with the thruster controller enabled by default.
        self.thruster_enable_checkbox.setChecked(True)
        self.thruster_enable_checkbox.stateChanged.connect(
            self._handle_thruster_options_changed
        )  # type: ignore[arg-type]
        thruster_form.addRow(self.thruster_enable_checkbox)

        self.thruster_thrust_input = QDoubleSpinBox()
        self.thruster_thrust_input.setRange(0.0, 1000.0)
        self.thruster_thrust_input.setDecimals(3)
        self.thruster_thrust_input.setSingleStep(0.01)
        self.thruster_thrust_input.setValue(0.01)
        self.thruster_thrust_input.setSuffix(" N")
        self.thruster_thrust_input.valueChanged.connect(lambda *_: self._mark_dirty())
        thruster_form.addRow("Thrust:", self.thruster_thrust_input)

        self.thruster_mass_input = QDoubleSpinBox()
        self.thruster_mass_input.setRange(1.0, 10_000.0)
        self.thruster_mass_input.setDecimals(1)
        self.thruster_mass_input.setSingleStep(1.0)
        self.thruster_mass_input.setValue(200.0)
        self.thruster_mass_input.setSuffix(" kg")
        self.thruster_mass_input.valueChanged.connect(lambda *_: self._mark_dirty())
        thruster_form.addRow("Spacecraft mass:", self.thruster_mass_input)

        self.thruster_target_altitude_input = QDoubleSpinBox()
        self.thruster_target_altitude_input.setRange(150.0, 50_000.0)
        self.thruster_target_altitude_input.setDecimals(1)
        self.thruster_target_altitude_input.setSingleStep(5.0)
        self.thruster_target_altitude_input.setValue(250.0)
        self.thruster_target_altitude_input.setSuffix(" km")
        self.thruster_target_altitude_input.valueChanged.connect(
            lambda *_: self._mark_dirty()
        )
        thruster_form.addRow("Target geodetic altitude:", self.thruster_target_altitude_input)

        self.thruster_deadband_width_input = QDoubleSpinBox()
        self.thruster_deadband_width_input.setRange(1.0, 5000.0)
        self.thruster_deadband_width_input.setDecimals(1)
        self.thruster_deadband_width_input.setSingleStep(1.0)
        self.thruster_deadband_width_input.setValue(1.0)
        self.thruster_deadband_width_input.setSuffix(" km")
        self.thruster_deadband_width_input.valueChanged.connect(
            lambda *_: self._mark_dirty()
        )
        thruster_form.addRow("Deadband width:", self.thruster_deadband_width_input)

        layout.addWidget(self.thruster_group)

        # Initialise drag-related derived fields and control states.
        self._handle_drag_parameters_changed()
        self._handle_thruster_options_changed(0)
        return group

    def _handle_drag_toggle(self, state: int) -> None:
        """Enable or disable drag configuration controls."""
        _ = state
        enabled = bool(getattr(self, "drag_checkbox", None) and self.drag_checkbox.isChecked())
        if hasattr(self, "drag_params_group") and self.drag_params_group is not None:
            self.drag_params_group.setEnabled(enabled)
        self._mark_dirty()
        self._update_drag_cd_area()

    def _handle_drag_parameters_changed(self) -> None:
        """React to drag parameter input changes."""
        self._update_drag_cd_area()
        self._mark_dirty()

    def _handle_thruster_options_changed(self, state: int) -> None:
        """Enable or disable thruster controller configuration controls."""
        _ = state
        enabled = bool(
            getattr(self, "thruster_enable_checkbox", None)
            and self.thruster_enable_checkbox.isChecked()
        )
        for widget_name in (
            "thruster_thrust_input",
            "thruster_mass_input",
            "thruster_target_altitude_input",
            "thruster_deadband_width_input",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(enabled)
        self._mark_dirty()

    def _handle_analysis_options_changed(self, state: int) -> None:
        """React to analysis option toggles (e.g., ground-station passes)."""
        _ = state
        enabled = self.ground_pass_checkbox.isChecked()
        # Only gate the inner controls (like min elevation) on this toggle, not
        # the checkbox itself, so the user can always re-enable GS passes.
        if hasattr(self, "min_elev_input") and self.min_elev_input is not None:
            self.min_elev_input.setEnabled(enabled)
        self._mark_dirty()

    def _update_drag_cd_area(self) -> None:
        """Compute and display the combined drag term."""
        if not hasattr(self, "drag_cd_area_label") or self.drag_cd_area_label is None:
            return
        if not self.drag_checkbox.isChecked():
            self.drag_cd_area_label.setText("Disabled")
            return
        area = self.drag_area_input.value() if hasattr(self, "drag_area_input") else 0.0
        cd = self.drag_cd_input.value() if hasattr(self, "drag_cd_input") else 0.0
        product = cd * area
        self.drag_cd_area_label.setText(f"{product:.3f} m²")

    def _handle_mission_frame_changed(self, index: int) -> None:
        """Toggle the mission globe between inertial and Earth-fixed frames."""
        mode = "ECI" if index == 0 else "ECEF"
        if mode == self._mission_frame_mode:
            return
        self._mission_frame_mode = mode
        if getattr(self, "_mission_globe_refresh_timer", None) is not None:
            self._mission_globe_refresh_timer.stop()
        self._apply_mission_control_state()
        self._refresh_mission_globe()

    def _handle_mission_window_changed(self, value: int) -> None:
        """Adjust the portion of the scenario shown on the mission globe."""
        coverage = min(max(value / 1000.0, 0.0), 1.0)
        if coverage <= 0.0:
            self._mission_window_fractions = (0.0, 0.0)
        else:
            self._mission_window_fractions = (0.0, coverage)
        self._update_mission_window_label()
        self._schedule_mission_globe_refresh()

    def _apply_mission_control_state(self) -> None:
        """Enable or disable orbit controls."""
        if self.mission_window_slider:
            self.mission_window_slider.setEnabled(True)
        self._update_mission_window_label()

    def _update_mission_window_label(self) -> None:
        """Update the scenario window label."""
        label = self.mission_window_label
        if label is None:
            return
        if self._current_config is None:
            label.setText("Scenario window")
            return
        start_frac, end_frac = self._mission_window_fractions
        start_dt = self._current_config.scenario.start_time
        end_dt = self._current_config.scenario.end_time
        total = (end_dt - start_dt).total_seconds()
        if total <= 0:
            label.setText("Scenario window")
            return
        window_start = start_dt + timedelta(seconds=total * start_frac)
        window_end = start_dt + timedelta(seconds=total * end_frac)
        label.setText(
            f"{window_start:%d-%b %H:%M UTC} → {window_end:%d-%b %H:%M UTC}"
        )

    def _refresh_mission_globe(self) -> None:
        """Re-render the mission globe track using the latest state."""
        if getattr(self, "_mission_globe_refresh_timer", None) is not None:
            self._mission_globe_refresh_timer.stop()
        if getattr(self, "mission_globe_widget", None) is None:
            return
        track_points = self._mission_ground_track
        if track_points is None and getattr(self, "_last_result", None):
            track_points = self._last_result.ground_track
        if not track_points:
            return
        self._update_mission_globe_track(track_points)

    def _schedule_mission_globe_refresh(self) -> None:
        """Debounce mission globe refreshes when controls change."""
        if self._mission_globe_refresh_timer is None:
            self._refresh_mission_globe()
            return
        # Don't clear the track immediately - let the timer handle the refresh
        # This prevents flickering when the slider is moved
        self._mission_globe_refresh_timer.start()

    def _update_mission_globe_track(self, track_points: list[GroundTrackPoint]) -> None:
        """Plot the satellite ground track on the mission globe."""
        widget = getattr(self, "mission_globe_widget", None)
        if widget is None:
            return
        if not track_points or self._current_config is None:
            if self.mission_globe_status_label:
                self.mission_globe_status_label.setText(
                    "Run analysis to view ground track."
                )
            widget.update_track(None)
            widget.update_satellite_position(None)
            return
        segment = self._extract_window_segment(track_points)
        if not segment:
            if self.mission_globe_status_label:
                slider_value = (
                    self.mission_window_slider.value()
                    if self.mission_window_slider is not None
                    else None
                )
                if slider_value == 0:
                    message = "Move the orbit slider to reveal the ground track."
                else:
                    message = "No ground track samples in the selected window."
                self.mission_globe_status_label.setText(message)
            widget.update_track(None)
            widget.update_satellite_position(None)
            widget.update_direction_arrow(None, None)
            return
        coords = np.array(
            [
                self._mission_transform_vector(
                    ecef_to_globe_coords(pt.x_km, pt.y_km, pt.z_km), pt.timestamp
                )
                for pt in segment
            ],
            dtype=float,
        )

        # Downsample if too many points for rendering performance
        MAX_RENDER_POINTS = 25000
        if coords.shape[0] > MAX_RENDER_POINTS:
            step = coords.shape[0] // MAX_RENDER_POINTS
            coords = coords[::step]

        self._update_mission_earth_rotation(segment[0].timestamp if segment else None)
        widget.update_track(coords)
        widget.update_satellite_position(None)
        if coords.shape[0] >= 2:
            direction = coords[-1] - coords[-2]
            widget.update_direction_arrow(tuple(coords[-1]), tuple(direction))
        else:
            widget.update_direction_arrow(None, None)
        if self.mission_globe_status_label:
            start = segment[0].timestamp.strftime("%d-%b %H:%M")
            end = segment[-1].timestamp.strftime("%d-%b %H:%M")
            self.mission_globe_status_label.setText(
                f"Showing {start} → {end} ({len(segment)} samples)"
            )

    def _extract_window_segment(
        self, track_points: list[GroundTrackPoint]
    ) -> list[GroundTrackPoint]:
        """Return track points falling inside the selected scenario window."""
        if (
            not track_points
            or self._current_config is None
            or self._current_config.scenario is None
        ):
            return track_points
        start_frac, end_frac = self._mission_window_fractions
        start_dt = self._current_config.scenario.start_time
        end_dt = self._current_config.scenario.end_time
        total = (end_dt - start_dt).total_seconds()
        if total <= 0:
            return track_points
        window_start = start_dt + timedelta(seconds=total * start_frac)
        window_end = start_dt + timedelta(seconds=total * end_frac)
        if window_end <= window_start:
            segment: list[GroundTrackPoint] = []
        else:
            segment = [
                point
                for point in track_points
                if window_start <= point.timestamp <= window_end
            ]


        return segment

    def _mission_transform_vector(
        self, vector: tuple[float, float, float], timestamp: datetime
    ) -> tuple[float, float, float]:
        """Transform vectors for the mission overview globe."""
        if self._mission_frame_mode != "ECI":
            return vector
        if self._mission_reference_epoch is None:
            return vector
        delta = (timestamp - self._mission_reference_epoch).total_seconds()
        angle = EARTH_ROTATION_RATE_RAD_PER_SEC * delta
        return rotate_vector_z(vector, angle)

    def _update_mission_earth_rotation(self, timestamp: datetime | None) -> None:
        """Rotate the mission globe actors to match the selected frame."""
        if self._mission_frame_mode == "ECI":
            if timestamp is None or self._mission_reference_epoch is None:
                angle_deg = 0.0
            else:
                delta = (timestamp - self._mission_reference_epoch).total_seconds()
                angle_deg = math.degrees(EARTH_ROTATION_RATE_RAD_PER_SEC * delta)
        else:
            angle_deg = 0.0
        widget = getattr(self, "mission_globe_widget", None)
        if widget is None:
            return
        widget.set_frame_rotation(self._mission_frame_mode, angle_deg)

