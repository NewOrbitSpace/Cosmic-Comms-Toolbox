"""Visualization tab mixin."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.models import GroundStationConfig, GroundTrackPoint, PassStatistic
from src.ui.constants import EARTH_ROTATION_RATE_RAD_PER_SEC
from src.ui.globe_math import rotate_vector_z
from src.ui.opengl import GlobeWidget


def ecef_to_globe_coords(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert ECEF/ITRF coordinates to globe plotting coordinate system.

    The globe's texture mapping uses atan2(x,y) which creates a coordinate system
    rotated 90° from standard ECEF. This function applies a +90° Z-rotation:

    Standard ECEF:      Globe Coordinates:
    +X → 0° longitude   +Y → 0° longitude (Prime Meridian)
    +Y → 90° E         -X → 90° E
    +Z → North Pole     +Z → North Pole

    Args:
        x: ECEF X coordinate (towards Prime Meridian)
        y: ECEF Y coordinate (towards 90° East)
        z: ECEF Z coordinate (towards North Pole)

    Returns:
        Tuple (globe_x, globe_y, globe_z) in globe coordinate system
    """
    return (-y, x, z)


class VisualizationTabMixin:
    """Builds and controls the visualization tab and animations."""

    def _build_visualization_tab(self) -> QWidget:
        """Create the visualization tab with pass selection and globe view."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        tab_layout.addWidget(splitter)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        instructions = QLabel(
            "Select a station tab to browse its passes. Passes are sorted by max elevation."
        )
        instructions.setWordWrap(True)
        left_layout.addWidget(instructions)
        frame_toggle_row = QHBoxLayout()
        frame_toggle_row.addWidget(QLabel("Reference frame:"))
        self.visual_frame_combo = QComboBox()
        self.visual_frame_combo.addItems(["ECI (inertial)", "ECEF (earth-fixed)"])
        self.visual_frame_combo.setCurrentIndex(0)
        self.visual_frame_combo.currentIndexChanged.connect(
            self._handle_visual_frame_changed
        )  # type: ignore[attr-defined]
        frame_toggle_row.addWidget(self.visual_frame_combo, stretch=1)
        left_layout.addLayout(frame_toggle_row)
        self.visual_station_tabs = QTabWidget()
        self.visual_station_tabs.setTabPosition(QTabWidget.TabPosition.West)
        self.visual_station_tabs.setDocumentMode(True)
        left_layout.addWidget(self.visual_station_tabs, stretch=1)
        splitter.addWidget(left_panel)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        globe_panel = self._build_visual_globe_panel()
        right_layout.addWidget(globe_panel, stretch=1)
        controls_row = QHBoxLayout()
        self.visual_play_button = QPushButton("Play")
        self.visual_play_button.setEnabled(False)
        self.visual_play_button.clicked.connect(self._toggle_visualization_playback)  # type: ignore[attr-defined]
        controls_row.addWidget(self.visual_play_button)
        controls_row.addWidget(QLabel("Speed"))
        self.visual_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.visual_speed_slider.setRange(25, 300)
        self.visual_speed_slider.setValue(100)
        self.visual_speed_slider.setFixedWidth(120)
        self.visual_speed_slider.valueChanged.connect(
            self._handle_visual_speed_changed
        )  # type: ignore[attr-defined]
        controls_row.addWidget(self.visual_speed_slider)
        self.visual_time_slider = QSlider(Qt.Orientation.Horizontal)
        self.visual_time_slider.setRange(0, 0)
        self.visual_time_slider.setEnabled(False)
        self.visual_time_slider.sliderMoved.connect(
            self._handle_visualization_slider_moved
        )  # type: ignore[attr-defined]
        controls_row.addWidget(self.visual_time_slider, stretch=1)
        self.visual_pass_status_label = QLabel("Run an analysis to populate passes.")
        self.visual_pass_status_label.setWordWrap(True)
        controls_row.addWidget(self.visual_pass_status_label, stretch=1)
        right_layout.addLayout(controls_row)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        self._refresh_visualization_pass_tabs(None)
        return tab

    def _build_visual_globe_panel(self) -> QWidget:
        """Create the visualization globe panel showing ground tracks."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.visual_globe_widget = GlobeWidget()
        self.visual_globe_widget.reset_camera()
        self._update_visual_earth_rotation(None)
        layout.addWidget(self.visual_globe_widget, stretch=1)
        brightness_row = QHBoxLayout()
        brightness_row.addWidget(QLabel("Sun brightness"))
        self.visual_brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.visual_brightness_slider.setRange(20, 300)
        self.visual_brightness_slider.setValue(100)
        self.visual_brightness_slider.setSingleStep(5)
        self.visual_brightness_slider.valueChanged.connect(
            self._handle_visual_brightness_changed
        )  # type: ignore[attr-defined]
        brightness_row.addWidget(self.visual_brightness_slider, stretch=1)
        layout.addLayout(brightness_row)
        self._handle_visual_brightness_changed(self.visual_brightness_slider.value())
        return panel

    def _build_visualization_placeholder_panel(self) -> QWidget:
        """Show a hint in the Mission tab pointing to the Visualization tab."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        label = QLabel(
            "Open the Visualization tab to explore animated passes on the globe."
        )
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignCenter)
        return panel

    def _refresh_visualization_pass_tabs(self, result) -> None:
        """Populate the per-station pass tabs used for visualization playback."""
        if self.visual_station_tabs is None:
            return
        self.visual_station_tabs.blockSignals(True)
        self.visual_station_tabs.clear()
        self._visual_station_lists = {}
        if result is None or not result.passes:
            placeholder = QWidget()
            placeholder_layout = QVBoxLayout(placeholder)
            label = QLabel("Run the analysis to see pass animations.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            placeholder_layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignCenter)
            self.visual_station_tabs.addTab(placeholder, "Stations")
            self.visual_station_tabs.blockSignals(False)
            self._clear_visualization_display("Run an analysis to populate passes.")
            self._update_visual_frame_label()
            return
        station_groups: dict[str, list[PassStatistic]] = {}
        for item in result.passes:
            name = item.station_name or "Ground Station"
            station_groups.setdefault(name, []).append(item)
        self._compute_visualization_reference_frames(result)
        station_records: list[tuple[str, float, list[PassStatistic]]] = []
        for name, items in station_groups.items():
            sorted_items = sorted(
                items, key=lambda p: p.max_elevation_deg, reverse=True
            )
            max_elev = sorted_items[0].max_elevation_deg if sorted_items else 0.0
            station_records.append((name, max_elev, sorted_items))
        station_records.sort(key=lambda entry: entry[1], reverse=True)
        for station_name, _, passes in station_records:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            list_widget = QListWidget()
            list_widget.setProperty("station_name", station_name)
            list_widget.itemSelectionChanged.connect(self._handle_visual_pass_selection)  # type: ignore[attr-defined]
            for idx, pass_stat in enumerate(passes, start=1):
                text = (
                    f"{idx:02d}. {pass_stat.aos:%d-%b %H:%M:%S} UTC  |  "
                    f"Max {pass_stat.max_elevation_deg:.1f}°  |  "
                    f"{pass_stat.duration_minutes:.1f} min"
                )
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, pass_stat)
                list_widget.addItem(item)
            tab_layout.addWidget(list_widget)
            self.visual_station_tabs.addTab(tab, station_name)
            self._visual_station_lists[station_name] = list_widget
        self.visual_station_tabs.blockSignals(False)
        if station_records:
            first_station = station_records[0][0]
            first_list = self._visual_station_lists.get(first_station)
            if first_list and first_list.count() > 0:
                first_list.setCurrentRow(0)
        self._update_visual_frame_label()

    def _compute_visualization_reference_frames(self, result) -> None:
        """Initialize reference epoch information for visualization transforms."""
        if self._current_config is not None:
            self._visual_reference_epoch = self._current_config.scenario.start_time
        elif result.passes:
            self._visual_reference_epoch = result.passes[0].aos
        elif result.ground_track:
            self._visual_reference_epoch = result.ground_track[0].timestamp

    def _handle_visual_pass_selection(self) -> None:
        """React to pass selection within a station tab."""
        widget = self.sender()
        if not isinstance(widget, QListWidget):
            return
        item = widget.currentItem()
        if item is None:
            return
        pass_stat = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(pass_stat, PassStatistic):
            self._load_visualization_pass(pass_stat)

    def _handle_visual_frame_changed(self, index: int) -> None:
        """Switch between ECI and ECEF rendering modes."""
        mode = "ECI" if index == 0 else "ECEF"
        if mode == self._visual_frame_mode:
            return
        self._visual_frame_mode = mode
        if self._visual_animation_timer.isActive():
            self._visual_animation_timer.stop()
            if self.visual_play_button:
                self.visual_play_button.setText("Play")
        self._rebuild_visualization_scene()
        self._update_visual_frame_label()

    def _handle_visual_brightness_changed(self, value: int) -> None:
        """Update sun brightness multiplier from the UI slider."""
        if self.visual_globe_widget is None:
            return
        intensity = max(0.05, value / 100.0)
        self.visual_globe_widget.set_sun_brightness(intensity)

    def _rebuild_visualization_scene(self, *, force: bool = False) -> None:
        """Reapply the visualization assets for the current frame mode."""
        _ = force  # force is kept for backwards compatibility with older calls
        if self.visual_globe_widget is None:
            return
        self._update_visual_earth_rotation(None)
        if self._visual_pass_track is not None:
            self._update_visualization_pass_geometry(self._visual_pass_track)
            self._update_visualization_frame()

    def _update_visual_frame_label(self) -> None:
        """Show the active frame mode in the status label."""
        if not self.visual_pass_status_label:
            return
        prefix = "ECI" if self._visual_frame_mode == "ECI" else "ECEF"
        text = self.visual_pass_status_label.text()
        if " | " in text:
            text = text.split(" | ", 1)[1]
        self.visual_pass_status_label.setText(f"{prefix} | {text}")

    def _load_visualization_pass(self, pass_stat: PassStatistic) -> None:
        """Prepare the globe animation for the selected pass."""
        if self._visual_animation_timer.isActive():
            self._visual_animation_timer.stop()
        track = self._extract_pass_track(pass_stat)
        if not track:
            self._clear_visualization_display("No track samples for the selected pass.")
            return
        self._visual_selected_pass = pass_stat
        self._visual_pass_track = track
        self._visual_animation_index = 0
        self._visual_anim_fraction = 0.0
        self._visual_contact_window = (pass_stat.aos, pass_stat.los)
        self._visual_station_ecef = self._resolve_station_coordinates(pass_stat)
        self._update_visualization_pass_geometry(track)
        slider_max = max(len(track) - 1, 0)
        if self.visual_time_slider:
            self.visual_time_slider.setEnabled(slider_max > 0)
            self.visual_time_slider.setRange(0, slider_max)
            self.visual_time_slider.setPageStep(max(1, slider_max // 20 or 1))
            self.visual_time_slider.setValue(0)
        if track:
            self.visual_globe_widget.set_sun_datetime(track[0].timestamp)
        if self.visual_play_button:
            self.visual_play_button.setEnabled(True)
            self.visual_play_button.setText("Play")
        if self.visual_pass_status_label:
            station_name = pass_stat.station_name or "Ground Station"
            self.visual_pass_status_label.setText(
                f"{self._visual_frame_mode} | {station_name}: {pass_stat.aos:%d-%b %H:%M:%S} → "
                f"{pass_stat.los:%H:%M:%S} UTC (max {pass_stat.max_elevation_deg:.1f}°)"
            )
        self._update_visualization_frame()

    def _extract_pass_track(self, pass_stat: PassStatistic) -> list[GroundTrackPoint]:
        """Return the subset of ground-track points covering the selected pass."""
        if self._last_result is None or not self._last_result.ground_track:
            return []
        pad = timedelta(minutes=2)
        start = pass_stat.aos - pad
        end = pass_stat.los + pad
        segment = [
            point
            for point in self._last_result.ground_track
            if start <= point.timestamp <= end
        ]
        return segment or list(self._last_result.ground_track)

    def _resolve_station_coordinates(
        self, pass_stat: PassStatistic
    ) -> tuple[float, float, float] | None:
        """Resolve the ECEF coordinates for the pass station."""
        station: GroundStationConfig | None = None
        station_name = pass_stat.station_name or ""
        if station_name:
            station = self._active_station_lookup.get(station_name)
            if station is None:
                station = next(
                    (s for s in self._station_presets if s.name == station_name),
                    None,
                )
        if station is None and self._station_presets:
            station = self._station_presets[0]
        if station is None:
            return None
        return self._station_to_ecef_km(station)

    def _station_to_ecef_km(
        self, station: GroundStationConfig
    ) -> tuple[float, float, float]:
        """Convert a ground-station lat/lon/alt to ECEF coordinates in kilometers."""
        lat = math.radians(station.latitude_deg)
        lon = math.radians(station.longitude_deg)
        alt_km = station.altitude_m / 1000.0
        a = 6378.137
        e2 = 6.69437999014e-3
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
        x = (N + alt_km) * math.cos(lat) * math.cos(lon)
        y = (N + alt_km) * math.cos(lat) * math.sin(lon)
        z = (N * (1 - e2) + alt_km) * sin_lat
        return ecef_to_globe_coords(x, y, z)

    def _seconds_since_reference(self, timestamp: datetime) -> float:
        """Return elapsed seconds from the visualization reference epoch."""
        ref = self._visual_reference_epoch
        if ref is None:
            return 0.0
        return (timestamp - ref).total_seconds()

    def _convert_vector_to_current_frame(
        self, vector: tuple[float, float, float], timestamp: datetime
    ) -> tuple[float, float, float]:
        """Convert an ECEF vector into the currently selected frame."""
        if self._visual_frame_mode != "ECI":
            return vector
        delta = self._seconds_since_reference(timestamp)
        angle = EARTH_ROTATION_RATE_RAD_PER_SEC * delta
        return rotate_vector_z(vector, angle)

    def _get_visualization_track_coords(
        self, track: list[GroundTrackPoint]
    ) -> np.ndarray:
        """Return track coordinates transformed into the current frame."""
        coords = np.array(
            [ecef_to_globe_coords(pt.x_km, pt.y_km, pt.z_km) for pt in track],
            dtype=float,
        )
        if coords.size == 0 or self._visual_frame_mode != "ECI":
            return coords
        times = np.array(
            [self._seconds_since_reference(pt.timestamp) for pt in track], dtype=float
        )
        angles = EARTH_ROTATION_RATE_RAD_PER_SEC * times
        cos_ang = np.cos(angles)
        sin_ang = np.sin(angles)
        x = coords[:, 0].copy()
        y = coords[:, 1].copy()
        coords[:, 0] = cos_ang * x - sin_ang * y
        coords[:, 1] = sin_ang * x + cos_ang * y
        return coords

    def _update_visualization_pass_geometry(
        self, track: list[GroundTrackPoint]
    ) -> None:
        """Render the selected pass path and initialize the satellite marker."""
        if self.visual_globe_widget is None or not track:
            return
        coords = self._get_visualization_track_coords(track)
        if coords.size == 0:
            return
        first_timestamp = track[0].timestamp if track else None
        self._update_visual_earth_rotation(first_timestamp)
        if first_timestamp:
            self.visual_globe_widget.set_sun_datetime(first_timestamp)
        self.visual_globe_widget.update_track(coords)
        self.visual_globe_widget.update_satellite_position(tuple(coords[0]))
        self.visual_globe_widget.update_link_segment(None, None)
        self.visual_globe_widget.update_direction_arrow(None, None)

    def _clear_visualization_display(self, message: str | None = None) -> None:
        """Reset visualization playback state and remove temporary actors."""
        if self._visual_animation_timer.isActive():
            self._visual_animation_timer.stop()
        if self.visual_globe_widget is not None:
            self.visual_globe_widget.update_track(None)
            self.visual_globe_widget.update_satellite_position(None)
            self.visual_globe_widget.update_link_segment(None, None)
            self.visual_globe_widget.update_direction_arrow(None, None)
            self.visual_globe_widget.set_sun_datetime(None)
            self.visual_globe_widget.update_direction_arrow(None, None)
        self._visual_pass_track = None
        self._visual_selected_pass = None
        self._visual_station_ecef = None
        self._visual_contact_window = None
        self._visual_animation_index = 0
        self._visual_anim_fraction = 0.0
        if self.visual_time_slider:
            self.visual_time_slider.setEnabled(False)
            self.visual_time_slider.setRange(0, 0)
            self.visual_time_slider.setValue(0)
        if self.visual_play_button:
            self.visual_play_button.setEnabled(False)
            self.visual_play_button.setText("Play")
        if message and self.visual_pass_status_label:
            self.visual_pass_status_label.setText(message)

    def _toggle_visualization_playback(self) -> None:
        """Play or pause the current pass animation."""
        if not self._visual_pass_track or self.visual_play_button is None:
            return
        if self._visual_animation_timer.isActive():
            self._visual_animation_timer.stop()
            self.visual_play_button.setText("Play")
            return
        if self._visual_animation_index >= len(self._visual_pass_track) - 1:
            self._visual_animation_index = 0
            self._visual_anim_fraction = 0.0
            self._update_visualization_frame(render=False)
        else:
            self._visual_anim_fraction = 0.0
        self._visual_animation_timer.start()
        self.visual_play_button.setText("Pause")

    def _advance_visualization_animation(self) -> None:
        """Advance the satellite along the pass during playback."""
        if not self._visual_pass_track:
            self._stop_visualization_animation()
            return
        if self._visual_animation_index >= len(self._visual_pass_track) - 1:
            self._stop_visualization_animation(final=True)
            return
        step = self._visual_base_step * self._visual_speed_multiplier
        self._visual_anim_fraction += step
        while (
            self._visual_anim_fraction >= 1.0
            and self._visual_animation_index < len(self._visual_pass_track) - 1
        ):
            self._visual_anim_fraction -= 1.0
            self._visual_animation_index += 1
        if self._visual_animation_index >= len(self._visual_pass_track) - 1:
            self._visual_animation_index = len(self._visual_pass_track) - 1
            self._visual_anim_fraction = 0.0
            self._update_visualization_frame()
            self._stop_visualization_animation(final=True)
            return
        self._update_visualization_frame()

    def _stop_visualization_animation(self, final: bool = False) -> None:
        """Stop playback and update control labels."""
        if self._visual_animation_timer.isActive():
            self._visual_animation_timer.stop()
        if self.visual_play_button:
            self.visual_play_button.setText("Replay" if final else "Play")

    def _handle_visualization_slider_moved(self, value: int) -> None:
        """Scrub through the pass timeline."""
        if not self._visual_pass_track:
            return
        clamped = max(0, min(value, len(self._visual_pass_track) - 1))
        self._visual_animation_index = clamped
        self._visual_anim_fraction = 0.0
        if self._visual_animation_timer.isActive():
            self._visual_animation_timer.stop()
            if self.visual_play_button:
                self.visual_play_button.setText("Play")
        self._update_visualization_frame()

    def _handle_visual_speed_changed(self, value: int) -> None:
        """Adjust playback speed multiplier."""
        self._visual_speed_multiplier = max(0.1, value / 100.0)

    def _update_visualization_frame(self, render: bool = True) -> None:
        """Render the satellite position, slider, and ground link for the current frame."""
        _ = render
        if (
            not self._visual_pass_track
            or self.visual_globe_widget is None
            or self._visual_animation_index >= len(self._visual_pass_track)
        ):
            return
        point = self._visual_pass_track[self._visual_animation_index]
        base_vec = np.array(
            self._convert_vector_to_current_frame(
                ecef_to_globe_coords(point.x_km, point.y_km, point.z_km),
                point.timestamp,
            ),
            dtype=float,
        )
        timestamp = point.timestamp
        coords_vec = base_vec
        if (
            self._visual_anim_fraction > 1e-4
            and self._visual_animation_index < len(self._visual_pass_track) - 1
        ):
            next_point = self._visual_pass_track[self._visual_animation_index + 1]
            next_vec = np.array(
                self._convert_vector_to_current_frame(
                    ecef_to_globe_coords(next_point.x_km, next_point.y_km, next_point.z_km),
                    next_point.timestamp,
                ),
                dtype=float,
            )
            alpha = self._visual_anim_fraction
            coords_vec = (1.0 - alpha) * base_vec + alpha * next_vec
            dt = next_point.timestamp - point.timestamp
            timestamp = point.timestamp + alpha * dt
        coords = tuple(coords_vec.tolist())
        self._update_visual_earth_rotation(timestamp)
        if self.visual_globe_widget:
            self.visual_globe_widget.set_sun_datetime(timestamp)
        self.visual_globe_widget.update_satellite_position(coords)
        if self.visual_time_slider:
            self.visual_time_slider.blockSignals(True)
            self.visual_time_slider.setValue(self._visual_animation_index)
            self.visual_time_slider.blockSignals(False)
        if self.visual_pass_status_label and self._visual_selected_pass is not None:
            station_name = self._visual_selected_pass.station_name or "Ground Station"
            self.visual_pass_status_label.setText(
                f"{self._visual_frame_mode} | {station_name}: {timestamp:%d-%b %H:%M:%S} UTC"
            )
        self._update_visualization_link_actor(timestamp, coords)

    def _get_station_position(
        self, timestamp: datetime
    ) -> tuple[float, float, float] | None:
        """Return the station position vector in the current frame."""
        if self._visual_station_ecef is None:
            return None
        return self._convert_vector_to_current_frame(
            self._visual_station_ecef, timestamp
        )

    def _update_visual_earth_rotation(self, timestamp: datetime | None) -> None:
        """Rotate the globe actors to match Earth orientation in the selected frame."""
        if self._visual_frame_mode != "ECI":
            angle_deg = 0.0
        elif timestamp is None:
            angle_deg = 0.0
        else:
            delta = self._seconds_since_reference(timestamp)
            angle_deg = math.degrees(EARTH_ROTATION_RATE_RAD_PER_SEC * delta)
        widget = getattr(self, "visual_globe_widget", None)
        if widget is None:
            return
        widget.set_frame_rotation(self._visual_frame_mode, angle_deg)

    def _update_visualization_link_actor(
        self, timestamp: datetime, satellite_coords: tuple[float, float, float]
    ) -> None:
        """Show or hide the green contact link for the current frame."""
        widget = getattr(self, "visual_globe_widget", None)
        if widget is None:
            return
        if self._visual_contact_window is None or self._visual_station_ecef is None:
            widget.update_link_segment(None, None)
            return
        aos, los = self._visual_contact_window
        if not (aos <= timestamp <= los):
            widget.update_link_segment(None, None)
            return
        station_point = self._get_station_position(timestamp)
        if station_point is None:
            return
        widget.update_link_segment(station_point, satellite_coords)

