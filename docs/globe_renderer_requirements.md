# ModernGL Globe Renderer Requirements

This document inventories the features currently provided by the PyVista/VTK
implementation so the ModernGL migration can reach feature parity (or exceed it).

## Shared globe capabilities (`src/ui/globe_scene.py`)

- Load reusable Earth geometry (6,378 km radius sphere, 180×90 resolution) with the
  option to fall back to locally bundled textures: `8k_earth_daymap`, `8k_earth_clouds`,
  and `8k_stars_milky_way`.
- Generate cloud alpha channel procedurally by computing luminance from the RGB map.
- Build actors for:
  - Textured Earth surface, optionally tinted if textures fail to load.
  - Slightly scaled cloud layer with transparency (opacity 0.85).
  - Inverted starfield sphere (360×180 resolution) textured on the inside surface.
- Maintain camera state (`_globe_camera_radius_km`, `_globe_camera_height_km`,
  `_globe_view_angle`) and enforce constrained interaction via `_lock_globe_camera`.
  Camera defaults: position `(0, -horizontal, height)`, up `(0, 0, 1)`, look-at `(0,0,0)`.
- Register PyVista interactor observers to re-lock the camera when the user finishes
  mouse interaction.
- Provide rotation helpers (`_set_actor_rotation`, `_rotate_vector_z`) used by both
  mission and visualization tabs to keep Earth/cloud textures synced with the chosen
  reference frame (ECI/ECEF).

## Mission tab usage (`src/ui/tabs/mission_tab.py`)

- Embeds a `QtInteractor` and keeps references to PyVista actors:
  `_mission_earth_rotation_actor`, `_mission_cloud_rotation_actor`,
  `_mission_track_actor`, `_mission_terminal_actor`.
- Rebuilds globe scene on startup and whenever the frame mode switches between
  ECI/ECEF.
- Draws a satellite ground track spline (`pv.Spline`) rendered in red with width 3.
- Highlights the terminal point using a small sphere at the final track coordinate.
- Refreshes the track when:
  - A new analysis result arrives (`_refresh_mission_globe`).
  - The orbit slider changes (ECEF mode only) after a debounce timer.
- Applies camera locking in `_lock_mission_globe_camera` after each geometry update.
- Rotates Earth/cloud actors via `_update_mission_earth_rotation`, using either the
  reference epoch time delta (ECI) or slider-derived offset (ECEF).
- Requires ability to remove/recreate meshes efficiently (`plotter.remove_actor` +
  `render=False` usage to avoid redundant draws until geometry complete).

## Visualization tab usage (`src/ui/tabs/visualization_tab.py`)

- Maintains separate ModernGL-equivalent actors for:
  `_visual_earth_rotation_actor`, `_visual_cloud_rotation_actor`,
  `_visual_pass_path_actor`, `_visual_sat_actor`, `_visual_link_actor`.
- Needs to:
  - Render a cyan pass path spline (high-resolution smoothing).
  - Render a bright yellow satellite marker sphere that can be repositioned every
    animation frame (roughly 13 Hz based on 75 ms timer).
  - Draw/clear a green station-to-satellite line segment only during contact windows.
- Supports both ECI and ECEF visualization:
  - In ECI mode, applies time-based Earth rotation and track coordinate transforms.
  - In ECEF mode, no Earth rotation is applied; vectors are used directly.
- Offers playback controls (play/pause, slider scrubbing) that require
  re-rendering the scene for each sample while keeping UI responsive.
- Removes actors cleanly when switching passes or clearing state, without leaking
  GPU resources.

## Asset & interaction requirements

- Texture files reside under `resources/textures`. The renderer should lazily load
  them (falling back gracefully if missing) and reuse GPU textures across views.
- The mission and visualization widgets can share the same renderer implementation,
  but each needs isolated GL state because they are separate widgets.
- Camera interactions must stay constrained to polar rotations and maintain a
  consistent field of view comparable to the current PyVista setup.
- Rendering must remain performant with frequent updates (animation loop, orbit
  slider refreshes) and should support vsync-less redraws to target ~60 FPS.

