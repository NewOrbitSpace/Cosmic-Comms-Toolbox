# Ground Station Access GUI

PySide6 desktop application that wraps the Orekit-based ground-station access
analysis. The UI allows you to configure the ground station, satellite orbital
elements, propagator, and scenario dates, then shows a statistics table of the
resulting passes.

## Project Layout

```
.
├── resources/          # Placeholder for icons, QSS, etc.
└── src/
    ├── main.py         # Application entry point
    ├── models.py       # Shared dataclasses for configs/results
    ├── services/
    │   └── access_analysis.py
    └── ui/
        └── main_window.py
```

## Coding Standards

- **PEP 8** for formatting, imports, and naming.
- **Google-style docstrings** for all public modules, classes, and functions.
- Inline comments explain the *why* for non-obvious logic; avoid restating code.

## Requirements

Install dependencies into your preferred environment:

```bash
pip install -r requirements.txt
```

The GUI depends on:

- PySide6
- orekit
- numpy
- moderngl (OpenGL 3.3+ driver)

Ensure the `orekitdata` package is available so the embedded data files can be
loaded automatically.

## Running the Application

```bash
python -m src.main
```

Select the desired inputs, click **Run Analysis**, and review the statistics
table at the bottom of the window.

## Ground Station Importer

- Use the **Import stations...** button inside the *Ground Station* group to
  load presets from any CSV or Excel file containing `name`, `latitude`,
  `longitude`, and `altitude` columns.
- A starter dataset is available at
  `resources/sample_data/ground_stations.csv`.
- After importing, picking a preset automatically populates the station fields.

## Visualization

- The right-side panel uses PyQtGraph for interactive plots:
  - A Gantt-style bar view of each access window along the scenario timeline.
  - A pass-duration histogram whose bin width equals twice the configured
    sample time.

## ModernGL Globe

- The Mission and Visualization tabs render their globes with the ModernGL-based
  `GlobeWidget`, which lives in `src/ui/opengl/globe_widget.py`.
- Ensure the host GPU exposes at least OpenGL 3.3. When running headless (CI),
  set `QT_QPA_PLATFORM=offscreen` to let Qt create an off-screen context.
- Feature parity targets are summarized in `docs/globe_renderer_requirements.md`.

## License
This project is licensed under the MIT License — see the LICENSE file for details.
