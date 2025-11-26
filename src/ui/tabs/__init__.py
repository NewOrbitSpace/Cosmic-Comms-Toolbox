"""Tab mixins for the main window."""

from .ground_tab import GroundTabMixin
from .link_budget_tab import LinkBudgetTabMixin
from .mission_tab import MissionTabMixin
from .orbit_summary_tab import OrbitSummaryTabMixin
from .thruster_tab import ThrusterTabMixin
from .visualization_tab import VisualizationTabMixin

__all__ = [
    "GroundTabMixin",
    "LinkBudgetTabMixin",
    "MissionTabMixin",
    "OrbitSummaryTabMixin",
    "ThrusterTabMixin",
    "VisualizationTabMixin",
]

