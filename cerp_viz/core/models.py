from dataclasses import dataclass, field
from typing import Any


@dataclass
class BuildResult:
    """Return value of BaseVisualization.build(). Carries the figure and any
    warnings about data that was transformed or dropped before rendering."""
    figure: Any
    warnings: list[str] = field(default_factory=list)


@dataclass
class ColumnSpec:
    role: str
    dtype: str       # "numeric" | "categorical" | "datetime" | "any"
    label: str
    required: bool = True


@dataclass
class AssumptionSpec:
    key: str
    widget: str      # "slider" | "selectbox" | "number_input" | "multiselect" | "toggle"
    label: str
    default: Any
    options: dict = field(default_factory=dict)
    category: str = "Display"    # "Data" | "Display"
