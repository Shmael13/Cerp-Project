from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd

from cerp_viz.core.models import AssumptionSpec, BuildResult, ColumnSpec


class BaseVisualization(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    supports_comparison: ClassVar[bool] = False  # opt-in per chart

    @abstractmethod
    def required_columns(self) -> list[ColumnSpec]: ...

    @abstractmethod
    def assumptions(self) -> list[AssumptionSpec]: ...

    @abstractmethod
    def build(
        self,
        df: pd.DataFrame,
        columns: dict[str, str | None],
        params: dict[str, Any],
    ) -> BuildResult:
        """Return a BuildResult with the Plotly Figure and any data-transformation warnings."""
        ...

    def compare(
        self,
        df: pd.DataFrame,
        columns: dict[str, str | None],
        scenarios: dict[str, dict[str, Any]],
    ) -> BuildResult:
        """Overlay multiple named assumption sets on one figure.
        Override in subclasses and set supports_comparison = True to enable."""
        raise NotImplementedError(f"'{self.name}' does not support scenario comparison.")


class BaseLoader(ABC):
    @abstractmethod
    def accepts(self, source: Any) -> bool: ...

    @abstractmethod
    def load(self, source: Any) -> dict[str, pd.DataFrame]:
        """Return a mapping of sheet/table name → DataFrame."""
        ...
