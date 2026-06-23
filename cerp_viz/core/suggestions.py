from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class SuggestionResult:
    """One ranked visualization suggestion produced by any BaseSuggester."""
    chart_name: str                      # must match a key in the registry
    columns: dict[str, str | None]       # role → column name (None = optional, not used)
    params: dict[str, Any]               # assumption key → value (overrides defaults)
    title: str                           # short human-readable headline
    rationale: str                       # one sentence explaining why this is interesting
    score: float = 0.0                   # 0.0–1.0; higher = more confident / interesting
    # Optional transforms to apply before building this chart.
    # Each entry: {"type": "derived"|"filter"|"date_part", ...fields}
    transforms: list[dict[str, Any]] = field(default_factory=list)


class BaseSuggester(ABC):
    """
    Plug-in interface for chart suggestion engines.

    Implementations must never raise — return an empty list on failure so the
    caller can fall back gracefully.  Results should be ordered by descending
    score, but the caller will re-sort anyway.
    """

    @abstractmethod
    def suggest(self, df: pd.DataFrame, query: str = "") -> list[SuggestionResult]:
        """Analyse df and return ranked suggestions. Must not raise.

        query — optional free-text instruction from the user, e.g.
                 "best heatmaps" or "what drives revenue?"
                 Engines may use this to focus or filter results.
        """
        ...
