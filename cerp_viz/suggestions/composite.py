"""
CompositeSuggester: chains multiple BaseSuggester implementations.

Merges results by keeping the highest-scored suggestion per chart type,
then re-sorts globally.  Statistical analysis wins over rule-based heuristics
when it finds stronger evidence; rule-based fills chart types statistical missed.

Self-registers as "Smart (Statistical + Rule-Based)" in the suggester registry.
"""
from __future__ import annotations

import pandas as pd

from cerp_viz.core.suggestions import BaseSuggester, SuggestionResult
from cerp_viz.suggestions._utils import validate_and_complete


class CompositeSuggester(BaseSuggester):
    """
    Chains an ordered list of BaseSuggester implementations.
    For each chart type, keeps whichever sub-suggester produced the highest score.
    A failure in any sub-suggester is silently skipped.
    """

    def __init__(self, suggesters: list[BaseSuggester]) -> None:
        self._suggesters = suggesters

    def suggest(self, df: pd.DataFrame, query: str = "") -> list[SuggestionResult]:
        best: dict[str, SuggestionResult] = {}

        for suggester in self._suggesters:
            try:
                for r in suggester.suggest(df, query=query):
                    v = validate_and_complete(r, df)
                    if v is None:
                        continue
                    existing = best.get(v.chart_name)
                    if existing is None or v.score > existing.score:
                        best[v.chart_name] = v
            except Exception:
                pass

        return sorted(best.values(), key=lambda r: r.score, reverse=True)


# ── Self-registration ─────────────────────────────────────────────────────────

from cerp_viz.suggestions.registry import register  # noqa: E402
from cerp_viz.suggestions.statistical import StatisticalSuggester  # noqa: E402
from cerp_viz.suggestions.rule_based import RuleBasedSuggester    # noqa: E402


def _make_smart() -> BaseSuggester:
    return CompositeSuggester([StatisticalSuggester(), RuleBasedSuggester()])


register("Smart (Statistical + Rule-Based)", _make_smart)
