"""
Suggester registry — mirrors the chart VisualizationRegistry pattern.

Each suggester module calls register() at module level.
suggestions/__init__.py imports all modules to trigger registration.
The UI reads names() and calls get(name) to obtain an instance.
"""
from __future__ import annotations

from typing import Callable

from cerp_viz.core.suggestions import BaseSuggester

# name → zero-arg factory that returns a ready BaseSuggester instance
_REGISTRY: dict[str, Callable[[], BaseSuggester]] = {}


def register(name: str, factory: Callable[[], BaseSuggester]) -> None:
    _REGISTRY[name] = factory


def names() -> list[str]:
    return list(_REGISTRY.keys())


def get(name: str) -> BaseSuggester:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown suggester '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()
