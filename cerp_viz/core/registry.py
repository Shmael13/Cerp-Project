from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cerp_viz.core.base import BaseVisualization


class VisualizationRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, type[BaseVisualization]] = {}

    def register(self, cls: type[BaseVisualization]) -> type[BaseVisualization]:
        self._registry[cls.name] = cls
        return cls

    def get(self, name: str) -> type[BaseVisualization]:
        if name not in self._registry:
            raise KeyError(f"Visualization '{name}' not found. Available: {list(self._registry)}")
        return self._registry[name]

    def all(self) -> dict[str, type[BaseVisualization]]:
        return dict(self._registry)

    def names(self) -> list[str]:
        return list(self._registry.keys())


registry = VisualizationRegistry()
