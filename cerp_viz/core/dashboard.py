from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SlotConfig:
    """One chart slot in a dashboard grid."""
    chart_name: str
    columns: dict[str, str | None]
    params: dict[str, Any]
    title: str = ""


@dataclass
class DashboardConfig:
    """Ordered list of chart slots that make up a dashboard layout."""
    slots: list[SlotConfig] = field(default_factory=list)

    def add(self, slot: SlotConfig) -> None:
        self.slots.append(slot)

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.slots):
            self.slots.pop(index)

    def clear(self) -> None:
        self.slots.clear()

    def count(self) -> int:
        return len(self.slots)
