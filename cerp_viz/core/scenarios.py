from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scenario:
    name: str
    params: dict[str, Any]
    sheet_name: str = ""
    col_mapping: dict = field(default_factory=dict)


class ScenarioStore:
    """Pure-Python, UI-agnostic store for named assumption sets.
    The UI layer is responsible for persisting an instance in session state."""

    def __init__(self) -> None:
        self._store: dict[str, Scenario] = {}

    def save(
        self,
        name: str,
        params: dict[str, Any],
        sheet_name: str = "",
        col_mapping: dict | None = None,
    ) -> None:
        self._store[name] = Scenario(
            name=name,
            params=dict(params),
            sheet_name=sheet_name,
            col_mapping=dict(col_mapping or {}),
        )

    def get(self, name: str) -> Scenario | None:
        return self._store.get(name)

    def delete(self, name: str) -> None:
        self._store.pop(name, None)

    def clear(self) -> None:
        self._store.clear()

    def all(self) -> dict[str, Scenario]:
        return dict(self._store)

    def names(self) -> list[str]:
        return list(self._store.keys())

    def count(self) -> int:
        return len(self._store)
