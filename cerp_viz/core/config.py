from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ChartConfig:
    """Serialisable snapshot of a chart configuration including saved scenarios."""
    chart_name: str
    columns: dict[str, str | None]
    params: dict[str, Any]
    scenarios: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, raw: str) -> ChartConfig:
        data = json.loads(raw)
        return cls(**data)
