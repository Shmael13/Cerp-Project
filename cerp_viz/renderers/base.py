from abc import ABC, abstractmethod
from typing import Any


class BaseRenderer(ABC):
    @abstractmethod
    def render(self, figure: Any) -> None: ...
