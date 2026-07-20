from abc import ABC, abstractmethod

from ..config import Settings
from ..contracts import Task
from ..schemas import SolveResult


class Solver(ABC):
    """Общий интерфейс трёх условий сравнения (b0 / b1 / agent)."""

    condition: str

    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    def solve(self, task: Task) -> SolveResult: ...

    @abstractmethod
    def build_messages(self, task: Task) -> list:
        """Отдельно от solve — чтобы проверять вход без вызова модели (--dry-run)."""
