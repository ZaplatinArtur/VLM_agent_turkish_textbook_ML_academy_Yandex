"""B1-routed — поиск по необходимости: роутинг по предмету.

Данные прогонов B0/B1 показали: веб-поиск систематически помогает знаниевым
предметам и систематически вредит вычислительным (математика: −13 пп).
Роутер отключает инструмент там, где он вреден: такие задачи идут чистым
B0-путём (одна модель, тот же промпт), остальные — полным B1-циклом.
"""

from ..config import Settings
from ..contracts import Task
from ..schemas import SolveResult
from .b0_no_tools import B0NoTools
from .b1_search import B1Search


class B1Routed(B1Search):
    condition = "b1_routed"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.no_search = {s.strip().casefold()
                          for s in settings.b1_no_search_subjects.split(",") if s.strip()}

    def _routed_off(self, task: Task) -> bool:
        return task.subject.strip().casefold() in self.no_search

    def build_messages(self, task: Task) -> list:
        # без поиска — и системный промпт без упоминания инструмента
        if self._routed_off(task):
            return B0NoTools.build_messages(self, task)
        return super().build_messages(task)

    def solve(self, task: Task) -> SolveResult:
        if self._routed_off(task):
            return B0NoTools.solve(self, task)  # condition остаётся b1_routed
        return super().solve(task)
