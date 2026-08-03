"""B1-deep — тот же ReAct-цикл, но web_search читает страницы целиком.

Отличие от b1_search единственное: инструмент возвращает реранкованные
фрагменты полных текстов (SearXNG -> trafilatura -> bge-reranker-v2-m3),
а не сырые сниппеты. Поиск доступен на всех предметах — прямой тест
гипотезы «качественный поиск не должен ухудшать метрики».
"""

from ..tools.deep_search import deep_search
from .b1_search import B1Search


class B1Deep(B1Search):
    condition = "b1_deep"

    def _search(self, args: dict) -> str:
        return deep_search(self.settings, str(args["query"]))
