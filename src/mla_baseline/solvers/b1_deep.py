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

    def _run_tool(self, name: str, args: dict, seen: set[str]) -> str:
        if name != "web_search":
            return f"Bilinmeyen araç: {name}"
        query = str(args.get("query") or "").strip()
        if not query:
            return "Boş sorgu. query parametresini doldur veya aramadan çöz."
        if query.casefold() in seen:
            return ("Bu sorguyu zaten yaptın, sonuçlar yukarıda. "
                    "Yeni arama yapma; mevcut bilgiyle çözümü tamamla.")
        seen.add(query.casefold())
        return deep_search(self.settings, query)
