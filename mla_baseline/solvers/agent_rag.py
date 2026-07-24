"""agent_rag — модель + поиск по корпусу учебников (BM25-стек команды ретрива).

Каркас (ReAct-цикл, финализация, трейсинг) полностью от B1Search — меняется
только инструмент: вместо веб-поиска search_textbooks по контракту
retrieval_tool_contract.md. Так сравнение условий остаётся честным.
"""

import json

from ..config import Settings
from ..tools.textbook import TEXTBOOK_TOOL_SCHEMA, textbook_search
from .b1_search import B1Search


class AgentRag(B1Search):
    condition = "agent_rag"
    tool_note_key = "agent_tool_note"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.llm_tools = self.llm.bind(tools=[TEXTBOOK_TOOL_SCHEMA])

    def _run_tool(self, name: str, args: dict, seen: set[str]) -> str:
        if name != "search_textbooks":
            return f"Bilinmeyen araç: {name}"
        query = str(args.get("query") or "").strip()
        if not query:
            return "Boş sorgu. query parametresini doldur veya aramadan çöz."
        key = json.dumps({"q": query.casefold(), "s": args.get("subject"),
                          "g": args.get("grade"), "m": args.get("mode")},
                         ensure_ascii=False, sort_keys=True)
        if key in seen:
            return ("Bu sorguyu zaten yaptın, sonuçlar yukarıda. "
                    "Yeni arama yapma; mevcut bilgiyle çözümü tamamla.")
        if len(seen) >= self.settings.rag_max_calls:
            return ("Arama limitine ulaştın. Eldeki kanıtlarla çözümü tamamla.")
        seen.add(key)
        return textbook_search(
            self.settings, query,
            top_k=args.get("top_k"),
            subject=args.get("subject"),
            grade=args.get("grade"),
            mode=str(args.get("mode") or "or"),
        )
