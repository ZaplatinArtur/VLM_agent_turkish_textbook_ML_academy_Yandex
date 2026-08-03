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

    # дисциплина цикла (лимит, дедуп, реакция на несуществующий тул) — в B1Search;
    # здесь только сам источник и ключ дедупликации с учётом фильтров
    tool_name = "search_textbooks"

    def _dedup_key(self, args: dict) -> str:
        return json.dumps({"q": str(args["query"]).casefold(), "s": args.get("subject"),
                           "g": args.get("grade"), "m": args.get("mode")},
                          ensure_ascii=False, sort_keys=True)

    def _max_calls(self) -> int:
        return self.settings.rag_max_calls

    def _search(self, args: dict) -> str:
        return textbook_search(
            self.settings, str(args["query"]),
            top_k=args.get("top_k"),
            subject=args.get("subject"),
            grade=args.get("grade"),
            mode=str(args.get("mode") or "or"),
        )
