"""Routed HTTP textbook-search condition built on the shared ReAct loop."""

import json

from ..config import Settings
from ..tools.textbook import TEXTBOOK_TOOL_SCHEMA, textbook_search
from .b1_routed import B1Routed


class AgentRagRouted(B1Routed):
    condition = "agent_rag_routed"
    tool_note_key = "agent_tool_note"
    tool_name = "search_textbooks"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.llm_tools = self.llm.bind(tools=[TEXTBOOK_TOOL_SCHEMA])

    def _dedup_key(self, args: dict) -> str:
        return json.dumps(
            {
                "q": str(args["query"]).casefold(),
                "s": args.get("subject"),
                "g": args.get("grade"),
                "m": args.get("mode"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _max_calls(self) -> int:
        return self.settings.rag_max_calls

    def _search(self, args: dict) -> str:
        return textbook_search(
            self.settings,
            str(args["query"]),
            top_k=args.get("top_k"),
            subject=args.get("subject"),
            grade=args.get("grade"),
            mode=str(args.get("mode") or "or"),
        )
