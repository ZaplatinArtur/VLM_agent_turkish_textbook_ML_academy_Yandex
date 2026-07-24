"""agent_rag + роутинг: предметы из MLA_B1_NO_SEARCH_SUBJECTS идут B0-путём.

Композиция та же, что у B1DeepRouted: роутинг из B1Routed, инструмент из
AgentRag (MRO: routed-ветка перекрывает build_messages/solve, поисковая —
llm_tools/_run_tool).
"""

from .agent_rag import AgentRag
from .b1_routed import B1Routed


class AgentRagRouted(B1Routed, AgentRag):
    condition = "agent_rag_routed"
