from .agent_rag import AgentRag
from .b0_no_tools import B0NoTools
from .base import Solver

SOLVERS: dict[str, type[Solver]] = {
    B0NoTools.condition: B0NoTools,
    AgentRag.condition: AgentRag,
    # "b1_search": B1Search,  # появится следующим этапом
}
