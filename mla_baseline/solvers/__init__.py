from .b0_no_tools import B0NoTools
from .b1_search import B1Search
from .base import Solver

SOLVERS: dict[str, type[Solver]] = {
    B0NoTools.condition: B0NoTools,
    B1Search.condition: B1Search,
    # "agent_rag": AgentRag,  # тулы команды ретрива, после интеграции
}
