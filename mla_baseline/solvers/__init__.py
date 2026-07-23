from .b0_no_tools import B0NoTools
from .b1_deep import B1Deep
from .b1_deep_routed import B1DeepRouted
from .b1_routed import B1Routed
from .b1_search import B1Search
from .base import Solver

SOLVERS: dict[str, type[Solver]] = {
    B0NoTools.condition: B0NoTools,
    B1Search.condition: B1Search,
    B1Routed.condition: B1Routed,
    B1Deep.condition: B1Deep,
    B1DeepRouted.condition: B1DeepRouted,
    # "agent_rag": AgentRag,  # тулы команды ретрива, после интеграции
}
