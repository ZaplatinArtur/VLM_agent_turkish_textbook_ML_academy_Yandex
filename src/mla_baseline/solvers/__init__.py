from .b0_no_tools import B0NoTools
from .base import Solver

SOLVERS: dict[str, type[Solver]] = {
    B0NoTools.condition: B0NoTools,
    # "b1_search": B1Search,  # появится следующим этапом
    # "agent_rag": AgentRag,  # тулы команды ретрива, после интеграции
}
