"""Привязка к официальному источнику перед обычным RAG-агентом.

Если текст задачи опознан как упражнение из учебника, ответ берётся из ключа и
модель не вызывается. Не опознан — работает AgentRag без изменений.
"""

from __future__ import annotations

from typing import Any

from source_router import Route, route as route_source

from ..config import Settings
from ..contracts import Task
from ..schemas import SolveResult
from ..tools import TextbookSearchBackend
from .agent_rag import AgentRag


class AgentRagSourced(AgentRag):
    """Каскад: роутер по официальным источникам, иначе поиск по учебникам."""

    condition = "agent_rag_sourced"
    experiment_id = "e5_source_router_then_checked_rag_v1"

    def __init__(
        self,
        settings: Settings,
        *,
        llm: Any | None = None,
        search_client: TextbookSearchBackend | None = None,
        router: Any | None = None,
    ) -> None:
        super().__init__(settings, llm=llm, search_client=search_client)
        self._route = router or route_source

    def _route_task(self, task: Task) -> Route | None:
        """Роутеру уходят только наблюдаемые поля: эталонный ответ ему не положен."""
        return self._route(task.question, answer_type=task.answer_type)

    def _from_source(self, task: Task, route: Route) -> SolveResult:
        return SolveResult(
            task_id=task.task_id,
            condition=self.condition,
            model=self.settings.llm_model_name,
            prompt_version=self.settings.prompt_version,
            final_answer=route.answer,
            exit_reason="answered_from_official_source",
            retrieval_relevance="not_attempted",
            retrieval_conflict=False,
            answer_source="official_source",
            generation={
                "experiment_id": self.experiment_id,
                "route_decision": "official_source",
                "route_family": route.family,
                "route_record_id": route.record_id,
                "route_closure": route.closure,
                "route_score": route.score,
                "route_margin": route.margin,
                "route_source_page": route.source_page,
            },
        )

    def solve(self, task: Task) -> SolveResult:
        route = self._route_task(task)
        if route is not None:
            return self._from_source(task, route)

        result = super().solve(task)
        result.generation.update(
            {
                "experiment_id": self.experiment_id,
                "route_decision": "abstain",
            }
        )
        if result.answer_source is None:
            result.answer_source = "retrieval"
        return result
