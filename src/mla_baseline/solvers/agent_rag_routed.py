"""E4: deterministic subject router in front of image-first checked RAG."""

from __future__ import annotations

from typing import Any

from ..config import Settings
from ..contracts import Task
from ..schemas import SolveResult
from ..tools import TextbookSearchBackend
from .agent_rag import AgentRag
from .b0_no_tools import B0NoTools


class AgentRagRouted(AgentRag):
    """Skip retrieval for configured subjects; use checked RAG elsewhere."""

    condition = "agent_rag_routed"
    experiment_id = "e4_routed_image_first_rag_v1"

    def __init__(
        self,
        settings: Settings,
        *,
        llm: Any | None = None,
        search_client: TextbookSearchBackend | None = None,
    ) -> None:
        super().__init__(settings, llm=llm, search_client=search_client)
        self.no_retrieval_subjects = {
            subject.strip().casefold()
            for subject in settings.rag_no_retrieval_subjects.split(",")
            if subject.strip()
        }

    def _routed_off(self, task: Task) -> bool:
        return task.subject.strip().casefold() in self.no_retrieval_subjects

    def build_messages(self, task: Task) -> list:
        if self._routed_off(task):
            return B0NoTools.build_messages(self, task)
        return super().build_messages(task)

    def _annotate_route(
        self,
        result: SolveResult,
        *,
        task: Task,
        route: str,
        reason: str,
    ) -> SolveResult:
        result.generation.update(
            {
                "experiment_id": self.experiment_id,
                "agent_strategy": "subject_routed_image_first_checked_retrieval_v1",
                "retrieval_route": route,
                "retrieval_route_reason": reason,
                "retrieval_route_subject": task.subject,
            }
        )
        return result

    def solve(self, task: Task) -> SolveResult:
        if self._routed_off(task):
            result = B0NoTools.solve(self, task)
            result.exit_reason = result.error or "router_no_retrieval"
            result.retrieval_relevance = "not_attempted"
            result.retrieval_conflict = False
            result.answer_source = (
                "image_only_no_retrieval"
                if task.question_images and not self.settings.text_only
                else "text_only_no_retrieval"
            )
            return self._annotate_route(
                result,
                task=task,
                route="skip",
                reason="subject_blocklist",
            )

        return self._annotate_route(
            super().solve(task),
            task=task,
            route="allow",
            reason="subject_allowed",
        )
