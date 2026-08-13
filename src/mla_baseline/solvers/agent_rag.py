"""Agent condition: Qwen with the textbook retrieval tool."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..config import Settings
from ..contracts import Task
from ..parsing import parse_solve_output
from ..schemas import (
    CompactSolveOutput,
    FinalAnswerOnly,
    ImageTaskEvidence,
    RetrievalConflictCheck,
    SolveResult,
    ToolCallLog,
    Usage,
)
from ..tools import (
    LocalTextbookSearchClient,
    TextbookSearchBackend,
    create_search_textbooks_tool,
)
from .b0_no_tools import B0NoTools


class AgentRag(B0NoTools):
    """A bounded, traceable model-tool loop for textbook retrieval."""

    condition = "agent_rag"

    def __init__(
        self,
        settings: Settings,
        *,
        llm: Any | None = None,
        search_client: TextbookSearchBackend | None = None,
    ) -> None:
        super().__init__(settings, llm=llm)
        if settings.retrieval_fetch_k < settings.retrieval_top_k:
            raise ValueError("retrieval_fetch_k must be at least retrieval_top_k")
        self.search_client = search_client or LocalTextbookSearchClient(
            retrieval_fetch_k=settings.retrieval_fetch_k,
            mmr_lambda=(
                settings.retrieval_mmr_lambda
                if settings.retrieval_mmr_enabled
                else None
            ),
            context_order=settings.retrieval_context_order,
        )
        self.search_tool = create_search_textbooks_tool(
            self.search_client,
            top_k=settings.retrieval_top_k,
            max_text_chars=settings.retrieval_max_context_chars,
        )
        self.agent_llm = self.llm.bind_tools([self.search_tool])

    def build_messages(self, task: Task) -> list:
        messages = super().build_messages(task)
        base_system = str(messages[0].content)
        messages[0] = SystemMessage(
            content=f"{base_system}\n\n{self.prompt['rag_tool_policy']}"
        )
        return messages

    @staticmethod
    def _response_text(response: Any) -> str:
        if isinstance(response.content, str):
            return response.content
        if isinstance(response.content, list):
            text_parts = [
                str(block.get("text"))
                for block in response.content
                if isinstance(block, dict) and block.get("text") is not None
            ]
            if text_parts:
                return "\n".join(text_parts)
        return str(response.content)

    @staticmethod
    def _add_usage(usage: Usage, response: Any) -> None:
        metadata = response.usage_metadata or {}
        input_tokens = metadata.get("input_tokens")
        output_tokens = metadata.get("output_tokens")
        if input_tokens is not None:
            usage.input_tokens = (usage.input_tokens or 0) + int(input_tokens)
        if output_tokens is not None:
            usage.output_tokens = (usage.output_tokens or 0) + int(output_tokens)

    @staticmethod
    def _tool_trace(
        *,
        name: str,
        arguments: dict[str, Any],
        output: str,
    ) -> ToolCallLog:
        payload = AgentRag._tool_payload(output)

        hits = payload.get("hits") if isinstance(payload.get("hits"), list) else []
        chunk_ids = [
            str(hit["chunk_id"])
            for hit in hits
            if isinstance(hit, dict) and hit.get("chunk_id")
        ]
        latency = payload.get("latency_ms")
        return ToolCallLog(
            tool=name,
            args=arguments,
            result_preview=output[:1_000],
            returned_chunk_ids=chunk_ids,
            latency_ms=float(latency) if isinstance(latency, (int, float)) else None,
            relevance=(
                dict(payload["relevance"])
                if isinstance(payload.get("relevance"), dict)
                else None
            ),
            error=str(payload["error"]) if payload.get("error") else None,
        )

    @staticmethod
    def _tool_payload(output: str) -> dict[str, Any]:
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _normalize_query(arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query") or "").casefold()
        return " ".join(re.sub(r"[^\w]+", " ", query).split())

    @staticmethod
    def _error_tool_output(message: str) -> str:
        return json.dumps(
            {
                "error": message,
                "retrieved": 0,
                "returned": 0,
                "relevance": {
                    "label": "error",
                    "is_useful": False,
                    "top_score": None,
                    "reason": message,
                },
                "hits": [],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _image_stage_messages(self, task: Task, prompt_key: str) -> list:
        messages = super().build_messages(task)
        messages[0] = SystemMessage(content=str(self.prompt[prompt_key]))
        return messages

    def _extract_image_evidence(
        self,
        task: Task,
        usage: Usage,
    ) -> ImageTaskEvidence | None:
        if self.settings.text_only or not task.question_images:
            return None
        raw = self._invoke(
            self._image_stage_messages(task, "image_evidence"),
            task,
            usage,
            max_tokens=768,
            think=False,
            response_schema=ImageTaskEvidence.model_json_schema(),
        )
        return ImageTaskEvidence.model_validate_json(raw)

    @staticmethod
    def _retrieval_query_from_evidence(evidence: ImageTaskEvidence) -> str:
        parts = [evidence.topic, *evidence.unknown_concepts]
        unique_parts = dict.fromkeys(part.strip() for part in parts if part.strip())
        return " ".join(unique_parts)

    @staticmethod
    def _evidence_note(evidence: ImageTaskEvidence, retrieval_query: str) -> str:
        payload = {
            **evidence.model_dump(),
            "retrieval_query": retrieval_query,
        }
        return (
            "Yapılandırılmış görsel okuması (görsel yine birincil kaynaktır):\n"
            + json.dumps(payload, ensure_ascii=False)
            + "\nİlk aramada retrieval_query değerini aynen kullan."
        )

    def _filter_conflicting_chunks(
        self,
        *,
        output: str,
        task: Task,
        evidence: ImageTaskEvidence | None,
        usage: Usage,
    ) -> tuple[str, bool]:
        payload = self._tool_payload(output)
        hits = payload.get("hits") if isinstance(payload.get("hits"), list) else []
        if evidence is None or not hits:
            return output, False

        check_input = {
            "image_evidence": evidence.model_dump(),
            "chunks": [
                {"chunk_id": hit.get("chunk_id"), "text": hit.get("text")}
                for hit in hits
                if isinstance(hit, dict)
            ],
        }
        messages = self._image_stage_messages(task, "retrieval_conflict")
        messages.append(
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": json.dumps(check_input, ensure_ascii=False),
                    }
                ]
            )
        )
        try:
            raw = self._invoke(
                messages,
                task,
                usage,
                max_tokens=512,
                think=False,
                response_schema=RetrievalConflictCheck.model_json_schema(),
            )
            check = RetrievalConflictCheck.model_validate_json(raw)
            reason = check.reason
            requested_conflicts = set(check.conflicting_chunk_ids)
        except Exception as exc:
            reason = f"conflict check failed; chunks hidden: {type(exc).__name__}"
            requested_conflicts = {
                str(hit.get("chunk_id"))
                for hit in hits
                if isinstance(hit, dict) and hit.get("chunk_id")
            }

        known_ids = {
            str(hit.get("chunk_id"))
            for hit in hits
            if isinstance(hit, dict) and hit.get("chunk_id")
        }
        conflicting_ids = requested_conflicts & known_ids
        if not conflicting_ids:
            payload["retrieval_conflict"] = False
            payload["conflict_reason"] = reason
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), False

        visible_hits = [
            hit
            for hit in hits
            if not isinstance(hit, dict)
            or str(hit.get("chunk_id")) not in conflicting_ids
        ]
        payload["hits"] = visible_hits
        payload["returned"] = len(visible_hits)
        payload["retrieval_conflict"] = True
        payload["conflict_reason"] = reason
        if not visible_hits:
            previous = payload.get("relevance")
            top_score = previous.get("top_score") if isinstance(previous, dict) else None
            payload["relevance"] = {
                "label": "conflict",
                "is_useful": False,
                "top_score": top_score,
                "reason": reason,
            }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), True

    def _verify_against_image(
        self,
        *,
        messages: list,
        task: Task,
        evidence: ImageTaskEvidence,
        candidate: str,
        usage: Usage,
    ) -> str | None:
        verification = {
            "image_evidence": evidence.image_evidence,
            "candidate_response": candidate,
        }
        final_messages = [
            *messages,
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            str(self.prompt["image_final_verification"])
                            + "\n"
                            + json.dumps(verification, ensure_ascii=False)
                        ),
                    }
                ]
            ),
        ]
        try:
            raw = self._invoke(
                final_messages,
                task,
                usage,
                max_tokens=512,
                think=False,
                response_schema=CompactSolveOutput.model_json_schema(),
            )
        except Exception:
            return None
        return raw if parse_solve_output(raw) is not None else None

    def _force_final_answer(
        self,
        messages: list,
        task: Task,
        usage: Usage,
    ) -> str:
        """Disable tools and require one schema-constrained final response."""
        final_messages = [
            *messages,
            HumanMessage(
                content=(
                    "Artık araç çağırma. Mevcut soru ve arama sonuçlarıyla karar ver. "
                    "reasoning alanını yazma. solution_steps en fazla üç kısa cümle "
                    "olsun. YALNIZCA solution_steps ve final_answer alanlarını içeren "
                    "kısa nihai JSON nesnesini döndür."
                )
            ),
        ]
        try:
            return self._invoke(
                final_messages,
                task,
                usage,
                max_tokens=512,
                think=False,
                response_schema=CompactSolveOutput.model_json_schema(),
            )
        except Exception as exc:
            if "LengthFinishReason" not in type(exc).__name__:
                raise

        answer_only_messages = [
            *messages,
            HumanMessage(
                content=(
                    "Uzun çözüm yazma. YALNIZCA final_answer alanını içeren tek "
                    "satırlık JSON döndür. Çoktan seçmeli soruda değer yalnızca "
                    "A, B, C, D veya E harfi olsun."
                )
            ),
        ]
        raw = self._invoke(
            answer_only_messages,
            task,
            usage,
            max_tokens=128,
            think=False,
            response_schema=FinalAnswerOnly.model_json_schema(),
        )
        answer = FinalAnswerOnly.model_validate_json(raw)
        return json.dumps(
            {
                "solution_steps": "Soru ve getirilen ders kitabı bağlamına göre sonuç belirlendi.",
                "final_answer": answer.final_answer,
            },
            ensure_ascii=False,
        )

    def solve(self, task: Task) -> SolveResult:
        messages = self.build_messages(task)
        usage = Usage()
        tool_logs: list[ToolCallLog] = []
        seen_queries: set[str] = set()
        executed_calls = 0
        last_relevance: str | None = None
        rewrite_used = False
        raw: str | None = None
        parsed = None
        error: str | None = None
        exit_reason: str | None = None
        forced = False
        image_evidence: ImageTaskEvidence | None = None
        first_retrieval_query: str | None = None
        retrieval_conflict = False

        started = time.perf_counter()
        try:
            image_evidence = self._extract_image_evidence(task, usage)
            if image_evidence is not None:
                first_retrieval_query = self._retrieval_query_from_evidence(
                    image_evidence
                )
                messages.append(
                    HumanMessage(
                        content=self._evidence_note(
                            image_evidence,
                            first_retrieval_query,
                        )
                    )
                )

            # Tool-enabled rounds are bounded. Any malformed final response,
            # duplicate call, or exhausted tool budget is followed by one
            # tool-disabled, schema-constrained final response.
            for _ in range(self.settings.retrieval_max_calls + 1):
                response = self.agent_llm.invoke(messages)
                self._add_usage(usage, response)
                messages.append(response)

                requested_calls = list(response.tool_calls or [])
                if not requested_calls:
                    raw = self._response_text(response)
                    parsed = parse_solve_output(raw)
                    malformed = parsed is None
                    if parsed is None:
                        raw = self._force_final_answer(messages, task, usage)
                        forced = True
                        parsed = parse_solve_output(raw)
                        if parsed is None:
                            error = "parse_error"
                            exit_reason = "parse_error"
                    if parsed is not None:
                        if malformed:
                            exit_reason = "malformed_response"
                        elif executed_calls == 0:
                            exit_reason = "answered_without_retrieval"
                        elif last_relevance == "confident":
                            exit_reason = "answered_with_retrieval"
                        else:
                            exit_reason = "answered_after_weak_retrieval"
                    break

                executed_this_round = 0
                for index, call in enumerate(requested_calls):
                    name = str(call.get("name") or "")
                    arguments = call.get("args")
                    if not isinstance(arguments, dict):
                        arguments = {}
                    else:
                        arguments = dict(arguments)
                    call_id = str(call.get("id") or f"tool-call-{len(tool_logs) + index}")
                    if (
                        name == self.search_tool.name
                        and executed_calls == 0
                        and first_retrieval_query
                    ):
                        arguments["query"] = first_retrieval_query
                    normalized_query = self._normalize_query(arguments)

                    if name != self.search_tool.name:
                        output = self._error_tool_output(f"unknown tool: {name}")
                    elif not normalized_query:
                        output = self._error_tool_output(
                            "empty textbook query rejected; answer without retrieval"
                        )
                    elif normalized_query in seen_queries:
                        output = self._error_tool_output(
                            "duplicate tool call rejected; reformulate or answer"
                        )
                    elif executed_this_round > 0:
                        output = self._error_tool_output(
                            "only one textbook search is allowed per agent step"
                        )
                    elif executed_calls >= self.settings.retrieval_max_calls:
                        output = self._error_tool_output(
                            "tool call limit reached; answer using available evidence"
                        )
                    else:
                        # Повтор разрешён и после confident: порог отсеивает выдачу
                        # не по теме, но не страницу по теме без нужного содержания.
                        # Потолок держит retrieval_max_calls.
                        seen_queries.add(normalized_query)
                        if executed_calls > 0:
                            rewrite_used = True
                        executed_calls += 1
                        executed_this_round += 1
                        try:
                            output = str(self.search_tool.invoke(arguments))
                        except Exception as exc:
                            output = self._error_tool_output(
                                f"{type(exc).__name__}: {exc}"
                            )
                        output, has_conflict = self._filter_conflicting_chunks(
                            output=output,
                            task=task,
                            evidence=image_evidence,
                            usage=usage,
                        )
                        retrieval_conflict = retrieval_conflict or has_conflict
                        payload = self._tool_payload(output)
                        relevance = payload.get("relevance")
                        if isinstance(relevance, dict):
                            last_relevance = str(relevance.get("label") or "") or None
                        elif isinstance(payload.get("hits"), list):
                            last_relevance = (
                                "confident" if payload["hits"] else "empty"
                            )

                    tool_logs.append(
                        self._tool_trace(
                            name=name,
                            arguments=arguments,
                            output=output,
                        )
                    )
                    messages.append(
                        ToolMessage(
                            content=output,
                            tool_call_id=call_id,
                            name=name or None,
                        )
                    )

                if (
                    executed_calls >= self.settings.retrieval_max_calls
                    or executed_this_round == 0
                ):
                    raw = self._force_final_answer(messages, task, usage)
                    forced = True
                    parsed = parse_solve_output(raw)
                    if parsed is None:
                        error = "parse_error"
                        exit_reason = "parse_error"
                    elif executed_this_round == 0:
                        exit_reason = "tool_call_rejected"
                    elif rewrite_used:
                        exit_reason = "forced_final_after_rewrite"
                    else:
                        exit_reason = "tool_call_limit"
                    break
            else:
                raw = self._force_final_answer(messages, task, usage)
                forced = True
                parsed = parse_solve_output(raw)
                if parsed is None:
                    error = "parse_error"
                    exit_reason = "parse_error"
                else:
                    exit_reason = "tool_call_limit"

            if parsed is not None and raw is not None and image_evidence is not None:
                verified_raw = self._verify_against_image(
                    messages=messages,
                    task=task,
                    evidence=image_evidence,
                    candidate=raw,
                    usage=usage,
                )
                if verified_raw is not None:
                    raw = verified_raw
                    parsed = parse_solve_output(raw)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            exit_reason = "solver_error"

        usage.latency_s = round(time.perf_counter() - started, 3)
        answer_source: str | None = None
        if parsed is not None:
            if image_evidence is not None and executed_calls:
                answer_source = (
                    "image_with_retrieval_support"
                    if last_relevance == "confident"
                    else "image_after_retrieval_rejected"
                )
            elif image_evidence is not None:
                answer_source = "image_only"
            elif executed_calls:
                answer_source = "text_with_retrieval_support"
            else:
                answer_source = "text_only"

        return SolveResult(
            task_id=task.task_id,
            condition=self.condition,
            model=self.settings.llm_model_name,
            prompt_version=self.settings.prompt_version,
            final_answer=parsed.final_answer if parsed else None,
            solution_steps=parsed.solution_steps if parsed else None,
            reasoning=parsed.reasoning if parsed else None,
            raw_response=raw,
            forced_answer=forced,
            exit_reason=exit_reason,
            image_evidence=(
                image_evidence.image_evidence if image_evidence is not None else []
            ),
            image_evidence_structured=(
                image_evidence.model_dump() if image_evidence is not None else None
            ),
            retrieval_relevance=last_relevance,
            retrieval_conflict=(retrieval_conflict if executed_calls else None),
            answer_source=answer_source,
            generation={
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "top_k": self.settings.top_k,
                "presence_penalty": self.settings.presence_penalty,
                "max_tokens": self.settings.max_tokens,
                "structured_mode": self.settings.structured_mode,
                "enable_thinking": self.settings.enable_thinking,
                "llm_provider": self.settings.llm_provider,
                "retrieval_strategy": (
                    "mmr" if self.settings.retrieval_mmr_enabled else "dense"
                ),
                "retrieval_fetch_k": self.settings.retrieval_fetch_k,
                "retrieval_mmr_lambda": (
                    self.settings.retrieval_mmr_lambda
                    if self.settings.retrieval_mmr_enabled
                    else None
                ),
                "retrieval_context_order": self.settings.retrieval_context_order,
                "agent_strategy": "image_first_checked_retrieval_v4",
                "experiment_id": "e3_image_first_checked_rag_v1",
            },
            tool_calls=tool_logs,
            usage=usage,
            error=error,
        )
