"""Agent condition: Qwen with the textbook retrieval tool."""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..config import Settings
from ..contracts import Task
from ..parsing import parse_solve_output
from ..schemas import (
    CompactSolveOutput,
    FinalAnswerOnly,
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
        self.search_client = search_client or LocalTextbookSearchClient()
        self.search_tool = create_search_textbooks_tool(
            self.search_client,
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
        payload: dict[str, Any] = {}
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            pass

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
            error=str(payload["error"]) if payload.get("error") else None,
        )

    @staticmethod
    def _error_tool_output(message: str) -> str:
        return json.dumps(
            {"error": message, "returned": 0, "hits": []},
            ensure_ascii=False,
            separators=(",", ":"),
        )

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
        seen_calls: set[str] = set()
        executed_calls = 0
        raw: str | None = None
        parsed = None
        error: str | None = None
        forced = False

        started = time.perf_counter()
        try:
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
                    if parsed is None:
                        raw = self._force_final_answer(messages, task, usage)
                        forced = True
                        parsed = parse_solve_output(raw)
                        if parsed is None:
                            error = "parse_error"
                    break

                executed_this_round = 0
                for index, call in enumerate(requested_calls):
                    name = str(call.get("name") or "")
                    arguments = call.get("args")
                    if not isinstance(arguments, dict):
                        arguments = {}
                    call_id = str(call.get("id") or f"tool-call-{len(tool_logs) + index}")
                    call_key = json.dumps(
                        {"name": name, "args": arguments},
                        ensure_ascii=False,
                        sort_keys=True,
                    )

                    if name != self.search_tool.name:
                        output = self._error_tool_output(f"unknown tool: {name}")
                    elif call_key in seen_calls:
                        output = self._error_tool_output(
                            "duplicate tool call rejected; reformulate or answer"
                        )
                    elif executed_calls >= self.settings.retrieval_max_calls:
                        output = self._error_tool_output(
                            "tool call limit reached; answer using available evidence"
                        )
                    else:
                        seen_calls.add(call_key)
                        executed_calls += 1
                        executed_this_round += 1
                        try:
                            output = str(self.search_tool.invoke(arguments))
                        except Exception as exc:
                            output = self._error_tool_output(
                                f"{type(exc).__name__}: {exc}"
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
                    break
            else:
                raw = self._force_final_answer(messages, task, usage)
                forced = True
                parsed = parse_solve_output(raw)
                if parsed is None:
                    error = "parse_error"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        usage.latency_s = round(time.perf_counter() - started, 3)
        return SolveResult(
            task_id=task.task_id,
            condition=self.condition,
            model=self.settings.model_name,
            prompt_version=self.settings.prompt_version,
            final_answer=parsed.final_answer if parsed else None,
            solution_steps=parsed.solution_steps if parsed else None,
            reasoning=parsed.reasoning if parsed else None,
            raw_response=raw,
            forced_answer=forced,
            generation={
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "top_k": self.settings.top_k,
                "presence_penalty": self.settings.presence_penalty,
                "max_tokens": self.settings.max_tokens,
                "structured_mode": self.settings.structured_mode,
                "enable_thinking": self.settings.enable_thinking,
                "agent_strategy": "bounded_tools_then_structured_final_v2",
            },
            tool_calls=tool_logs,
            usage=usage,
            error=error,
        )
