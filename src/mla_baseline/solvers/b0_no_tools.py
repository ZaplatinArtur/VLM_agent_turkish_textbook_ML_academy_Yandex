"""B0 — «голая» модель: один вызов VLM без тулов.

Опорная точка сравнения: «Çöz» + картинка -> строгий JSON с решением.
"""

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config import Settings
from ..contracts import Task
from ..images import image_ref_to_block
from ..parsing import parse_solve_output
from ..prompts import PROMPTS
from ..schemas import SolveResult, Usage
from .base import Solver


class B0NoTools(Solver):
    condition = "b0_no_tools"

    def __init__(self, settings: Settings, *, llm: Any | None = None):
        super().__init__(settings)
        self.prompt = PROMPTS[settings.prompt_version]
        self.llm = llm or ChatOpenAI(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            model=settings.model_name,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            timeout=settings.request_timeout_s,
            max_retries=2,
        )

    def build_messages(self, task: Task) -> list:
        content: list[dict] = [{"type": "text", "text": self.prompt["user_text"]}]

        for ref in task.question_images:
            content.append(image_ref_to_block(ref, self.settings.data_root))

        # Сценарий «ленивый школьник»: при наличии картинки текст условия не шлём
        if not task.question_images or self.settings.include_question_text_with_images:
            content.append({"type": "text", "text": task.question})

        hint = self.prompt["answer_type_hints"].get(task.answer_type)
        if hint:
            content.append({"type": "text", "text": hint})

        return [
            SystemMessage(content=self.prompt["system"]),
            HumanMessage(content=content),
        ]

    def _invoke_kwargs(self) -> dict:
        from ..schemas import SolveOutput

        schema = SolveOutput.model_json_schema()
        mode = self.settings.structured_mode
        if mode == "response_format":
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "solve_output", "schema": schema},
                }
            }
        if mode == "guided_json":
            return {"extra_body": {"guided_json": schema}}
        return {}

    def solve(self, task: Task) -> SolveResult:
        messages = self.build_messages(task)
        raw: str | None = None
        parsed = None
        error: str | None = None
        usage = Usage()

        started = time.perf_counter()
        try:
            response = self.llm.invoke(messages, **self._invoke_kwargs())
            raw = response.content if isinstance(response.content, str) else str(response.content)
            meta = response.usage_metadata or {}
            usage.input_tokens = meta.get("input_tokens")
            usage.output_tokens = meta.get("output_tokens")
            parsed = parse_solve_output(raw)
            if parsed is None:
                error = "parse_error"
        except Exception as exc:  # сетевые/серверные ошибки — в результат, не наружу
            error = f"{type(exc).__name__}: {exc}"
        usage.latency_s = round(time.perf_counter() - started, 3)

        return SolveResult(
            task_id=task.task_id,
            condition=self.condition,
            model=self.settings.model_name,
            prompt_version=self.settings.prompt_version,
            final_answer=parsed.final_answer if parsed else None,
            solution_steps=parsed.solution_steps if parsed else None,
            raw_response=raw,
            usage=usage,
            error=error,
        )
