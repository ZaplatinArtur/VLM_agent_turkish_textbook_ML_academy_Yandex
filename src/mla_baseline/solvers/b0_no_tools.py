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
        from ..tracing import langchain_callbacks

        self.prompt = PROMPTS[settings.prompt_version]
        self.callbacks = langchain_callbacks(settings)
        self.llm = llm or ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model_name,
            temperature=settings.temperature,
            top_p=settings.top_p,
            presence_penalty=settings.presence_penalty,
            timeout=settings.request_timeout_s,
            max_retries=2,
            extra_body=self._generation_extra_body(),
        )

    def _generation_extra_body(
        self,
        *,
        max_tokens: int | None = None,
        think: bool = True,
    ) -> dict:
        enabled = (
            self.settings.enable_thinking
            if think and not self.settings.disable_thinking
            else False
        )
        body: dict[str, Any] = {
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        # transformers serve отвергает незнакомые поля с 422; vLLM top_k принимает.
        # MLA_TOP_K=0 убирает его из запроса.
        if self.settings.top_k > 0:
            body["top_k"] = self.settings.top_k
        if self.settings.llm_provider == "openrouter":
            body["reasoning"] = (
                {"enabled": True} if enabled else {"effort": "none"}
            )
        else:
            body["chat_template_kwargs"] = {"enable_thinking": enabled}
        return body

    def build_messages(self, task: Task) -> list:
        content: list[dict] = [{"type": "text", "text": self.prompt["user_text"]}]

        active_images = [] if self.settings.text_only else task.question_images
        for ref in active_images:
            content.append(image_ref_to_block(ref, self.settings.data_root))

        # Сценарий «ленивый школьник»: при наличии картинки текст условия не шлём
        if not active_images or self.settings.include_question_text_with_images:
            content.append({"type": "text", "text": task.question})

        hint = self.prompt["answer_type_hints"].get(task.answer_type)
        if hint:
            content.append({"type": "text", "text": hint})

        return [
            SystemMessage(content=self.prompt["system"]),
            HumanMessage(content=content),
        ]

    def _invoke_kwargs(self, response_schema: dict | None = None) -> dict:
        from ..schemas import SolveOutput

        schema = response_schema or SolveOutput.model_json_schema()
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

    def _invoke(
        self,
        messages: list,
        task: Task,
        usage: Usage,
        *,
        structured: bool = True,
        max_tokens: int | None = None,
        think: bool = True,
        response_schema: dict | None = None,
    ) -> str:
        """Один вызов модели с общей трассировкой и подсчётом токенов."""

        llm = self.llm
        if max_tokens is not None or not think:
            llm = self.llm.bind(
                extra_body=self._generation_extra_body(
                    max_tokens=max_tokens,
                    think=think,
                )
            )
        response = llm.invoke(
            messages,
            config={
                "callbacks": self.callbacks,
                "run_name": f"{self.condition}:{task.task_id}",
                "metadata": {
                    "task_id": task.task_id,
                    "subject": task.subject,
                    "answer_type": task.answer_type,
                    "langfuse_tags": [self.condition, self.settings.prompt_version],
                },
            },
            **(self._invoke_kwargs(response_schema) if structured else {}),
        )
        meta = response.usage_metadata or {}
        usage.input_tokens = (usage.input_tokens or 0) + (meta.get("input_tokens") or 0)
        usage.output_tokens = (usage.output_tokens or 0) + (meta.get("output_tokens") or 0)
        return response.content if isinstance(response.content, str) else str(response.content)

    def _finalize(self, task: Task, usage: Usage, draft: str) -> str:
        """Запросить короткий структурированный финал по оборванному черновику."""

        import json

        messages = self.build_messages(task)
        messages.append(
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": self.prompt["wrapup"].format(draft=draft[-6_000:]),
                    }
                ]
            )
        )
        try:
            return self._invoke(
                messages,
                task,
                usage,
                max_tokens=2_048,
                think=False,
            )
        except Exception as exc:
            if "LengthFinishReason" not in type(exc).__name__:
                raise

        messages.append(
            HumanMessage(
                content=[{"type": "text", "text": self.prompt["last_resort"]}]
            )
        )
        answer = self._invoke(
            messages,
            task,
            usage,
            structured=False,
            max_tokens=64,
            think=False,
        ).strip()
        answer = answer.splitlines()[0][:120] if answer else ""
        return json.dumps(
            {
                "solution_steps": draft[-1_500:],
                "final_answer": answer,
            },
            ensure_ascii=False,
        )

    def _forced_wrapup(
        self,
        task: Task,
        usage: Usage,
    ) -> tuple[str | None, str | None]:
        draft = self._invoke(
            self.build_messages(task),
            task,
            usage,
            structured=False,
        )
        return self._finalize(task, usage, draft), draft

    def solve(self, task: Task) -> SolveResult:
        raw: str | None = None
        draft: str | None = None
        parsed = None
        error: str | None = None
        forced = False
        usage = Usage()

        started = time.perf_counter()
        try:
            raw = self._invoke(self.build_messages(task), task, usage)
        except Exception as exc:
            if "LengthFinishReason" in type(exc).__name__:
                try:
                    raw, draft = self._forced_wrapup(task, usage)
                    forced = True
                except Exception as wrapup_exc:
                    error = f"{type(wrapup_exc).__name__}: {wrapup_exc}"
            else:
                error = f"{type(exc).__name__}: {exc}"

        if raw is not None and error is None:
            parsed = parse_solve_output(raw)
            if parsed is None:
                error = "parse_error"
        usage.latency_s = round(time.perf_counter() - started, 3)

        return SolveResult(
            task_id=task.task_id,
            condition=self.condition,
            model=self.settings.llm_model_name,
            prompt_version=self.settings.prompt_version,
            final_answer=parsed.final_answer if parsed else None,
            solution_steps=parsed.solution_steps if parsed else None,
            reasoning=(
                parsed.reasoning if parsed else None
            ) or (draft[-4_000:] if draft else None),
            forced_answer=forced,
            raw_response=raw,
            generation={
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "top_k": self.settings.top_k,
                "presence_penalty": self.settings.presence_penalty,
                "max_tokens": self.settings.max_tokens,
                "structured_mode": self.settings.structured_mode,
                "enable_thinking": self.settings.enable_thinking,
                "llm_provider": self.settings.llm_provider,
                "experiment_id": "e0_no_tools_v1",
            },
            usage=usage,
            error=error,
        )
