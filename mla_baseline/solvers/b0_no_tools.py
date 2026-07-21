"""B0 — «голая» модель: один вызов VLM без тулов.

Опорная точка сравнения: «Çöz» + картинка -> строгий JSON с решением.
"""

import time

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

    def __init__(self, settings: Settings):
        super().__init__(settings)
        from ..tracing import langchain_callbacks

        self.prompt = PROMPTS[settings.prompt_version]
        self.callbacks = langchain_callbacks(settings)
        self.llm = ChatOpenAI(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            model=settings.model_name,
            # именно max_tokens в payload: langchain шлёт max_completion_tokens,
            # который Ollama молча игнорирует (вылезает за лимит до num_ctx);
            # extra_body кладёт ключ в payload в обход маппинга langchain
            # max_tokens в payload в обход langchain-маппинга; top_k — параметр vLLM
            extra_body={"max_tokens": settings.max_tokens, "top_k": settings.top_k},
            temperature=settings.temperature,
            top_p=settings.top_p,
            presence_penalty=settings.presence_penalty,
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

    def _invoke(self, messages, task: Task, usage: Usage, structured: bool = True,
                max_tokens: int | None = None, think: bool = True):
        """Один вызов модели с трейсингом; токены суммируются в usage."""
        llm = self.llm
        if max_tokens is not None or not think:
            extra: dict = {"max_tokens": max_tokens or self.settings.max_tokens,
                           "top_k": self.settings.top_k}
            if not think:
                # финалу думать не надо — иначе thinking сжигает весь бюджет
                extra["chat_template_kwargs"] = {"enable_thinking": False}
            llm = self.llm.bind(extra_body=extra)
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
            **(self._invoke_kwargs() if structured else {}),
        )
        meta = response.usage_metadata or {}
        usage.input_tokens = (usage.input_tokens or 0) + (meta.get("input_tokens") or 0)
        usage.output_tokens = (usage.output_tokens or 0) + (meta.get("output_tokens") or 0)
        return response.content if isinstance(response.content, str) else str(response.content)

    def _forced_wrapup(self, task: Task, usage: Usage) -> tuple[str | None, str | None]:
        """Бюджет сгорел: получить черновик без guided decoding и потребовать финал.

        Возвращает (raw_final, draft). Модель-«думатель» может блуждать дольше
        любого бюджета; чат-продукты в этот момент принуждают к ответу — делаем
        так же, а judge видит flag forced_answer.
        """
        draft = self._invoke(self.build_messages(task), task, usage, structured=False)
        wrapup = self.prompt["wrapup"].format(draft=draft[-6000:])
        messages = self.build_messages(task)
        messages.append(HumanMessage(content=[{"type": "text", "text": wrapup}]))
        # финалу — короткий бюджет и без thinking: JSON влезает, блуждать негде
        return self._invoke(messages, task, usage, max_tokens=2048, think=False), draft

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
                # рассуждение не сошлось за бюджет — принудительный финал
                try:
                    raw, draft = self._forced_wrapup(task, usage)
                    forced = True
                except Exception as exc2:
                    error = f"{type(exc2).__name__}: {exc2}"
            else:  # сетевые/серверные ошибки — в результат, не наружу
                error = f"{type(exc).__name__}: {exc}"
        if raw is not None and error is None:
            parsed = parse_solve_output(raw)
            if parsed is None:
                error = "parse_error"
        usage.latency_s = round(time.perf_counter() - started, 3)

        return SolveResult(
            task_id=task.task_id,
            condition=self.condition,
            model=self.settings.model_name,
            prompt_version=self.settings.prompt_version,
            final_answer=parsed.final_answer if parsed else None,
            solution_steps=parsed.solution_steps if parsed else None,
            reasoning=(parsed.reasoning if parsed else None) or (draft[-4000:] if draft else None),
            forced_answer=forced,
            raw_response=raw,
            usage=usage,
            error=error,
        )
