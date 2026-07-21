"""B1 — модель + веб-поиск: ReAct-цикл с инструментом web_search.

Каркас (промпт, картинки, guided JSON, wrap-up) наследуется от B0 — меняется
только доступность инструмента, как того требует честное сравнение условий.
Все вызовы инструмента логируются в SolveResult.tool_calls.
"""

import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..config import Settings
from ..contracts import Task
from ..parsing import parse_solve_output
from ..schemas import SolveResult, ToolCallLog, Usage
from ..tools.search import WEB_SEARCH_TOOL_SCHEMA, searxng_search
from .b0_no_tools import B0NoTools


class B1Search(B0NoTools):
    condition = "b1_search"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.llm_tools = self.llm.bind(tools=[WEB_SEARCH_TOOL_SCHEMA])

    def build_messages(self, task: Task) -> list:
        messages = super().build_messages(task)
        # тот же системный промпт + примечание об инструменте
        system = messages[0].content + self.prompt["b1_tool_note"]
        return [SystemMessage(content=system), *messages[1:]]

    def _run_tool(self, name: str, args: dict, seen: set[str]) -> str:
        if name != "web_search":
            return f"Bilinmeyen araç: {name}"
        query = str(args.get("query") or "").strip()
        if not query:
            return "Boş sorgu. query parametresini doldur veya aramadan çöz."
        if query.casefold() in seen:
            # модель зацикливается на одном запросе — стоп-сигнал
            return ("Bu sorguyu zaten yaptın, sonuçlar yukarıda. "
                    "Yeni arama yapma; mevcut bilgiyle çözümü tamamla.")
        seen.add(query.casefold())
        return searxng_search(self.settings, query)

    def _react_loop(self, task: Task, usage: Usage,
                    log: list[ToolCallLog]) -> str:
        """Свободный цикл рассуждение→поиск→…; возвращает финальный текст."""
        messages = self.build_messages(task)
        content = ""
        seen_queries: set[str] = set()
        for _ in range(self.settings.agent_max_steps):
            response = self.llm_tools.invoke(
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
            )
            meta = response.usage_metadata or {}
            usage.input_tokens = (usage.input_tokens or 0) + (meta.get("input_tokens") or 0)
            usage.output_tokens = (usage.output_tokens or 0) + (meta.get("output_tokens") or 0)
            if not response.tool_calls:
                content = response.content if isinstance(response.content, str) else str(response.content)
                break
            messages.append(response)
            for tc in response.tool_calls:
                result = self._run_tool(tc["name"], tc["args"] or {}, seen_queries)
                log.append(ToolCallLog(tool=tc["name"], args=tc["args"] or {},
                                       result_preview=result[:300]))
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        return content

    def solve(self, task: Task) -> SolveResult:
        raw: str | None = None
        parsed = None
        error: str | None = None
        forced = False
        usage = Usage()
        log: list[ToolCallLog] = []

        started = time.perf_counter()
        try:
            raw = self._react_loop(task, usage, log)
            parsed = parse_solve_output(raw) if raw else None
            if parsed is None:
                # цикл кончился без валидного JSON (болтовня/обрыв/лимит шагов):
                # принудительный финал с черновиком, как в B0
                wrapup = self.prompt["wrapup"].format(draft=(raw or "")[-6000:])
                messages = self.build_messages(task)
                messages.append(HumanMessage(content=[{"type": "text", "text": wrapup}]))
                raw = self._invoke(messages, task, usage, max_tokens=2048, think=False)
                forced = True
                parsed = parse_solve_output(raw)
                if parsed is None:
                    error = "parse_error"
        except Exception as exc:  # сетевые/серверные ошибки — в результат
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
            forced_answer=forced,
            raw_response=raw,
            tool_calls=log,
            usage=usage,
            error=error,
        )
