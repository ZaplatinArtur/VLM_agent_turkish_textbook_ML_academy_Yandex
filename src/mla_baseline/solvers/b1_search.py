"""B1 — модель + веб-поиск: ReAct-цикл с инструментом web_search.

Каркас (промпт, картинки, guided JSON, wrap-up) наследуется от B0 — меняется
только доступность инструмента, как того требует честное сравнение условий.
Все вызовы инструмента логируются в SolveResult.tool_calls.
"""

import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..config import Settings
from ..contracts import Task
from ..parsing import parse_solve_output  # noqa: F401  (используется в _react_loop)
from ..schemas import SolveResult, ToolCallLog, Usage
from ..tools import ToolUnavailable
from ..tools.search import WEB_SEARCH_TOOL_SCHEMA, searxng_search
from .b0_no_tools import B0NoTools


class B1Search(B0NoTools):
    condition = "b1_search"
    tool_note_key = "b1_tool_note"  # какая тул-политика уходит в системный промпт

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.llm_tools = self.llm.bind(tools=[WEB_SEARCH_TOOL_SCHEMA])

    def build_messages(self, task: Task) -> list:
        messages = super().build_messages(task)
        # тот же системный промпт + примечание об инструменте
        system = messages[0].content + self.prompt[self.tool_note_key]
        return [SystemMessage(content=system), *messages[1:]]

    tool_name = "web_search"

    def _search(self, args: dict) -> str:
        """Сам поиск; подклассы меняют только его (сниппеты / полные страницы)."""
        return searxng_search(self.settings, str(args["query"]))

    def _dedup_key(self, args: dict) -> str:
        return str(args["query"]).casefold()

    def _max_calls(self) -> int:
        return self.settings.search_max_calls

    def _run_tool(self, name: str, args: dict,
                  seen: set[str]) -> tuple[str, bool, dict | None]:
        """Выполняет вызов: (текст модели, оставить ли тул, диагностика).

        Дисциплина цикла (лимит, дедуп, реакция на несуществующий тул и на
        неработающий бэкенд) живёт здесь для всех поисковых условий; подклассы
        переопределяют `_search`.
        """
        if name != self.tool_name:
            # модель «вызвала» несуществующий тул (reasoning, JSON, final_answer):
            # это попытка закрыть задачу, а не поискать — снимаем инструмент
            return (f"Bilinmeyen araç: {name}. Araç kullanmayı bırak ve çözümü tamamla.",
                    False, None)
        query = str(args.get("query") or "").strip()
        if not query:
            return "Boş sorgu. query parametresini doldur veya aramadan çöz.", True, None
        args = {**args, "query": query}
        if self._dedup_key(args) in seen:
            # модель зацикливается на одном запросе — тул снимаем, текстовый
            # стоп-сигнал она игнорирует (дубли до 24% вызовов в прогонах)
            return "Bu sorguyu zaten yaptın, sonuçlar yukarıda.", False, None
        if len(seen) >= self._max_calls():
            return "Arama limitine ulaştın.", False, None
        seen.add(self._dedup_key(args))
        try:
            return self._search(args), True, None
        except ToolUnavailable as exc:
            # бэкенд не работает: переформулировка не поможет, снимаем тул
            return exc.message_for_model, False, exc.diag

    def _step_llm(self, budget: int):
        """LLM с инструментом и укороченным бюджетом шага."""
        # Метод B0 переименовали в _generation_extra_body, ветку b1 тогда не
        # тронули — до первого запуска условия это не всплывало.
        return self.llm_tools.bind(
            extra_body=self._generation_extra_body(max_tokens=budget)
        )

    def _finish_call(self, messages: list, task: Task, usage: Usage) -> str:
        """Финал без инструмента: полный бюджет + структурный JSON."""
        messages.append(HumanMessage(content=[{
            "type": "text", "text": self.prompt["finish_now"]}]))
        try:
            return self._invoke(messages, task, usage)
        except Exception as exc:
            if "LengthFinishReason" not in type(exc).__name__:
                raise
            return ""  # пусть solve() уйдёт в несгораемую лестницу финализации

    def _react_loop(self, task: Task, usage: Usage,
                    log: list[ToolCallLog]) -> tuple[str, list]:
        """Цикл рассуждение→поиск→…; возвращает (текст, диалог).

        Два ограничителя, добавленные по разбору ошибок
        (reports/tool_errors_analysis.md): бюджет шага не даёт сжечь весь
        лимит токенов на первом же рассуждении, а исчерпание/дубль/попытка
        «вызвать» финал физически снимают инструмент вместо текстового
        стоп-сигнала, который модель игнорировала.
        """
        messages = self.build_messages(task)
        content = ""
        seen_queries: set[str] = set()
        tools_on = True
        # шаг не может стоить дороже всей задачи: при малом общем бюджете
        # (дефолт 3072) укороченный лимит просто совпадает с ним
        step_budget = min(self.settings.agent_step_max_tokens or self.settings.max_tokens,
                          self.settings.max_tokens)
        for step in range(self.settings.agent_max_steps):
            last_step = step == self.settings.agent_max_steps - 1
            if not tools_on or last_step:
                content = self._finish_call(messages, task, usage)
                break
            response = self._step_llm(step_budget).invoke(
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
            messages.append(response)
            if not response.tool_calls:
                content = (response.content if isinstance(response.content, str)
                           else str(response.content))
                # Шаг мог оборваться на бюджете посреди размышления: тогда
                # текст ушёл в reasoning_content, а content пуст или без JSON.
                # Замер b1_search на V100: 49 из 89 таких задач упирались
                # ровно в 4096 токенов шага. Добиваем полным бюджетом здесь,
                # а не аварийной лестницей в solve() — она даёт 28% точности.
                if parse_solve_output(content) is None:
                    content = self._finish_call(messages, task, usage)
                break
            for tc in response.tool_calls:
                result, keep, diag = self._run_tool(
                    tc["name"], tc["args"] or {}, seen_queries)
                log.append(ToolCallLog(tool=tc["name"], args=tc["args"] or {},
                                       result_preview=result[:300], diag=diag))
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                tools_on = tools_on and keep
        return content, messages

    def solve(self, task: Task) -> SolveResult:
        raw: str | None = None
        parsed = None
        error: str | None = None
        forced = False
        usage = Usage()
        log: list[ToolCallLog] = []

        started = time.perf_counter()
        try:
            raw, convo = self._react_loop(task, usage, log)
            parsed = parse_solve_output(raw) if raw else None
            if parsed is None:
                # цикл кончился без валидного JSON: финализируем ПОВЕРХ всего
                # диалога (с результатами поиска), а не по огрызку черновика
                convo.append(HumanMessage(content=[{
                    "type": "text",
                    "text": "Şimdi çözümünü bitir ve YALNIZCA istenen JSON "
                            "formatında nihai cevabını ver."}]))
                try:
                    raw = self._invoke(convo, task, usage, max_tokens=2048, think=False)
                except Exception as exc:
                    if "LengthFinishReason" not in type(exc).__name__:
                        raise
                    # несгораемая ступень: по черновику
                    raw = self._finalize(task, usage, raw or "")
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
            model=self.settings.llm_model_name,
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
