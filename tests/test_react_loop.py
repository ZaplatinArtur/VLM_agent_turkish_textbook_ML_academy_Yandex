# -*- coding: utf-8 -*-
"""Проверка ограничителей ReAct-цикла на подставной модели (без vLLM).

Проверяем то, что чинилось по reports/tool_errors_analysis.md:
инструмент физически снимается после лимита, дубля и обращения к
несуществующему тулу, а шаги цикла получают укороченный бюджет.

Запуск: python tests/test_react_loop.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mla_baseline.config import Settings          # noqa: E402
from mla_baseline.contracts import Task           # noqa: E402
from mla_baseline.schemas import Usage            # noqa: E402
from mla_baseline.tools import ToolUnavailable    # noqa: E402
from mla_baseline.parsing import parse_solve_output as parse  # noqa: E402
from mla_baseline.solvers.agent_rag import AgentRag  # noqa: E402

TASK = Task(task_id="t1", question="2+2?", answer_type="numeric", grade=8,
            subject="Math", reference_answer="4", question_images=[])


class FakeResponse:
    """Ответ модели: либо тул-вызовы, либо текст."""

    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 20}


class FakeLLM:
    """Подставная модель: отдаёт заранее заданный сценарий и пишет журнал."""

    def __init__(self, script, journal, label):
        self.script = list(script)
        self.journal = journal
        self.label = label
        self.bound = {}

    def bind(self, **kwargs):
        clone = FakeLLM(self.script, self.journal, self.label)
        clone.script = self.script          # общий список: шаги идут по порядку
        clone.bound = {**self.bound, **kwargs}
        return clone

    def invoke(self, messages, config=None, **kwargs):
        budget = (self.bound.get("extra_body") or {}).get("max_tokens")
        self.journal.append({"llm": self.label, "budget": budget,
                             "has_tools": "tools" in self.bound})
        return self.script.pop(0) if self.script else FakeResponse(content="{}")


def build(script_tools, script_plain=None, search=None):
    settings = Settings(_env_file=None, langfuse_enabled=False)
    solver = AgentRag(settings)
    # сам поиск подставной: проверяем дисциплину цикла, а не бэкенд
    solver._search = search or (lambda args: f"sonuç: {args['query']}")
    journal: list[dict] = []
    tools_llm = FakeLLM(script_tools, journal, "tools")
    tools_llm.bound = {"tools": ["search_textbooks"]}
    solver.llm_tools = tools_llm
    solver.llm = FakeLLM(script_plain or [FakeResponse(content='{"final_answer": "4"}')],
                         journal, "plain")
    return solver, settings, journal


def call(query):
    return {"name": "search_textbooks", "args": {"query": query}, "id": f"c-{query}"}


def run(solver):
    usage, log = Usage(), []
    content, _ = solver._react_loop(TASK, usage, log)
    return content, log, usage


def check(name, cond):
    print(("OK   " if cond else "FAIL ") + name)
    return cond


def main() -> int:
    ok = True

    # 1. Лимит вызовов: после rag_max_calls инструмент снимается,
    #    четвёртый шаг идёт уже без тула
    solver, settings, journal = build([
        FakeResponse([call("a")]), FakeResponse([call("b")]),
        FakeResponse([call("c")]), FakeResponse([call("d")]),
    ])
    content, log, _ = run(solver)
    ok &= check("лимит: сделано ровно rag_max_calls реальных поисков",
                sum(1 for e in log if not e.result_preview.startswith("Arama limitine"))
                == settings.rag_max_calls)
    ok &= check("лимит: последний вызов отбит сообщением о лимите",
                log[-1].result_preview.startswith("Arama limitine"))
    ok &= check("лимит: финальный вызов без инструмента",
                journal[-1]["llm"] == "plain" and not journal[-1]["has_tools"])

    # 2. Дубль запроса снимает инструмент сразу, без ещё одного круга
    solver, _, journal = build([
        FakeResponse([call("a")]), FakeResponse([call("a")]),
        FakeResponse([call("a")]), FakeResponse([call("a")]),
    ])
    content, log, _ = run(solver)
    ok &= check("дубль: всего два тул-вызова, дальше финал", len(log) == 2)
    ok &= check("дубль: финал без инструмента", journal[-1]["llm"] == "plain")

    # 3. Обращение к несуществующему тулу трактуется как попытка ответить
    solver, _, journal = build([
        FakeResponse([{"name": "reasoning", "args": {}, "id": "x"}]),
        FakeResponse([call("a")]),
    ])
    content, log, _ = run(solver)
    ok &= check("галлюцинация тула: один вызов и сразу финал", len(log) == 1)
    ok &= check("галлюцинация тула: финал без инструмента",
                journal[-1]["llm"] == "plain")

    # 4. Бюджет шага укорочен и никогда не превышает общий бюджет задачи
    solver, settings, journal = build([FakeResponse([call("a")])])
    run(solver)
    step_budgets = [e["budget"] for e in journal if e["llm"] == "tools"]
    expected = min(settings.agent_step_max_tokens, settings.max_tokens)
    ok &= check(f"бюджет шага = min(agent_step_max_tokens, max_tokens) = {expected}",
                bool(step_budgets) and all(b == expected for b in step_budgets))

    # канонический прогон: общий бюджет 16384 — шаг обязан быть строго меньше
    canon = Settings(_env_file=None, langfuse_enabled=False, max_tokens=16384)
    solver_c, _, journal_c = build([FakeResponse([call("a")])])
    solver_c.settings = canon
    run(solver_c)
    canon_budgets = [e["budget"] for e in journal_c if e["llm"] == "tools"]
    ok &= check("канон 16k: шаг строго меньше общего бюджета",
                bool(canon_budgets) and all(0 < b < canon.max_tokens for b in canon_budgets))

    # 5. Неработающий бэкенд снимает инструмент с первого же отказа:
    #    переформулировка не поможет, а шаги цикла стоят токенов
    def dead(_args):
        raise ToolUnavailable("Arama servisi şu anda çalışmıyor.",
                              {"attempts": [{"error": "TimeoutError"}]})

    solver, _, journal = build([
        FakeResponse([call("a")]), FakeResponse([call("b")]),
        FakeResponse([call("c")]),
    ], search=dead)
    content, log, _ = run(solver)
    ok &= check("мёртвый бэкенд: один вызов и сразу финал", len(log) == 1)
    ok &= check("мёртвый бэкенд: финал без инструмента",
                journal[-1]["llm"] == "plain" and not journal[-1]["has_tools"])
    ok &= check("мёртвый бэкенд: диагностика попала в лог прогона",
                log[0].diag == {"attempts": [{"error": "TimeoutError"}]})

    # 6. Модель ответила сразу валидным JSON — цикл не делает лишних вызовов
    good = '{"solution_steps": "2+2", "final_answer": "4"}'
    solver, _, journal = build([FakeResponse(content=good)])
    content, log, _ = run(solver)
    ok &= check("ответ без поиска: ни одного тул-вызова и один запрос к модели",
                not log and len(journal) == 1)

    # 7. Шаг оборвался посреди размышления: content пуст, тул-вызовов нет.
    #    Раньше цикл выходил с пустотой и уходил в аварийную лестницу
    #    (28% точности на замере V100); теперь добиваем полным бюджетом.
    solver, settings, journal = build([FakeResponse(content="")],
                                      script_plain=[FakeResponse(content=good)])
    content, log, _ = run(solver)
    ok &= check("обрыв на бюджете шага: цикл добивает ответ, а не отдаёт пустоту",
                parse(content) is not None)
    ok &= check("добивание идёт без инструмента и полным бюджетом",
                journal[-1]["llm"] == "plain" and not journal[-1]["has_tools"])

    print("\nИТОГ:", "все проверки пройдены" if ok else "ЕСТЬ ПАДЕНИЯ")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
