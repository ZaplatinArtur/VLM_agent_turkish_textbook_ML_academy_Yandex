"""Схемы бейзлайнов: что должна вернуть модель и что мы отдаём судье."""

from pydantic import BaseModel


class SolveOutput(BaseModel):
    """Строгий JSON, который обязана вернуть модель (guided decoding).

    reasoning — CoT: идёт ПЕРВЫМ полем, чтобы автогрессивная модель сначала
    рассуждала, а потом писала ответ. Поле опционально: промпты v1 его
    не требуют, v2_cot — требует.
    """

    reasoning: str | None = None
    solution_steps: str
    final_answer: str


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float = 0.0


class ToolCallLog(BaseModel):
    tool: str
    args: dict = {}
    result_preview: str | None = None


class SolveResult(BaseModel):
    """Одна строка результирующего JSONL — вход для LLM-as-Judge."""

    task_id: str
    condition: str               # b0_no_tools | b1_search | agent_rag
    model: str
    prompt_version: str
    final_answer: str | None = None
    solution_steps: str | None = None
    reasoning: str | None = None
    # Ответ получен принудительным финалом после исчерпания бюджета токенов
    # (модель не сошлась сама; судье стоит смотреть на такие строже)
    forced_answer: bool = False
    raw_response: str | None = None
    tool_calls: list[ToolCallLog] = []
    usage: Usage = Usage()
    error: str | None = None
