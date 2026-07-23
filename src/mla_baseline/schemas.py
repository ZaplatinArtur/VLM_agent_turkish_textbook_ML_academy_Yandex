"""Схемы бейзлайнов: что должна вернуть модель и что мы отдаём судье."""

from pydantic import BaseModel, Field


class SolveOutput(BaseModel):
    """Строгий JSON, который обязана вернуть модель (guided decoding)."""

    reasoning: str | None = None
    solution_steps: str
    final_answer: str


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float = 0.0


class ToolCallLog(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)
    result_preview: str | None = None
    returned_chunk_ids: list[str] = Field(default_factory=list)
    latency_ms: float | None = None
    error: str | None = None


class SolveResult(BaseModel):
    """Одна строка результирующего JSONL — вход для LLM-as-Judge."""

    task_id: str
    condition: str               # b0_no_tools | b1_search | agent_rag
    model: str
    prompt_version: str
    final_answer: str | None = None
    solution_steps: str | None = None
    reasoning: str | None = None
    forced_answer: bool = False
    raw_response: str | None = None
    generation: dict = Field(default_factory=dict)
    tool_calls: list[ToolCallLog] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None
