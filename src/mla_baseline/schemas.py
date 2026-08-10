"""Схемы бейзлайнов: что должна вернуть модель и что мы отдаём судье."""

from pydantic import BaseModel, ConfigDict, Field


class SolveOutput(BaseModel):
    """Строгий JSON, который обязана вернуть модель (guided decoding)."""

    reasoning: str | None = None
    solution_steps: str
    final_answer: str


class CompactSolveOutput(BaseModel):
    """Short tool-disabled final response used after the RAG loop."""

    model_config = ConfigDict(extra="forbid")

    solution_steps: str = Field(
        max_length=800,
        description="At most three short Turkish sentences.",
    )
    final_answer: str = Field(
        max_length=120,
        description="Only the concise final answer; one choice letter when applicable.",
    )


class FinalAnswerOnly(BaseModel):
    """Last-resort schema when even the compact solution reaches its limit."""

    model_config = ConfigDict(extra="forbid")

    final_answer: str = Field(
        max_length=120,
        description="Only the concise final answer; one choice letter when applicable.",
    )


class ImageTaskEvidence(BaseModel):
    """Structured facts extracted from the authoritative task image."""

    model_config = ConfigDict(extra="forbid")

    image_evidence: list[str]
    question: str
    topic: str = Field(min_length=1, max_length=200)
    unknown_concepts: list[str]


class RetrievalConflictCheck(BaseModel):
    """Chunk IDs whose content contradicts the task image."""

    model_config = ConfigDict(extra="forbid")

    conflicting_chunk_ids: list[str]
    reason: str


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
    relevance: dict | None = None
    diag: dict | None = None
    error: str | None = None


class SolveResult(BaseModel):
    """Одна строка результирующего JSONL — вход для LLM-as-Judge."""

    task_id: str
    condition: str  # b0_no_tools | b1_search | agent_rag | agent_rag_routed
    model: str
    prompt_version: str
    final_answer: str | None = None
    solution_steps: str | None = None
    reasoning: str | None = None
    forced_answer: bool = False
    raw_response: str | None = None
    exit_reason: str | None = None
    image_evidence: list[str] = Field(default_factory=list)
    image_evidence_structured: dict | None = None
    retrieval_relevance: str | None = None
    retrieval_conflict: bool | None = None
    answer_source: str | None = None
    generation: dict = Field(default_factory=dict)
    tool_calls: list[ToolCallLog] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None
