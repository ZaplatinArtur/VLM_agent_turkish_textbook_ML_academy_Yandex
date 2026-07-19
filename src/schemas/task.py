from typing import Literal

from pydantic import BaseModel, Field

from .media import ImageRef


class Task(BaseModel):
    task_id: str
    subject: str  # "math", "physics", ...
    grade: int | None
    question: str
    question_images: list[ImageRef] = Field(default_factory=list)
    reference_answer: str
    answer_type: Literal["numeric", "short_text", "free_form", "choice"]
    reference_solution: str | None = None  # если есть, для judge
