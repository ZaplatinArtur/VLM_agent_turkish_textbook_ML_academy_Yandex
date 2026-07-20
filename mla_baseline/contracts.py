"""Общекомандные контракты (владелец — вся команда, не менять в одиночку).

Источник: договорённость с командами ретрива и данных.
"""

from typing import Literal

from pydantic import BaseModel


class ImageRef(BaseModel):
    image_id: str
    format: Literal["base64", "url", "file_path"]
    data: str                    # base64-строка, URL или путь
    mime_type: str               # "image/png", "image/jpeg"
    caption: str | None = None   # подпись из учебника, если есть


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    images: list[ImageRef] = []  # изображения, привязанные к этому чанку
    score: float
    metadata: dict               # textbook, subject, grade, paragraph, page


class Task(BaseModel):
    task_id: str
    subject: str                 # "math", "physics", ...
    grade: int | None
    question: str
    question_images: list[ImageRef] = []
    reference_answer: str
    answer_type: Literal["numeric", "short_text", "free_form", "choice"]
    reference_solution: str | None = None  # если есть, для judge
