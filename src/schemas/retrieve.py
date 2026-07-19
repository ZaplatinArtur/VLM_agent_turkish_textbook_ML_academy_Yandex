from pydantic import BaseModel, Field

from .media import ImageRef


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    images: list[ImageRef] = Field(default_factory=list)  # изображения, привязанные к этому чанку
    score: float
    metadata: dict  # textbook, subject, grade, paragraph, page
