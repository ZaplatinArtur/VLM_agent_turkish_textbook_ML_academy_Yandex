from typing import Literal

from pydantic import BaseModel


class ImageRef(BaseModel):
    image_id: str
    format: Literal["base64", "url", "file_path"]
    data: str  # base64-строка, URL или путь
    mime_type: str  # "image/png", "image/jpeg"
    caption: str | None = None  # подпись из учебника, если есть
