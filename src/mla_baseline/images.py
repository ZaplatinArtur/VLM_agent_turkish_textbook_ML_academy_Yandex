"""Преобразование ImageRef (контракт команды) в OpenAI-совместимый content block.

file_path-картинки читаем и кодируем в base64 сами — так vLLM не нужен доступ
к локальным файлам (--allowed-local-media-path) и код одинаково работает
локально и на GPU-машине.
"""

import base64
from pathlib import Path

from .contracts import ImageRef


def image_ref_to_block(ref: ImageRef, data_root: Path) -> dict:
    if ref.format == "url":
        url = ref.data
    elif ref.format == "base64":
        url = f"data:{ref.mime_type};base64,{ref.data}"
    elif ref.format == "file_path":
        path = Path(ref.data)
        if not path.is_absolute():
            path = data_root / path
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        url = f"data:{ref.mime_type};base64,{payload}"
    else:  # pragma: no cover — закрыто Literal-типом контракта
        raise ValueError(f"Unknown image format: {ref.format}")
    return {"type": "image_url", "image_url": {"url": url}}
