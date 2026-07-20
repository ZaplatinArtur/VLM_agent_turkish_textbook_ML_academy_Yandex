"""Настройки читаются из окружения / .env с префиксом MLA_.

Один и тот же код работает локально (проверка) и на GPU-машине (прогоны) —
меняется только .env.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MLA_", extra="ignore")

    # vLLM OpenAI-совместимый эндпоинт
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "EMPTY"
    # Qwen/Qwen3.5-9B мультимодальна сама по себе (отдельного VL-варианта нет).
    # Локально через Ollama имя другое, напр. qwen3.5:9b-q4_K_M — задаётся в .env.
    model_name: str = "Qwen/Qwen3.5-9B"

    max_tokens: int = 3072
    temperature: float = 0.0
    request_timeout_s: float = 300.0
    enable_thinking: bool = False

    prompt_version: str = "v1"

    # Как выбивать строгий JSON из модели:
    #   response_format — OpenAI json_schema (новые vLLM)
    #   guided_json     — extra_body.guided_json (старые vLLM)
    #   none            — только промпт + робастный парсер
    structured_mode: Literal["response_format", "guided_json", "none"] = "response_format"

    # Сценарий «ленивый школьник»: если есть картинка, текст условия не шлём.
    include_question_text_with_images: bool = False

    data_root: Path = Path("data")
    results_dir: Path = Path("results")
    concurrency: int = 4

    # Optional HTTP retrieval adapter. AgentRag uses direct local retrieval by default.
    retrieval_base_url: str = "http://127.0.0.1:8770"
    retrieval_timeout_s: float = 10.0
    retrieval_top_k: int = 5
    retrieval_max_context_chars: int = 6_000
    retrieval_max_calls: int = 3


def get_settings() -> Settings:
    return Settings()
