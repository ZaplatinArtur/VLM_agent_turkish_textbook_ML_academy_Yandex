"""Настройки читаются из окружения / .env с префиксом MLA_.

Один и тот же код работает локально (проверка) и на GPU-машине (прогоны) —
меняется только .env.
"""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
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
    # Qwen для thinking-моделей не рекомендует greedy (temp=0): вырождается
    # в бесконечные повторы. Рекомендация вендора: temp 0.6, top_p 0.95,
    # top_k 20, а против повторов — presence_penalty до 1.5.
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    request_timeout_s: float = 300.0

    # Аблация «без reasoning»: thinking выключен во ВСЕХ вызовах
    # (chat_template_kwargs.enable_thinking=false), бюджеты не меняются
    disable_thinking: bool = False

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

    # B1: веб-поиск (self-hosted SearXNG с JSON API)
    searxng_url: str = "http://localhost:8080"
    search_k: int = 5                # результатов на запрос
    agent_max_steps: int = 6         # максимум итераций ReAct-цикла
    # b1_routed: предметы (через запятую), где поиск отключён — по данным
    # прогонов поиск вредит вычислительным задачам и помогает знаниевым
    b1_no_search_subjects: str = "Math"

    # b1_deep: полный поисковый стек (страницы + реранкер вместо сниппетов)
    rerank_url: str = "http://localhost:8002"
    deep_search_pages: int = 8      # сколько URL читать целиком
    deep_search_chunks: int = 6     # сколько верхних фрагментов отдавать модели

    # agent_rag: BM25-сервер команды ретрива (vlm_judge.retrieval_server)
    textbook_search_url: str = "http://localhost:8770"
    rag_top_k: int = 5              # фрагментов на запрос
    rag_max_calls: int = 3          # лимит обращений к корпусу на задачу
    rag_max_context_chars: int = 6000  # символов корпуса в контекст (контракт ретрива)

    # Трассировка в Langfuse. Ключи читаем из стандартных имён (без MLA_-префикса),
    # чтобы .env выглядел как в доке Langfuse; tracing.py прокинет их в SDK.
    langfuse_enabled: bool = False
    langfuse_public_key: str | None = Field(
        None, validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: str | None = Field(
        None, validation_alias=AliasChoices("LANGFUSE_SECRET_KEY"))
    langfuse_host: str | None = Field(
        None, validation_alias=AliasChoices("LANGFUSE_HOST"))


def get_settings() -> Settings:
    return Settings()
