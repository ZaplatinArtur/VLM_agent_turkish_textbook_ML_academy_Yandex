"""Настройки читаются из окружения / .env с префиксом MLA_."""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MLA_",
        extra="ignore",
        populate_by_name=True,
    )

    # OpenAI-compatible LLM backend. Inference is remote through OpenRouter.
    llm_provider: Literal["vllm", "openrouter"] = "openrouter"

    # Local vLLM endpoint.
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "EMPTY"
    # Qwen/Qwen3.5-9B мультимодальна сама по себе (отдельного VL-варианта нет).
    # Локально через Ollama имя другое, напр. qwen3.5:9b-q4_K_M — задаётся в .env.
    model_name: str = "Qwen/Qwen3.5-9B"

    # OpenRouter uses its own canonical model slug and a secret from the environment.
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model_name: str = "qwen/qwen3.5-9b"
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENROUTER_API_KEY",
            "MLA_OPENROUTER_API_KEY",
        ),
    )

    max_tokens: int = 3072
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    request_timeout_s: float = 300.0
    enable_thinking: bool = False

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
    # Text-only experiment: ignore image refs and always send the question text.
    text_only: bool = False

    data_root: Path = Path("data")
    results_dir: Path = Path("results")
    concurrency: int = 1

    # Optional HTTP retrieval adapter. AgentRag uses direct local retrieval by default.
    retrieval_base_url: str = "http://127.0.0.1:8770"
    retrieval_timeout_s: float = 10.0
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_fetch_k: int = Field(default=200, ge=1, le=1_000)
    retrieval_mmr_enabled: bool = False
    retrieval_mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieval_context_order: Literal["score", "edge"] = "score"
    retrieval_max_context_chars: int = 6_000
    retrieval_max_calls: int = Field(default=2, ge=1, le=2)
    # E4: subjects routed directly to the no-tools solver.
    rag_no_retrieval_subjects: str = "Math"

    # B1: веб-поиск через self-hosted SearXNG.
    searxng_url: str = "http://localhost:8080"
    search_k: int = 5
    # Устойчивость поиска. По разбору логов (reports/web_search_diag.txt)
    # пустая выдача шла окнами по 100% и не зависела от формулировки запроса:
    # отваливался бэкенд. Отсюда ретраи, пауза между запросами (бурст от
    # параллельных задач банят движки), кэш повторов и размыкатель цепи.
    searx_language: str = "tr"       # пусто — без ограничения языка
    searx_fallback_engines: str = ""  # напр. "duckduckgo,brave" — третья ступень
    searx_timeout_s: float = 30.0
    searx_retries: int = 2           # доп. попыток на ступень лестницы
    searx_backoff_s: float = 1.5     # пауза перед повтором (× номер попытки)
    # Минимум между запросами к инстансу. Измерено на живой пробе: при паузе
    # 0.5 с движки банят инстанс за десяток запросов (пустых 64%), при 6 с
    # они успевают отпустить (пустых 20%). Прогон это удлиняет, но поиск,
    # который не ищет, стоит дороже.
    searx_min_interval_s: float = 3.0
    searx_cache_ttl_s: float = 900.0   # 26% запросов в прогонах повторялись
    # Инстанс, отдавший выдачу за последние N секунд, считается живым: его
    # пустота — «не нашлось», а не поломка. Без этого правила клиент принимал
    # за поломку любую пустоту с отвалившимися движками, а на живом инстансе
    # с широким пулом пара движков лежит почти всегда.
    searx_alive_window_s: float = 300.0
    searx_unavailable_streak: int = 3  # отказов подряд до размыкания цепи
    searx_cooldown_s: float = 60.0     # пауза при разомкнутой цепи
    agent_max_steps: int = 6         # максимум итераций ReAct-цикла
    # Бюджет ОДНОГО шага цикла. Разбор прогонов (tool_errors_analysis.md):
    # при общем бюджете на шаг модель сжигала все 16k на первом же шаге и
    # не доходила до ответа (44 задачи agent_rag: 31.8% против 54.5% у B0).
    # Промежуточным шагам — короткий бюджет, финальному ответу — полный.
    # 4096 оказалось мало, когда у модели включено размышление: замер
    # b1_search на V100 показал, что 49 из 89 «сорванных» задач упирались
    # ровно в этот лимит посреди <think> и возвращали пустой content.
    agent_step_max_tokens: int = 8192
    search_max_calls: int = 3        # лимит веб-поисков на задачу (как у rag)
    # b1_routed: предметы (через запятую), где поиск отключён — по данным
    # прогонов поиск вредит вычислительным задачам и помогает знаниевым
    b1_no_search_subjects: str = "Math"

    # B1-deep: чтение страниц и внешний reranker.
    rerank_url: str = "http://localhost:8002"
    deep_search_pages: int = 8
    deep_search_chunks: int = 6

    # agent_rag: BM25-сервер команды ретрива (vlm_judge.retrieval_server)
    textbook_search_url: str = "http://localhost:8770"
    rag_top_k: int = 5              # фрагментов на запрос
    rag_max_calls: int = 3          # лимит обращений к корпусу на задачу
    rag_max_context_chars: int = 6000  # символов корпуса в контекст (контракт ретрива)

    # Трассировка в Langfuse. Ключи читаем из стандартных имён (без MLA_-префикса),
    # чтобы .env выглядел как в доке Langfuse; tracing.py прокинет их в SDK.
    langfuse_enabled: bool = False
    langfuse_public_key: str | None = Field(
        None,
        validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY"),
    )
    langfuse_secret_key: str | None = Field(
        None,
        validation_alias=AliasChoices("LANGFUSE_SECRET_KEY"),
    )
    langfuse_host: str | None = Field(
        None,
        validation_alias=AliasChoices("LANGFUSE_HOST"),
    )

    @property
    def llm_base_url(self) -> str:
        if self.llm_provider == "openrouter":
            return self.openrouter_base_url
        return self.vllm_base_url

    @property
    def llm_model_name(self) -> str:
        if self.llm_provider == "openrouter":
            return self.openrouter_model_name
        return self.model_name

    @property
    def llm_api_key(self) -> str:
        if self.llm_provider == "openrouter":
            if self.openrouter_api_key is None:
                raise ValueError(
                    "OPENROUTER_API_KEY is required when MLA_LLM_PROVIDER=openrouter"
                )
            return self.openrouter_api_key.get_secret_value()
        return self.vllm_api_key


def get_settings() -> Settings:
    return Settings()
