"""Опциональная трассировка прогонов в Langfuse (self-hosted или Cloud).

Включение: MLA_LANGFUSE_ENABLED=true в .env + стандартные переменные Langfuse
(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST). Пакет ставится
экстрой: pip install -e ".[tracing]". При выключенном флаге зависимость
не нужна вовсе — все условия сравнения гоняются одинаково, с трейсингом
или без него.
"""

from .config import Settings


def langchain_callbacks(settings: Settings) -> list:
    """Коллбеки для ChatOpenAI.invoke(config={"callbacks": ...})."""
    if not settings.langfuse_enabled:
        return []
    try:
        import os

        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except ImportError as exc:
        raise RuntimeError(
            "MLA_LANGFUSE_ENABLED=true, но пакет langfuse не установлен: "
            'pip install -e ".[tracing]"'
        ) from exc

    # SDK читает окружение процесса; ключи из .env прокидываем сами
    for env_key, value in (
        ("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key),
        ("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key),
        ("LANGFUSE_HOST", settings.langfuse_host),
    ):
        if value:
            os.environ.setdefault(env_key, value)
    Langfuse()  # инициализация глобального клиента по ключам из окружения
    return [CallbackHandler()]


def flush() -> None:
    """Дослать буферизованные трейсы перед выходом (вызывается runner-ом)."""
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        pass  # трейсинг не должен ронять прогон
