"""Общие типы инструментов агента."""


class ToolUnavailable(RuntimeError):
    """Бэкенд инструмента не работает — дальнейшие вызовы бессмысленны.

    Отличается от «ничего не нашлось»: там имеет смысл переформулировать
    запрос, здесь — нет, инструмент надо снимать. Разбор прогонов показал,
    что модель этой разницы не видит и продолжает искать в мёртвый бэкенд,
    сжигая шаги цикла (reports/web_search_diag.txt).
    """

    def __init__(self, message_for_model: str, diag: dict | None = None):
        super().__init__(message_for_model)
        self.message_for_model = message_for_model
        self.diag = diag or {}
