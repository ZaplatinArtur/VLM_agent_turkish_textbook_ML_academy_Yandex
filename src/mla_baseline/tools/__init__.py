"""Tools available to the homework-solving agent."""


class ToolUnavailable(RuntimeError):
    """A tool backend is unavailable, so retrying the query is pointless."""

    def __init__(self, message_for_model: str, diag: dict | None = None):
        super().__init__(message_for_model)
        self.message_for_model = message_for_model
        self.diag = diag or {}

from .textbook_search import (
    LocalTextbookSearchClient,
    TextbookSearchBackend,
    TextbookSearchClient,
    TextbookSearchError,
    TextbookSearchInput,
    create_search_textbooks_tool,
    format_search_result_for_model,
)

__all__ = [
    "LocalTextbookSearchClient",
    "TextbookSearchBackend",
    "TextbookSearchClient",
    "TextbookSearchError",
    "TextbookSearchInput",
    "ToolUnavailable",
    "create_search_textbooks_tool",
    "format_search_result_for_model",
]
