"""Tools available to the homework-solving agent."""

from .textbook_search import (
    TextbookSearchClient,
    TextbookSearchError,
    TextbookSearchInput,
    create_search_textbooks_tool,
    format_search_result_for_model,
)

__all__ = [
    "TextbookSearchClient",
    "TextbookSearchError",
    "TextbookSearchInput",
    "create_search_textbooks_tool",
    "format_search_result_for_model",
]
