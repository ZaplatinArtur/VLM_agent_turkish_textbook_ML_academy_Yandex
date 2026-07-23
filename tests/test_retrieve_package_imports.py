def test_installed_retrieval_namespace_imports() -> None:
    """The agent imports retrieval as an installed top-level package."""

    from retrieve.service import textbook_retrieve, textbook_retrieve_checked

    assert callable(textbook_retrieve)
    assert callable(textbook_retrieve_checked)
