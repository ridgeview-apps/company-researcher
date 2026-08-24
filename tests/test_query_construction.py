from company_researcher.query_construction import derive_query


def test_derive_query_strips_stopwords_and_punctuation() -> None:
    text = "What was Gymshark's turnover for the year ended 31 July 2025?"

    assert derive_query(text) == "Gymshark turnover year ended 31 July 2025"


def test_derive_query_keeps_content_word_order() -> None:
    text = "Who were the directors and company secretary?"

    assert derive_query(text) == "directors company secretary"


def test_derive_query_is_pure_function_of_text() -> None:
    """Two different questions sharing content words must derive independently."""
    first = derive_query("What was the turnover in 2022?")
    second = derive_query("What was the turnover in 2025?")

    assert first == "turnover 2022"
    assert second == "turnover 2025"


def test_derive_query_returns_empty_string_for_all_stopword_text() -> None:
    assert derive_query("What is it?") == ""
