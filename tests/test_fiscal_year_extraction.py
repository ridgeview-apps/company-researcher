from company_researcher.fiscal_year_extraction import extract_fiscal_years


def test_extract_fiscal_years_finds_a_plain_year() -> None:
    assert extract_fiscal_years("What was turnover in 2023?") == ["2023"]


def test_extract_fiscal_years_strips_the_fy_prefix() -> None:
    assert extract_fiscal_years("What happened in FY2023?") == ["2023"]


def test_extract_fiscal_years_finds_multiple_years_in_order() -> None:
    text = "How did turnover change year-over-year from FY2021 through FY2025?"

    assert extract_fiscal_years(text) == ["2021", "2025"]


def test_extract_fiscal_years_deduplicates_repeated_years() -> None:
    text = "Compare the 2022 accounts to the amended 2022 accounts."

    assert extract_fiscal_years(text) == ["2022"]


def test_extract_fiscal_years_returns_empty_list_when_no_year_is_present() -> None:
    assert extract_fiscal_years("Who were the directors and company secretary?") == []
