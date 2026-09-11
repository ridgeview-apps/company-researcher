import re

_YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def extract_fiscal_years(text: str) -> list[str]:
    """Deterministically extract plain 4-digit years mentioned in `text`."""
    # Matches "2023" or "FY2023" alike - the "FY" prefix, if present, is
    # simply not part of the match, so both yield "2023". Filing text itself
    # never uses an "FY" prefix, so normalising here keeps extracted years
    # directly usable as literal lexical-search tokens. Depends only on
    # text, so unlike an LLM-generated query, it can't intermittently omit one.
    return list(dict.fromkeys(_YEAR_PATTERN.findall(text)))
