import re

_YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def extract_fiscal_years(text: str) -> list[str]:
    """Deterministically extract plain 4-digit years mentioned in `text`.

    Matches a year whether it is written plainly ("2023") or with an "FY"
    prefix ("FY2023") -- the "FY" prefix, if present, is simply not part of
    the match, so both forms yield "2023". Filing text itself never uses an
    "FY" prefix (see `investigation_agent.py`'s query-generation prompt), so
    normalising to the plain form here keeps extracted years directly usable
    as literal lexical-search tokens. Depends only on `text`, so it is
    deterministic and cannot omit a year the way an LLM-generated query
    intermittently does.
    """
    return list(dict.fromkeys(_YEAR_PATTERN.findall(text)))
