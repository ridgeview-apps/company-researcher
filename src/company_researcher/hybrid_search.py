from collections.abc import Sequence
from dataclasses import dataclass

from company_researcher.lexical_search import PageMatch as LexicalPageMatch
from company_researcher.vector_search import PageMatch as VectorPageMatch

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class HybridMatch:
    """One document page ranked by Reciprocal Rank Fusion of two rankings.

    `lexical_rank`/`vector_rank` are `None` when the page did not appear in
    that ranking at all, kept alongside the fused `score` so the constituent
    signal behind a fused position stays auditable rather than only visible
    as a single opaque number.
    """

    document_extraction_id: int
    page_number: int
    score: float
    lexical_rank: int | None
    vector_rank: int | None


def reciprocal_rank_fusion(
    lexical_matches: Sequence[LexicalPageMatch],
    vector_matches: Sequence[VectorPageMatch],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[HybridMatch]:
    """Combine a lexical and a vector ranking by rank position, not raw score.

    `ts_rank` and cosine distance are on incomparable, oppositely-oriented
    scales, so combining them by value would require an uncalibrated
    normalization step. RRF instead scores each page by
    `sum(1 / (k + rank))` across whichever ranking(s) it appears in, using
    only rank position — a page missing from one ranking simply contributes
    nothing from that side, rather than being penalized by a guessed value.
    """
    scores: dict[tuple[int, int], float] = {}
    lexical_ranks: dict[tuple[int, int], int] = {}
    vector_ranks: dict[tuple[int, int], int] = {}

    for rank, lexical_match in enumerate(lexical_matches, start=1):
        key = (lexical_match.document_extraction_id, lexical_match.page_number)
        scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
        lexical_ranks[key] = rank

    for rank, vector_match in enumerate(vector_matches, start=1):
        key = (vector_match.document_extraction_id, vector_match.page_number)
        scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
        vector_ranks[key] = rank

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        HybridMatch(
            document_extraction_id=key[0],
            page_number=key[1],
            score=score,
            lexical_rank=lexical_ranks.get(key),
            vector_rank=vector_ranks.get(key),
        )
        for key, score in fused
    ]
