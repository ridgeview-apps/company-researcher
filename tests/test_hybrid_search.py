from company_researcher.hybrid_search import reciprocal_rank_fusion
from company_researcher.lexical_search import PageMatch as LexicalPageMatch
from company_researcher.vector_search import PageMatch as VectorPageMatch


def test_reciprocal_rank_fusion_ranks_a_page_present_in_both_lists_first() -> None:
    lexical_matches = [
        LexicalPageMatch(document_extraction_id=1, page_number=1, rank=0.9),
        LexicalPageMatch(document_extraction_id=1, page_number=2, rank=0.1),
    ]
    vector_matches = [
        VectorPageMatch(document_extraction_id=1, page_number=2, distance=0.1),
        VectorPageMatch(document_extraction_id=1, page_number=1, distance=0.9),
    ]

    fused = reciprocal_rank_fusion(lexical_matches, vector_matches)

    assert [match.page_number for match in fused] == [1, 2]
    assert fused[0].lexical_rank == 1
    assert fused[0].vector_rank == 2
    assert fused[1].lexical_rank == 2
    assert fused[1].vector_rank == 1


def test_reciprocal_rank_fusion_ranks_top_of_one_list_above_bottom_of_the_other() -> (
    None
):
    lexical_matches = [
        LexicalPageMatch(document_extraction_id=1, page_number=1, rank=1.0)
    ]
    vector_matches = [
        VectorPageMatch(document_extraction_id=1, page_number=2, distance=0.0),
        VectorPageMatch(document_extraction_id=1, page_number=3, distance=0.1),
        VectorPageMatch(document_extraction_id=1, page_number=1, distance=0.2),
    ]

    fused = reciprocal_rank_fusion(lexical_matches, vector_matches)

    assert [match.page_number for match in fused] == [1, 2, 3]


def test_reciprocal_rank_fusion_keeps_a_page_present_in_only_one_list() -> None:
    lexical_matches = [
        LexicalPageMatch(document_extraction_id=1, page_number=1, rank=0.5)
    ]
    vector_matches: list[VectorPageMatch] = []

    fused = reciprocal_rank_fusion(lexical_matches, vector_matches)

    assert len(fused) == 1
    assert fused[0].page_number == 1
    assert fused[0].lexical_rank == 1
    assert fused[0].vector_rank is None
    assert fused[0].score == 1 / 61


def test_reciprocal_rank_fusion_returns_empty_for_two_empty_lists() -> None:
    assert reciprocal_rank_fusion([], []) == []


def test_reciprocal_rank_fusion_k_dampens_rank_differences() -> None:
    lexical_matches = [
        LexicalPageMatch(document_extraction_id=1, page_number=1, rank=1.0),
        LexicalPageMatch(document_extraction_id=1, page_number=2, rank=0.5),
    ]

    fused_small_k = reciprocal_rank_fusion(lexical_matches, [], k=0)
    fused_large_k = reciprocal_rank_fusion(lexical_matches, [], k=1000)

    small_k_gap = fused_small_k[0].score - fused_small_k[1].score
    large_k_gap = fused_large_k[0].score - fused_large_k[1].score
    assert small_k_gap > large_k_gap
