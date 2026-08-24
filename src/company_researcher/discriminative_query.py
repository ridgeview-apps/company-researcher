from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.db.models import DocumentPage
from company_researcher.query_construction import derive_query

_TEXT_SEARCH_CONFIGURATION = "english"

DEFAULT_MAX_TERMS = 4


async def _document_frequency(session: AsyncSession, word: str) -> int:
    """Count persisted document pages whose text matches `word`."""
    tsquery = func.plainto_tsquery(_TEXT_SEARCH_CONFIGURATION, word)
    tsvector = func.to_tsvector(_TEXT_SEARCH_CONFIGURATION, DocumentPage.text)
    statement = (
        select(func.count()).select_from(DocumentPage).where(tsvector.op("@@")(tsquery))
    )
    result = await session.execute(statement)
    return result.scalar_one()


async def derive_discriminative_query(
    session: AsyncSession, text: str, *, max_terms: int = DEFAULT_MAX_TERMS
) -> str:
    """Build a query from the `max_terms` rarest content words in `text`.

    Starts from `derive_query(text)`'s stopword-filtered content words, then
    ranks them by document frequency across every persisted document page
    (rarer terms first) and keeps only the top `max_terms`. Terms that
    appear on zero pages are dropped entirely: an OR-combined term that
    matches nothing cannot contribute to ranking, and keeping it would just
    waste one of the `max_terms` slots.

    Like `derive_query`, this depends only on `text` and corpus-wide
    statistics -- never on which page is the known-correct answer for a
    specific question -- so a query it produces cannot leak a specific
    answer the way a hand-picked query can.
    """
    content_words = list(dict.fromkeys(derive_query(text).split()))
    frequencies = [
        (word, await _document_frequency(session, word)) for word in content_words
    ]
    present = [(word, frequency) for word, frequency in frequencies if frequency > 0]
    present.sort(key=lambda pair: pair[1])
    return " ".join(word for word, _frequency in present[:max_terms])
