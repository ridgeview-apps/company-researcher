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
    """Build a query from the `max_terms` rarest content words in `text`."""
    # Starts from derive_query(text)'s stopword-filtered content words, then
    # ranks by document frequency across the whole corpus (rarer first).
    # Depends only on text and corpus-wide statistics - never on which page
    # is the known-correct answer - so it can't leak an answer the way a
    # hand-picked query can.
    content_words = list(dict.fromkeys(derive_query(text).split()))
    frequencies = [
        (word, await _document_frequency(session, word)) for word in content_words
    ]
    # Zero-page terms are dropped: an OR-combined term matching nothing
    # can't contribute to ranking, and would just waste a max_terms slot.
    present = [(word, frequency) for word, frequency in frequencies if frequency > 0]
    present.sort(key=lambda pair: pair[1])
    return " ".join(word for word, _frequency in present[:max_terms])
