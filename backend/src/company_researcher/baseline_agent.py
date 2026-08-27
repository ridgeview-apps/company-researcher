from dataclasses import dataclass

from company_researcher.investigation_agent import Finding
from company_researcher.llm_client import ChatMessage, ChatUsage, UsageAwareChatProvider

_BASELINE_SYSTEM_PROMPT = (
    "You are answering a question about a specific UK company using only "
    "your own general knowledge. You have no access to Companies House "
    "filings, the internet, or any other tool - you cannot look anything "
    "up. If you happen to recall a specific filing page that supports "
    "your answer, you may cite it using its document_extraction_id and "
    "page_number, with an exact quote as supporting_text; otherwise leave "
    "citations empty rather than inventing one. Set evidence_sufficient to "
    "false if you are not confident in your answer rather than guessing. "
    "Classify your answer with claim_type, either 'fact' (states only what "
    "you recall directly) or 'interpretation' (adds a judgement beyond "
    "recalled facts)."
)


@dataclass(frozen=True)
class BaselineAnswer:
    """One question answered with no retrieval, for comparison against the specialized agent."""

    finding: Finding
    usage: ChatUsage | None


async def answer_without_retrieval(
    chat_client: UsageAwareChatProvider, question: str, company_name: str
) -> BaselineAnswer:
    """Answer one question using only the LLM's own knowledge - no retrieval, no tools.

    This is the project brief's "General LLM" baseline (the first of three
    suggested baselines) for measuring whether this project's evidence-
    driven agent actually produces more complete, grounded, and auditable
    answers than simply asking a frontier LLM. It deliberately reuses
    `Finding`, the same structured output `investigation_agent.py`
    produces, so the two paths are directly comparable in shape - a
    citation this path attempts can then be checked against real,
    persisted `DocumentPage` rows the same way any other citation would be
    (see `baseline_comparison.py`), rather than assuming this path simply
    has none.
    """
    user_message = f"Company: {company_name}\n\nQuestion: {question}"
    finding, usage = await chat_client.complete_structured_with_usage(
        [
            ChatMessage(role="system", content=_BASELINE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_message),
        ],
        Finding,
    )
    return BaselineAnswer(finding=finding, usage=usage)
