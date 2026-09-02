from typing import Literal

from pydantic import BaseModel, ConfigDict

from company_researcher.llm_client import ChatMessage, ChatProvider

_ENTAILMENT_JUDGE_SYSTEM_PROMPT = (
    "You are checking whether a cited excerpt from a company's filing "
    "genuinely substantiates a specific factual claim. You will be given "
    "the claim, the exact quoted excerpt a citation used as its "
    "supporting_text, and the full text of the page that excerpt came "
    "from (for context - the excerpt is quoted exactly from somewhere on "
    "this page). Decide whether the excerpt, in the context of the full "
    "page, actually substantiates the specific fact the claim attributes "
    "to it - not merely whether the excerpt is on-topic. Trust a "
    "computation the page performs itself: if the page states two line "
    "items and their own stated total, treat that total as supported by "
    "those line items rather than second-guessing the filer's own "
    "arithmetic. A claim is unsupported if it states a figure or fact "
    "for a different year, party, or category than the excerpt actually "
    "states; attributes a statement to the wrong party (for example "
    "treating the independent auditor's own conclusion as if the "
    "directors stated it, or vice versa); or asserts a reason, cause, or "
    "conclusion the excerpt does not itself state. A reasonable, "
    "non-misleading rounding of a figure is supported. Respond with a "
    "verdict of exactly 'supported' or 'unsupported', and a one-sentence "
    "reason consistent with that verdict."
)


class EntailmentJudgment(BaseModel):
    """A judge's verdict on whether a citation's excerpt substantiates a claim."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["supported", "unsupported"]
    reason: str


async def judge_entailment(
    chat_client: ChatProvider,
    *,
    claim: str,
    supporting_text: str,
    page_text: str,
) -> EntailmentJudgment:
    """Ask the judge whether `supporting_text`, in the context of its full page, substantiates `claim`.

    This is a calibration-only judge design: it is never called from
    `investigation_agent.py`'s live citation-validation path. See
    `judge_calibration.py` and README.md's "Calibrating an LLM judge"
    section for why - this is the same entailment-checking idea the
    project previously built, measured as unreliable on real runs, and
    reverted, being re-tested here only to produce an honest, deterministic
    agreement measurement against human labels before any decision is made
    about whether it belongs in the live pipeline at all.
    """
    user_message = (
        f"Claim: {claim}\n\n"
        f"Cited excerpt (supporting_text): {supporting_text}\n\n"
        f"Full page text:\n{page_text}"
    )
    return await chat_client.complete_structured(
        [
            ChatMessage(role="system", content=_ENTAILMENT_JUDGE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_message),
        ],
        EntailmentJudgment,
    )
