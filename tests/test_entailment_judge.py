from collections.abc import Sequence
from typing import TypeVar, cast

import pytest
from pydantic import BaseModel

from company_researcher.entailment_judge import EntailmentJudgment, judge_entailment
from company_researcher.llm_client import ChatMessage

_StructuredResponse = TypeVar("_StructuredResponse", bound=BaseModel)


class FakeJudgeChatClient:
    """Returns a fixed `EntailmentJudgment` and records the messages it was sent."""

    def __init__(self, judgment: EntailmentJudgment) -> None:
        self._judgment = judgment
        self.complete_structured_calls: list[Sequence[ChatMessage]] = []

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        raise AssertionError("judge_entailment should not call complete()")

    async def complete_structured(
        self, messages: Sequence[ChatMessage], response_model: type[_StructuredResponse]
    ) -> _StructuredResponse:
        self.complete_structured_calls.append(messages)
        return cast(_StructuredResponse, self._judgment)


@pytest.mark.asyncio
async def test_judge_entailment_includes_claim_quote_and_full_page_in_the_prompt() -> (
    None
):
    judgment = EntailmentJudgment(verdict="supported", reason="Matches the page.")
    chat_client = FakeJudgeChatClient(judgment)

    result = await judge_entailment(
        chat_client,
        claim="Turnover was 100.",
        supporting_text="Turnover 100",
        page_text="Full page text mentioning Turnover 100 among other things.",
    )

    assert result == judgment
    assert len(chat_client.complete_structured_calls) == 1
    user_message = chat_client.complete_structured_calls[0][-1].content
    assert "Turnover was 100." in user_message
    assert "Turnover 100" in user_message
    assert "Full page text mentioning Turnover 100 among other things." in user_message
