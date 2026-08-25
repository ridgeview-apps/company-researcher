from collections.abc import Sequence
from typing import TypeVar, cast

import pytest
from pydantic import BaseModel

from company_researcher.baseline_agent import answer_without_retrieval
from company_researcher.investigation_agent import Citation, Finding
from company_researcher.llm_client import ChatMessage, ChatUsage

_StructuredResponse = TypeVar("_StructuredResponse", bound=BaseModel)


class FakeUsageAwareChatClient:
    """Returns a fixed structured response and usage for any call."""

    def __init__(self, finding: Finding, usage: ChatUsage | None) -> None:
        self._finding = finding
        self._usage = usage
        self.complete_structured_calls: list[Sequence[ChatMessage]] = []

    async def complete_structured_with_usage(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[_StructuredResponse],
    ) -> tuple[_StructuredResponse, ChatUsage | None]:
        self.complete_structured_calls.append(messages)
        return cast(_StructuredResponse, self._finding), self._usage


@pytest.mark.asyncio
async def test_answer_without_retrieval_returns_the_models_finding_and_usage() -> None:
    finding = Finding(
        claim="Acme Ltd reported strong growth in 2023.",
        evidence_sufficient=True,
        citations=[],
    )
    usage = ChatUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70)
    chat_client = FakeUsageAwareChatClient(finding, usage)

    answer = await answer_without_retrieval(
        chat_client, "How did Acme Ltd perform in 2023?", "Acme Ltd"
    )

    assert answer.finding == finding
    assert answer.usage == usage


@pytest.mark.asyncio
async def test_answer_without_retrieval_includes_the_company_name_and_question() -> (
    None
):
    finding = Finding(claim="Unknown.", evidence_sufficient=False, citations=[])
    chat_client = FakeUsageAwareChatClient(finding, usage=None)

    await answer_without_retrieval(chat_client, "What were the directors?", "Acme Ltd")

    user_message = chat_client.complete_structured_calls[0][-1]
    assert user_message.role == "user"
    assert "Acme Ltd" in user_message.content
    assert "What were the directors?" in user_message.content


@pytest.mark.asyncio
async def test_answer_without_retrieval_passes_through_an_attempted_citation() -> None:
    """The baseline is allowed to attempt a citation - it is not stripped or validated here.

    Its realism is checked separately, against real persisted data, in
    baseline_comparison.py; this function's job is only to run the
    no-retrieval LLM call, not to judge its output.
    """
    fabricated_finding = Finding(
        claim="Acme Ltd's revenue was 5m.",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=999999,
                page_number=1,
                supporting_text="revenue was 5m",
            )
        ],
    )
    chat_client = FakeUsageAwareChatClient(fabricated_finding, usage=None)

    answer = await answer_without_retrieval(
        chat_client, "What was Acme Ltd's revenue?", "Acme Ltd"
    )

    assert answer.finding == fabricated_finding
