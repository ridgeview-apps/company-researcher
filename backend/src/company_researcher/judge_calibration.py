import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.db.models import (
    DocumentExtraction,
    DocumentPage,
    Filing,
    FilingDocument,
)
from company_researcher.entailment_judge import judge_entailment
from company_researcher.llm_client import ChatProvider

Verdict = Literal["supported", "unsupported"]


class JudgeCalibrationError(Exception):
    """Raised when a calibration dataset cannot be resolved against persisted data."""


@dataclass(frozen=True)
class EntailmentExample:
    """One hand-labelled (claim, citation) pair for judge calibration.

    `transaction_id`/`page_number` identify the real, persisted filing page
    the excerpt was drawn from, following the same stable-identifier
    convention the retrieval evaluation datasets already use, so this
    dataset survives a database reseed too.
    """

    id: str
    transaction_id: str
    page_number: int
    claim: str
    supporting_text: str
    human_verdict: Verdict
    human_reason: str


@dataclass(frozen=True)
class EntailmentCalibrationDataset:
    """A labelled citation-entailment calibration corpus for one company."""

    company_number: str
    company_name: str
    examples: tuple[EntailmentExample, ...]


def load_entailment_dataset(path: Path) -> EntailmentCalibrationDataset:
    """Parse a labelled judge-calibration dataset from JSON."""
    payload = json.loads(path.read_text())
    examples = tuple(
        EntailmentExample(
            id=example["id"],
            transaction_id=example["transaction_id"],
            page_number=example["page_number"],
            claim=example["claim"],
            supporting_text=example["supporting_text"],
            human_verdict=example["human_verdict"],
            human_reason=example["human_reason"],
        )
        for example in payload["examples"]
    )
    return EntailmentCalibrationDataset(
        company_number=payload["company_number"],
        company_name=payload["company_name"],
        examples=examples,
    )


async def _resolve_page_text(
    session: AsyncSession, company_number: str, transaction_id: str, page_number: int
) -> str:
    """Look up a labelled example's real, persisted page text by stable filing identifiers."""
    statement = (
        select(DocumentPage.text)
        .join(
            DocumentExtraction,
            DocumentExtraction.id == DocumentPage.document_extraction_id,
        )
        .join(
            FilingDocument,
            FilingDocument.id == DocumentExtraction.filing_document_id,
        )
        .join(Filing, Filing.id == FilingDocument.filing_id)
        .where(
            Filing.company_number == company_number,
            Filing.transaction_id == transaction_id,
            DocumentPage.page_number == page_number,
            DocumentExtraction.status == "succeeded",
        )
    )
    text = await session.scalar(statement)
    if text is None:
        raise JudgeCalibrationError(
            f"No successful extraction page is persisted for transaction_id="
            f"{transaction_id} page_number={page_number}"
        )
    return text


@dataclass(frozen=True)
class ExampleResult:
    """One example's human label alongside the judge's own verdict."""

    example_id: str
    human_verdict: Verdict
    judge_verdict: Verdict
    judge_reason: str

    @property
    def agrees(self) -> bool:
        return self.human_verdict == self.judge_verdict


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


@dataclass(frozen=True)
class CalibrationSummary:
    """Judge-vs-human agreement over a calibration dataset.

    Precision/recall/F1 treat 'unsupported' as the positive class rather
    than reporting only accuracy: this project's own prior, reverted
    attempt at this exact judge failed specifically by sometimes flagging
    a genuinely supported citation as unsupported (a false positive here),
    so collapsing that failure mode into one accuracy number would hide
    the thing most worth measuring.
    """

    per_example: tuple[ExampleResult, ...]
    accuracy: float
    precision_unsupported: float
    recall_unsupported: float
    f1_unsupported: float


def _summarize(per_example: tuple[ExampleResult, ...]) -> CalibrationSummary:
    total = len(per_example)
    correct = sum(1 for result in per_example if result.agrees)
    true_positive = sum(
        1
        for result in per_example
        if result.judge_verdict == "unsupported"
        and result.human_verdict == "unsupported"
    )
    false_positive = sum(
        1
        for result in per_example
        if result.judge_verdict == "unsupported" and result.human_verdict == "supported"
    )
    false_negative = sum(
        1
        for result in per_example
        if result.judge_verdict == "supported" and result.human_verdict == "unsupported"
    )
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = (
        _safe_divide(2 * precision * recall, precision + recall)
        if (precision + recall)
        else 0.0
    )
    return CalibrationSummary(
        per_example=per_example,
        accuracy=_safe_divide(correct, total),
        precision_unsupported=precision,
        recall_unsupported=recall,
        f1_unsupported=f1,
    )


async def evaluate_example(
    session: AsyncSession,
    chat_client: ChatProvider,
    example: EntailmentExample,
    company_number: str,
) -> ExampleResult:
    """Run the judge on one labelled example and compare it against the human verdict."""
    page_text = await _resolve_page_text(
        session, company_number, example.transaction_id, example.page_number
    )
    judgment = await judge_entailment(
        chat_client,
        claim=example.claim,
        supporting_text=example.supporting_text,
        page_text=page_text,
    )
    return ExampleResult(
        example_id=example.id,
        human_verdict=example.human_verdict,
        judge_verdict=judgment.verdict,
        judge_reason=judgment.reason,
    )


async def run_calibration(
    session: AsyncSession,
    chat_client: ChatProvider,
    dataset: EntailmentCalibrationDataset,
) -> CalibrationSummary:
    """Run the judge over every labelled example and summarize its agreement with human labels."""
    per_example = tuple(
        [
            await evaluate_example(
                session, chat_client, example, dataset.company_number
            )
            for example in dataset.examples
        ]
    )
    return _summarize(per_example)
