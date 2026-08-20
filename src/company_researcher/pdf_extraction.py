import asyncio
import re
from dataclasses import dataclass
from importlib.metadata import version

import pypdfium2 as pdfium  # type: ignore[import-untyped]
import pytesseract  # type: ignore[import-untyped]

LANGUAGE_PATTERN = re.compile(r"^[a-z0-9_+-]+$")


class PdfExtractionError(Exception):
    """Raised when page-aware PDF extraction cannot complete safely."""


@dataclass(frozen=True)
class ExtractedPage:
    """OCR text and provenance for one-indexed PDF page."""

    page_number: int
    text: str
    character_count: int


@dataclass(frozen=True)
class PdfExtractionResult:
    """Page-aware OCR output and the exact extraction configuration."""

    pages: list[ExtractedPage]
    extractor: str
    extractor_version: str
    renderer: str
    renderer_version: str
    language: str
    render_dpi: int
    page_segmentation_mode: int


class TesseractPdfExtractor:
    """Render PDF pages and extract their text with local Tesseract OCR."""

    def __init__(
        self,
        *,
        language: str = "eng",
        render_dpi: int = 300,
        page_segmentation_mode: int = 3,
    ) -> None:
        if not LANGUAGE_PATTERN.fullmatch(language):
            raise ValueError("OCR language contains unsupported characters")
        if render_dpi < 72:
            raise ValueError("render_dpi must be at least 72")
        if not 0 <= page_segmentation_mode <= 13:
            raise ValueError("page_segmentation_mode must be between 0 and 13")

        self._language = language
        self._render_dpi = render_dpi
        self._page_segmentation_mode = page_segmentation_mode

    async def extract(self, pdf_content: bytes) -> PdfExtractionResult:
        """Extract every PDF page without blocking the async event loop."""
        if not pdf_content:
            raise ValueError("PDF content must not be empty")
        return await asyncio.to_thread(self._extract_sync, pdf_content)

    def _extract_sync(self, pdf_content: bytes) -> PdfExtractionResult:
        try:
            document = pdfium.PdfDocument(pdf_content)
        except pdfium.PdfiumError as error:
            raise PdfExtractionError("Could not open PDF for extraction") from error

        pages: list[ExtractedPage] = []
        try:
            for page_index in range(len(document)):
                page = document[page_index]
                try:
                    image = page.render(scale=self._render_dpi / 72).to_pil()
                    text = pytesseract.image_to_string(
                        image,
                        lang=self._language,
                        config=f"--psm {self._page_segmentation_mode}",
                    ).strip()
                except (OSError, pytesseract.TesseractError) as error:
                    raise PdfExtractionError(
                        f"OCR failed for PDF page {page_index + 1}"
                    ) from error
                finally:
                    page.close()

                pages.append(
                    ExtractedPage(
                        page_number=page_index + 1,
                        text=text,
                        character_count=len(text),
                    )
                )
        finally:
            document.close()

        return PdfExtractionResult(
            pages=pages,
            extractor="tesseract",
            extractor_version=str(pytesseract.get_tesseract_version()).splitlines()[0],
            renderer="pypdfium2",
            renderer_version=version("pypdfium2"),
            language=self._language,
            render_dpi=self._render_dpi,
            page_segmentation_mode=self._page_segmentation_mode,
        )
