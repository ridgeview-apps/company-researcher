from io import BytesIO

import pytest
from PIL import Image, ImageDraw, ImageFont

from company_researcher.pdf_extraction import (
    PdfExtractionError,
    TesseractPdfExtractor,
)


def _image_pdf(page_texts: list[str]) -> bytes:
    font = ImageFont.load_default(size=48)
    images: list[Image.Image] = []
    for text in page_texts:
        image = Image.new("RGB", (1600, 500), "white")
        ImageDraw.Draw(image).text((80, 180), text, fill="black", font=font)
        images.append(image)

    output = BytesIO()
    images[0].save(
        output,
        format="PDF",
        save_all=True,
        append_images=images[1:],
        resolution=150,
    )
    return output.getvalue()


@pytest.mark.asyncio
async def test_extract_returns_page_aware_ocr_text_and_provenance() -> None:
    extractor = TesseractPdfExtractor(render_dpi=300)

    result = await extractor.extract(
        _image_pdf(["EVIDENCE DRIVEN RESEARCH", "SECOND SOURCE PAGE"])
    )

    assert len(result.pages) == 2
    assert result.pages[0].page_number == 1
    assert "EVIDENCE DRIVEN RESEARCH" in result.pages[0].text
    assert result.pages[0].character_count == len(result.pages[0].text)
    assert result.pages[1].page_number == 2
    assert "SECOND SOURCE PAGE" in result.pages[1].text
    assert result.extractor == "tesseract"
    assert result.extractor_version
    assert result.renderer == "pypdfium2"
    assert result.renderer_version
    assert result.language == "eng"
    assert result.render_dpi == 300
    assert result.page_segmentation_mode == 3


@pytest.mark.asyncio
async def test_extract_preserves_empty_page() -> None:
    result = await TesseractPdfExtractor().extract(_image_pdf([""]))

    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert result.pages[0].text == ""
    assert result.pages[0].character_count == 0


@pytest.mark.asyncio
async def test_extract_rejects_invalid_pdf() -> None:
    with pytest.raises(PdfExtractionError, match="Could not open PDF"):
        await TesseractPdfExtractor().extract(b"not a PDF")


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"language": "eng;command"}, "language"),
        ({"render_dpi": 71}, "render_dpi"),
        ({"page_segmentation_mode": 14}, "page_segmentation_mode"),
    ],
)
def test_extractor_rejects_invalid_configuration(
    kwargs: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        TesseractPdfExtractor(**kwargs)  # type: ignore[arg-type]
