from pathlib import Path

import pytest

from company_researcher.artifact_store import (
    ArtifactIntegrityError,
    ArtifactStoreError,
    LocalArtifactStore,
)


@pytest.mark.asyncio
async def test_put_stores_content_at_stable_content_address(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    stored = await store.put(b"%PDF-1.7 test document", extension=".PDF")

    assert stored.sha256 == (
        "ed6d7f0fa9b39a5cfc80ed24e0c5dbcb99d2e00c4ef884f130e90ed4d5a8bc7b"
    )
    assert stored.storage_key == (
        "sha256/ed/6d/"
        "ed6d7f0fa9b39a5cfc80ed24e0c5dbcb99d2e00c4ef884f130e90ed4d5a8bc7b.pdf"
    )
    assert stored.content_length == 22
    assert (tmp_path / stored.storage_key).read_bytes() == b"%PDF-1.7 test document"


@pytest.mark.asyncio
async def test_put_does_not_rewrite_existing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalArtifactStore(tmp_path)
    first = await store.put(b"same content", extension="pdf")

    def fail_replace(source: Path, destination: Path) -> None:
        pytest.fail("An existing artifact must not be rewritten")

    monkeypatch.setattr("company_researcher.artifact_store.os.replace", fail_replace)

    second = await store.put(b"same content", extension="pdf")

    assert second == first


@pytest.mark.asyncio
async def test_put_rejects_corrupt_existing_artifact(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    stored = await store.put(b"original content", extension="pdf")
    (tmp_path / stored.storage_key).write_bytes(b"corrupt content")

    with pytest.raises(ArtifactIntegrityError):
        await store.put(b"original content", extension="pdf")


@pytest.mark.asyncio
@pytest.mark.parametrize("extension", ["", "../pdf", "pdf/other", "pdf.exe"])
async def test_put_rejects_unsafe_extension(
    tmp_path: Path,
    extension: str,
) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="extension"):
        await store.put(b"content", extension=extension)


@pytest.mark.asyncio
async def test_put_rejects_empty_content(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="empty"):
        await store.put(b"", extension="pdf")


@pytest.mark.asyncio
async def test_get_reads_and_verifies_stored_artifact(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    stored = await store.put(b"verified content", extension="pdf")

    content = await store.get(stored.storage_key, expected_sha256=stored.sha256)

    assert content == b"verified content"


@pytest.mark.asyncio
async def test_get_rejects_corrupt_stored_artifact(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    stored = await store.put(b"original content", extension="pdf")
    (tmp_path / stored.storage_key).write_bytes(b"corrupt content")

    with pytest.raises(ArtifactIntegrityError):
        await store.get(stored.storage_key, expected_sha256=stored.sha256)


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_key", ["../artifact.pdf", "/artifact.pdf"])
async def test_get_rejects_unsafe_storage_key(
    tmp_path: Path,
    storage_key: str,
) -> None:
    with pytest.raises(ArtifactStoreError, match="key"):
        await LocalArtifactStore(tmp_path).get(
            storage_key,
            expected_sha256="a" * 64,
        )
