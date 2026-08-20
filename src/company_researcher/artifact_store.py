import asyncio
import os
import re
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

SAFE_EXTENSION_PATTERN = re.compile(r"^[a-z0-9]+$")


class ArtifactStoreError(Exception):
    """Base exception for artifact storage failures."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Raised when stored content does not match its content-addressed key."""


@dataclass(frozen=True)
class StoredArtifact:
    """Stable reference to content held by an artifact store."""

    storage_key: str
    sha256: str
    content_length: int


class ArtifactStore(Protocol):
    """Storage boundary for immutable source artifacts."""

    async def put(self, content: bytes, *, extension: str) -> StoredArtifact:
        """Store content and return its stable reference."""
        ...

    async def get(self, storage_key: str, *, expected_sha256: str) -> bytes:
        """Read content after verifying its expected checksum."""
        ...


class LocalArtifactStore:
    """Store immutable artifacts in a content-addressed local directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    async def put(self, content: bytes, *, extension: str) -> StoredArtifact:
        """Store content once and return its stable content-addressed reference."""
        return await asyncio.to_thread(self._put_sync, content, extension)

    async def get(self, storage_key: str, *, expected_sha256: str) -> bytes:
        """Read a stored artifact and verify its content-addressed identity."""
        return await asyncio.to_thread(
            self._get_sync,
            storage_key,
            expected_sha256,
        )

    def _get_sync(self, storage_key: str, expected_sha256: str) -> bytes:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")

        relative_path = Path(storage_key)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.name.split(".", maxsplit=1)[0] != expected_sha256
        ):
            raise ArtifactStoreError("Artifact storage key is invalid")

        artifact_path = self._root / relative_path
        try:
            content = artifact_path.read_bytes()
        except FileNotFoundError as error:
            raise ArtifactStoreError("Artifact does not exist") from error

        actual_checksum = sha256(content).hexdigest()
        if actual_checksum != expected_sha256:
            raise ArtifactIntegrityError(
                "Artifact does not match its recorded checksum"
            )
        return content

    def _put_sync(self, content: bytes, extension: str) -> StoredArtifact:
        if not content:
            raise ValueError("artifact content must not be empty")

        normalized_extension = extension.strip().lower().removeprefix(".")
        if not SAFE_EXTENSION_PATTERN.fullmatch(normalized_extension):
            raise ValueError("artifact extension must contain only letters and digits")

        checksum = sha256(content).hexdigest()
        relative_path = (
            Path("sha256")
            / checksum[:2]
            / checksum[2:4]
            / f"{checksum}.{normalized_extension}"
        )
        artifact_path = self._root / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        if artifact_path.exists():
            self._verify_existing(artifact_path, checksum)
        else:
            self._write_atomically(artifact_path, content)

        return StoredArtifact(
            storage_key=relative_path.as_posix(),
            sha256=checksum,
            content_length=len(content),
        )

    @staticmethod
    def _verify_existing(artifact_path: Path, expected_checksum: str) -> None:
        actual_checksum = sha256(artifact_path.read_bytes()).hexdigest()
        if actual_checksum != expected_checksum:
            raise ArtifactIntegrityError(
                "Existing artifact does not match its content-addressed key"
            )

    @staticmethod
    def _write_atomically(artifact_path: Path, content: bytes) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=artifact_path.parent,
                prefix=".artifact-",
                delete=False,
            ) as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = Path(temporary_file.name)

            os.replace(temporary_path, artifact_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
