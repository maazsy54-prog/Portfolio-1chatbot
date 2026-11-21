"""Upload validation: size, real file type (by content), hashing, safe names."""
import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import PurePath

from app.errors import BadRequestError, PayloadTooLargeError, UnsupportedMediaTypeError

PDF_MAGIC = b"%PDF-"
ZIP_MAGIC = b"PK\x03\x04"
# Extension -> logical file type. Extend here (txt, md, pptx) for the bonus.
SUPPORTED_EXTENSIONS: dict[str, str] = {".pdf": "pdf", ".docx": "docx"}


@dataclass(frozen=True)
class ValidatedUpload:
    """Result of a successful validation."""

    filename: str  # sanitized, safe for display and storage metadata
    file_type: str  # "pdf" | "docx"
    sha256: str  # content hash used for duplicate detection
    size_bytes: int


def sanitize_filename(name: str) -> str:
    """Strip any path components and unsafe characters from a client filename."""
    base = PurePath(name.replace("\\", "/")).name
    cleaned = re.sub(r"[^\w.\- ]", "_", base).strip(" .")
    return cleaned[:120] or "document"


def detect_file_type(data: bytes) -> str | None:
    """Identify the file type from its bytes, ignoring the filename."""
    if data[:1024].find(PDF_MAGIC) != -1:
        return "pdf"
    if data.startswith(ZIP_MAGIC):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if "word/document.xml" in archive.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            return None
    return None


def validate_upload(filename: str, data: bytes, max_bytes: int) -> ValidatedUpload:
    """Validate an uploaded file or raise the matching AppError.

    Raises:
        BadRequestError: empty file (400).
        PayloadTooLargeError: file over the size limit (413).
        UnsupportedMediaTypeError: unsupported or spoofed type (415).
    """
    if not data:
        raise BadRequestError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise PayloadTooLargeError(
            f"File is too large; the limit is {max_bytes // (1024 * 1024)} MB."
        )

    safe_name = sanitize_filename(filename)
    extension = PurePath(safe_name).suffix.lower()
    expected = SUPPORTED_EXTENSIONS.get(extension)
    if expected is None:
        raise UnsupportedMediaTypeError(
            f"Unsupported file type '{extension or 'none'}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    if detect_file_type(data) != expected:
        raise UnsupportedMediaTypeError(
            f"File content does not match its '{extension}' extension."
        )

    return ValidatedUpload(
        filename=safe_name,
        file_type=expected,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )
