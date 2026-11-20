"""PDF loading and chunking with access-control metadata.

Folder layout (per role)::

    data/<role>/*.pdf               -> shared by all tenants (tenant_id = GLOBAL)
    data/<role>/<tenant_id>/*.pdf   -> visible only to users of <tenant_id>

Every chunk carries the metadata the retrieval layer uses to enforce access:
``role_scope``, ``allowed_roles`` and ``tenant_id``, plus citation fields
``source`` (filename) and ``page`` (1-based).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.config import GLOBAL_TENANT_ID, Role

logger = logging.getLogger(__name__)

TENANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class DocumentLoadError(RuntimeError):
    """Base error for document loading problems."""


class EmptyFolderError(DocumentLoadError):
    """Raised when a role folder is missing or has no PDF files."""


class InvalidPDFError(DocumentLoadError):
    """Raised when a PDF cannot be opened or yields no extractable text."""


@dataclass
class LoadResult:
    """Chunks for one role plus per-file diagnostics."""

    role: Role
    chunks: list[Document] = field(default_factory=list)
    files_loaded: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    pages: int = 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_text(text: str) -> str:
    # Collapse runs of spaces while keeping paragraph breaks for the splitter.
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(path: Path, role: Role, tenant_id: str, data_root: Path) -> list[Document]:
    """Extract one ``Document`` per non-empty page of ``path``.

    Raises:
        InvalidPDFError: if the file is unreadable, encrypted, or has no text.
    """
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise InvalidPDFError(f"{path.name}: encrypted PDFs are not supported")
        raw_pages = list(reader.pages)
    except InvalidPDFError:
        raise
    except (PdfReadError, OSError, ValueError, KeyError) as exc:
        raise InvalidPDFError(f"{path.name}: cannot read PDF ({exc})") from exc

    doc_hash = _file_sha256(path)
    try:
        relative = path.relative_to(data_root).as_posix()
    except ValueError:
        relative = path.name

    pages: list[Document] = []
    for page_number, page in enumerate(raw_pages, start=1):
        try:
            text = _normalise_text(page.extract_text() or "")
        except Exception as exc:  # pypdf can raise many types on malformed pages
            logger.warning("Skipping unreadable page %s of %s: %s", page_number, path.name, exc)
            continue
        if not text:
            continue
        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "source_path": relative,
                    "page": page_number,
                    "total_pages": len(raw_pages),
                    "role_scope": role.value,
                    "allowed_roles": [role.value],
                    "tenant_id": tenant_id,
                    "doc_id": doc_hash[:16],
                },
            )
        )

    if not pages:
        raise InvalidPDFError(
            f"{path.name}: no extractable text (scanned/image-only PDFs need OCR first)"
        )
    return pages


def discover_role_pdfs(role_dir: Path) -> list[tuple[Path, str]]:
    """Return (pdf_path, tenant_id) pairs for a role folder.

    Files directly in the folder are GLOBAL; files in a sub-folder belong to the
    tenant named by that sub-folder. Invalid tenant folder names are skipped.
    """
    found: list[tuple[Path, str]] = []
    for pdf in sorted(role_dir.glob("*.pdf")) + sorted(role_dir.glob("*.PDF")):
        found.append((pdf, GLOBAL_TENANT_ID))
    for sub in sorted(p for p in role_dir.iterdir() if p.is_dir()):
        if sub.name.startswith("."):
            continue
        if not TENANT_ID_RE.match(sub.name) or sub.name.upper() == GLOBAL_TENANT_ID:
            logger.warning("Skipping folder with invalid tenant id: %s", sub)
            continue
        for pdf in sorted(sub.glob("*.pdf")) + sorted(sub.glob("*.PDF")):
            found.append((pdf, sub.name))
    # De-duplicate (case-insensitive filesystems match both globs).
    seen: set[Path] = set()
    unique: list[tuple[Path, str]] = []
    for pdf, tenant in found:
        key = pdf.resolve()
        if key not in seen:
            seen.add(key)
            unique.append((pdf, tenant))
    return unique


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def load_role_documents(
    role: Role,
    data_root: Path,
    chunk_size: int,
    chunk_overlap: int,
    strict: bool = False,
) -> LoadResult:
    """Load and chunk every PDF for ``role``.

    Args:
        strict: if True, the first invalid PDF raises instead of being recorded.

    Raises:
        EmptyFolderError: the role folder is missing or contains no PDFs, or no
            PDF produced any text.
        InvalidPDFError: only when ``strict`` is True.
    """
    role_dir = data_root / role.folder_name
    if not role_dir.is_dir():
        raise EmptyFolderError(f"Data folder for {role.value} not found: {role_dir}")

    pdfs = discover_role_pdfs(role_dir)
    if not pdfs:
        raise EmptyFolderError(f"No PDF files found for {role.value} in {role_dir}")

    result = LoadResult(role=role)
    splitter = build_splitter(chunk_size, chunk_overlap)

    for pdf_path, tenant_id in pdfs:
        try:
            pages = extract_pdf_pages(pdf_path, role, tenant_id, data_root)
        except InvalidPDFError as exc:
            if strict:
                raise
            logger.error("%s", exc)
            result.errors[pdf_path.name] = str(exc)
            continue

        result.pages += len(pages)
        result.files_loaded.append(pdf_path.name)
        # Splitting page-by-page keeps every chunk tied to exactly one page.
        for chunk_index, chunk in enumerate(splitter.split_documents(pages)):
            chunk.metadata["chunk_id"] = f"{chunk.metadata['doc_id']}-p{chunk.metadata['page']}-c{chunk_index}"
            result.chunks.append(chunk)

    if not result.chunks:
        raise EmptyFolderError(
            f"No readable text found in any PDF for {role.value}. Errors: {result.errors}"
        )
    return result
