import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from lxml.etree import XMLSyntaxError
from pypdf import PdfReader
from pypdf.errors import PyPdfError

SUFFIXES = {".pdf", ".docx", ".md", ".txt"}
MAX_BYTES = 10 * 1024 * 1024
MAX_CHARS = 2_000_000
CHUNK_TOKENS = 256
OVERLAP = 48


@dataclass(frozen=True)
class Section:
    text: str
    location_type: str
    number: int

    def location(self, start: int, end: int) -> str:
        if self.location_type == "page":
            return f"page {self.number}"
        first = self.number + self.text.count("\n", 0, start)
        last = self.number + self.text.count("\n", 0, end)
        if first == last:
            return f"{self.location_type} {first}"
        return f"{self.location_type}s {first}–{last}"


@dataclass(frozen=True)
class Chunk:
    text: str
    location: str


def markdown_sections(text: str) -> Iterator[Section]:
    lines = text.split("\n")
    start = 0
    for index, line in enumerate(lines):
        if index > start and re.match(r"^ {0,3}#{1,6}\s", line):
            yield Section("\n".join(lines[start:index]), "line", start + 1)
            start = index
    yield Section("\n".join(lines[start:]), "line", start + 1)


def sections(source: str, data: bytes) -> list[Section]:
    suffix = PurePosixPath(source).suffix.lower()
    if suffix not in SUFFIXES:
        raise ValueError("Supported document formats are PDF, DOCX, Markdown, and UTF-8 text.")
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Each document must contain 1 byte to 10 MiB.")
    try:
        if suffix == ".pdf":
            pdf = PdfReader(BytesIO(data))
            if pdf.is_encrypted or len(pdf.pages) > 200:
                raise ValueError("Upload an unlocked PDF with at most 200 pages.")
            return [Section(page.extract_text() or "", "page", i) for i, page in enumerate(pdf.pages, 1)]
        if suffix == ".docx":
            with ZipFile(BytesIO(data)) as archive:
                if sum(entry.file_size for entry in archive.infolist()) > 40 * 1024 * 1024:
                    raise ValueError("The expanded DOCX exceeds 40 MiB.")
            document = Document(BytesIO(data))
            blocks = []
            for block in document.iter_inner_content():
                if isinstance(block, Table):
                    rows = [" | ".join(" ".join(cell.text.split()) for cell in row.cells) for row in block.rows]
                    blocks.append(" / ".join(rows))
                else:
                    blocks.append(" ".join(block.text.split()))
            return [Section("\n".join(blocks), "block", 1)]
        text = data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        if "\x00" in text:
            raise ValueError("Text uploads must contain valid UTF-8 prose.")
        return list(markdown_sections(text)) if suffix == ".md" else [Section(text, "line", 1)]
    except (BadZipFile, PackageNotFoundError, PyPdfError, XMLSyntaxError, UnicodeError, KeyError) as exc:
        raise ValueError("The file could not be parsed as the requested document format.") from exc


def chunk_document(
    source: str, data: bytes, offsets_for: Callable[[str], list[tuple[int, int]]]
) -> list[Chunk]:
    parsed = sections(source, data)
    if sum(len(section.text) for section in parsed) > MAX_CHARS:
        raise ValueError("Extracted document text exceeds two million characters.")
    chunks = []
    for section in parsed:
        if not any(char.isalnum() for char in section.text):
            continue
        offsets = offsets_for(section.text)
        for start in range(0, len(offsets), CHUNK_TOKENS - OVERLAP):
            stop = min(start + CHUNK_TOKENS, len(offsets))
            begin_char = offsets[start][0]
            end_char = offsets[stop - 1][1]
            snippet = section.text[begin_char:end_char].strip()
            if snippet:
                chunks.append(Chunk(snippet, section.location(begin_char, end_char)))
            if stop == len(offsets):
                break
    if not chunks:
        raise ValueError("No readable text was extracted. Scanned PDFs need OCR before upload.")
    return chunks
