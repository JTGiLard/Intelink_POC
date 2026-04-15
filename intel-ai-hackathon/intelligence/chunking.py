from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from intelligence.loaders import LoadedDocument


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    source_type: str
    doc_title: str
    text: str
    char_start: int
    doc_occurred_at: datetime | None


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_document(
    doc: LoadedDocument,
    max_chars: int = 1600,
    overlap: int = 200,
) -> list[TextChunk]:
    paragraphs = _split_paragraphs(doc.text) or [doc.text.strip()]
    chunks: list[TextChunk] = []
    buf: list[str] = []
    buf_len = 0
    abs_start = 0
    chunk_idx = 0

    def emit(body: str, start: int) -> None:
        nonlocal chunk_idx
        cid = f"{doc.doc_id}::c{chunk_idx}"
        chunk_idx += 1
        chunks.append(
            TextChunk(
                chunk_id=cid,
                doc_id=doc.doc_id,
                source_type=doc.source_type,
                doc_title=doc.title,
                text=body,
                char_start=start,
                doc_occurred_at=doc.occurred_at,
            )
        )

    for para in paragraphs:
        plen = len(para)
        if plen > max_chars:
            start = 0
            while start < plen:
                piece = para[start : start + max_chars]
                emit(piece, abs_start + start)
                start += max_chars - overlap
            abs_start += plen + 2
            continue

        if buf_len + plen + (2 if buf else 0) > max_chars and buf:
            body = "\n\n".join(buf)
            emit(body, abs_start)
            abs_start += len(body) + 2
            if overlap > 0 and len(body) > overlap:
                tail = body[-overlap:]
                buf = [tail]
                buf_len = len(tail)
            else:
                buf = []
                buf_len = 0

        buf.append(para)
        buf_len += plen + (2 if len(buf) > 1 else 0)

    if buf:
        body = "\n\n".join(buf)
        emit(body, abs_start)

    return chunks
