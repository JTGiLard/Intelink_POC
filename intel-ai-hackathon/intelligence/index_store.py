from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from intelligence.chunking import TextChunk


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source_type: str
    doc_title: str
    text: str
    char_start: int
    doc_occurred_at: str | None

    @staticmethod
    def from_chunk(c: TextChunk) -> ChunkRecord:
        ts = c.doc_occurred_at.isoformat() if c.doc_occurred_at else None
        return ChunkRecord(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            source_type=c.source_type,
            doc_title=c.doc_title,
            text=c.text,
            char_start=c.char_start,
            doc_occurred_at=ts,
        )

    def occurred_dt(self) -> datetime | None:
        if not self.doc_occurred_at:
            return None
        try:
            dt = datetime.fromisoformat(self.doc_occurred_at)
            if dt.tzinfo is not None:
                # Normalize aware values to naive UTC so mixed sources sort safely.
                return dt.astimezone(UTC).replace(tzinfo=None)
            return dt
        except ValueError:
            return None


class FaissIndexStore:
    def __init__(self, vectors: np.ndarray, records: list[ChunkRecord]):
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors.astype(np.float32))
        self.index = index
        self.records = records

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[ChunkRecord, float]]:
        q = query_vec.astype(np.float32).reshape(1, -1)
        scores, idxs = self.index.search(q, min(k, len(self.records)))
        out: list[tuple[ChunkRecord, float]] = []
        for j, s in zip(idxs[0], scores[0]):
            if j < 0:
                continue
            out.append((self.records[int(j)], float(s)))
        return out

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        meta = [asdict(r) for r in self.records]
        (directory / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> FaissIndexStore:
        index = faiss.read_index(str(directory / "index.faiss"))
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        records = [ChunkRecord(**m) for m in meta]
        obj = cls.__new__(cls)
        obj.index = index
        obj.records = records
        return obj


def fingerprint_data_root(data_root: Path) -> str:
    mtimes: list[tuple[str, float]] = []
    for p in sorted(data_root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(data_root))
            mtimes.append((rel, p.stat().st_mtime))
    payload: dict[str, Any] = {"files": mtimes}
    return str(hash(json.dumps(payload, sort_keys=True)))


def cache_dir(base: Path, fp: str) -> Path:
    return base / fp


def try_load_cache(cache_base: Path, fp: str) -> FaissIndexStore | None:
    d = cache_dir(cache_base, fp)
    if not (d / "index.faiss").exists() or not (d / "metadata.json").exists():
        return None
    try:
        return FaissIndexStore.load(d)
    except Exception:
        return None


def save_cache(store: FaissIndexStore, cache_base: Path, fp: str) -> None:
    d = cache_dir(cache_base, fp)
    store.save(d)
