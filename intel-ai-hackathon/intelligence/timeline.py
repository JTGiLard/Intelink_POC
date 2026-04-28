from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from intelligence.index_store import ChunkRecord


_WS_TS = re.compile(
    r"^\s*\[(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})[,\s]+(\d{1,2}:\d{2})",
    re.MULTILINE,
)

_ISO_LIKE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?)\b",
)


@dataclass
class TimelineEvent:
    when: datetime | str | None
    label: str
    detail: str
    chunk_id: str


def timeline_sort_key(value: datetime | str | None) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    normalized = text.replace("T", " ")
    for fmt, size in (("%Y-%m-%d %H:%M:%S", 19), ("%Y-%m-%d %H:%M", 16)):
        try:
            return datetime.strptime(normalized[:size], fmt).isoformat(timespec="seconds")
        except ValueError:
            continue
    return text.lower()


def _parse_ws_line_ts(date_part: str, time_part: str) -> datetime | None:
    for fmt in ("%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M", "%d.%m.%Y %H:%M", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(f"{date_part} {time_part}", fmt)
        except ValueError:
            continue
    return None


def extract_timeline_events(record: ChunkRecord, max_events: int = 8) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    text = record.text
    base = record.occurred_dt()

    if base:
        events.append(
            TimelineEvent(
                when=base,
                label="Document time",
                detail=record.doc_title[:120],
                chunk_id=record.chunk_id,
            )
        )

    for m in _WS_TS.finditer(text):
        dt = _parse_ws_line_ts(m.group(1), m.group(2))
        if dt:
            line_end = text.find("\n", m.start())
            snippet = text[m.start() : line_end if line_end != -1 else m.start() + 120]
            events.append(
                TimelineEvent(
                    when=dt,
                    label="Chat",
                    detail=snippet.strip()[:200],
                    chunk_id=record.chunk_id,
                )
            )

    for m in _ISO_LIKE.finditer(text):
        try:
            raw = m.group(1).replace("T", " ")
            if len(raw) == 16:
                dt = datetime.strptime(raw, "%Y-%m-%d %H:%M")
            else:
                dt = datetime.strptime(raw[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        events.append(
            TimelineEvent(
                when=dt,
                label="Timestamp",
                detail=text[max(0, m.start() - 20) : m.end() + 80].replace("\n", " ")[:200],
                chunk_id=record.chunk_id,
            )
        )

    events.sort(key=lambda e: timeline_sort_key(e.when))
    dedup: list[TimelineEvent] = []
    seen: set[tuple[str, str]] = set()
    for e in events:
        key = (timeline_sort_key(e.when), e.detail[:80])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(e)
    return dedup[:max_events]
