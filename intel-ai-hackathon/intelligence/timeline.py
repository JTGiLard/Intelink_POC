from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from intelligence.index_store import ChunkRecord


_WS_TS = re.compile(
    r"^\s*\[(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})[,\s]+(\d{1,2}:\d{2})",
    re.MULTILINE,
)

_ISO_LIKE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?)\b",
)

_TEXTUAL_DATE = re.compile(
    r"\b(\d{1,2})\s+"
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+(\d{4})\b",
    flags=re.IGNORECASE,
)

_NUMERIC_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


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


def _datetime_explicit_in_text(dt: datetime, text: str) -> bool:
    if not text:
        return False
    t = text
    candidates = [
        dt.strftime("%Y-%m-%d"),
        dt.strftime("%d/%m/%Y"),
        dt.strftime("%d-%m-%Y"),
        dt.strftime("%Y/%m/%d"),
        f"{dt.day} {dt.strftime('%b %Y')}",
        f"{dt.day} {dt.strftime('%B %Y')}",
    ]
    tl = t.lower()
    return any(c.lower() in tl for c in candidates if c)


def _drop_suspect_runtime_proximate_events(
    events: list[TimelineEvent],
    text: str,
    reference: datetime,
) -> list[TimelineEvent]:
    """Remove dates within ±1 day of runtime when that date is not substantiated in chunk text."""
    out: list[TimelineEvent] = []
    window_lo = reference - timedelta(days=1)
    window_hi = reference + timedelta(days=1)
    for e in events:
        if not isinstance(e.when, datetime):
            out.append(e)
            continue
        if e.label == "Email date":
            out.append(e)
            continue
        if window_lo <= e.when <= window_hi and not _datetime_explicit_in_text(e.when, text):
            continue
        out.append(e)
    return out


def extract_timeline_events(
    record: ChunkRecord,
    max_events: int = 8,
    *,
    reference_now: datetime | None = None,
) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    text = record.text
    reference = reference_now or datetime.now()

    if (record.source_type or "").lower() == "email":
        hdr = record.occurred_dt()
        if hdr:
            events.append(
                TimelineEvent(
                    when=hdr,
                    label="Email date",
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

    for m in _TEXTUAL_DATE.finditer(text):
        raw = f"{m.group(1)} {m.group(2)} {m.group(3)}"
        dt: datetime | None = None
        for fmt in ("%d %b %Y", "%d %B %Y"):
            try:
                dt = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            continue
        events.append(
            TimelineEvent(
                when=dt,
                label="Event date",
                detail=text[max(0, m.start() - 30) : m.end() + 90].replace("\n", " ")[:200],
                chunk_id=record.chunk_id,
            )
        )

    for m in _NUMERIC_DATE.finditer(text):
        try:
            dt = datetime.strptime(f"{m.group(1)}/{m.group(2)}/{m.group(3)}", "%d/%m/%Y")
        except ValueError:
            continue
        events.append(
            TimelineEvent(
                when=dt,
                label="Event date",
                detail=text[max(0, m.start() - 30) : m.end() + 90].replace("\n", " ")[:200],
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
    dedup = _drop_suspect_runtime_proximate_events(dedup, text, reference)
    return dedup[:max_events]
