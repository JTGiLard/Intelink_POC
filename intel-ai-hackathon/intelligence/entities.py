from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# International and common local phone patterns
_PHONE_PATTERNS = [
    re.compile(
        r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}(?:[\s.-]?\d{2,4})?"
    ),
    re.compile(r"\b\d{8}\b"),
    re.compile(r"\b\d{10,11}\b"),
]

# Vehicle registration plates (generic + SG-style examples)
_PLATE_PATTERNS = [
    re.compile(r"\b[A-Z]{2,3}\s?\d{2,4}\s?[A-Z]?\b"),
    re.compile(r"\bS[A-Z]{1,2}\d{1,4}[A-Z]?\b", re.IGNORECASE),
    re.compile(r"\b\d{2,4}[A-Z]{1,3}\b"),
]

_VEHICLE_KEYWORDS = re.compile(
    r"\b(vehicle|car|van|lorry|truck|motorcycle|motorbike|Honda|Toyota|Mercedes|BMW|Audi|Hyundai|Kia)\b",
    re.IGNORECASE,
)

_MAKE_MODEL = re.compile(
    r"\b(Honda|Toyota|Mercedes|BMW|Audi|Hyundai|Kia)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
)

_PERSON_FULLNAME = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")
_VEHICLE_WORDS = {
    "vehicle",
    "car",
    "van",
    "lorry",
    "truck",
    "motorcycle",
    "motorbike",
    "honda",
    "toyota",
    "mercedes",
    "bmw",
    "audi",
    "hyundai",
    "kia",
}


@dataclass
class EntityHit:
    label: str  # person | vehicle | phone
    text: str
    span_start: int
    span_end: int


@dataclass
class EntitySummary:
    persons: Counter[str] = field(default_factory=Counter)
    vehicles: Counter[str] = field(default_factory=Counter)
    phones: Counter[str] = field(default_factory=Counter)


def _normalize_phone(raw: str) -> str | None:
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 8:
        return None
    if len(digits) > 15:
        return None
    return digits


def extract_entities_regex(text: str) -> list[EntityHit]:
    hits: list[EntityHit] = []
    seen_spans: set[tuple[int, int, str]] = set()

    def add(label: str, value: str, start: int, end: int) -> None:
        key = (start, end, label)
        if key in seen_spans:
            return
        seen_spans.add(key)
        hits.append(EntityHit(label=label, text=value.strip(), span_start=start, span_end=end))

    for rx in _PHONE_PATTERNS:
        for m in rx.finditer(text):
            norm = _normalize_phone(m.group(0))
            if norm:
                add("phone", m.group(0), m.start(), m.end())

    for rx in _PLATE_PATTERNS:
        for m in rx.finditer(text):
            token = m.group(0).strip()
            if len(token) < 4:
                continue
            if _VEHICLE_KEYWORDS.search(text[max(0, m.start() - 40) : m.end() + 40]) or re.search(
                r"[A-Z]{2,}\d|\d{3,}[A-Z]", token
            ):
                add("vehicle", token, m.start(), m.end())

    for m in _MAKE_MODEL.finditer(text):
        add("vehicle", f"{m.group(1)} {m.group(2)}", m.start(), m.end())

    # Regex fallback for full names when spaCy PERSON extraction is unavailable.
    for m in _PERSON_FULLNAME.finditer(text):
        add("person", m.group(0), m.start(), m.end())

    return hits


_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import warnings

            import spacy
        except Exception:
            _nlp = False
            return None

        try:
            logging.getLogger("spacy").setLevel(logging.ERROR)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _nlp = spacy.load("en_core_web_sm")
        except OSError:
            _nlp = False
            return None
    return _nlp if _nlp is not False else None


def extract_persons_spacy(text: str) -> list[EntityHit]:
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text[:100000])
    hits: list[EntityHit] = []
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.strip()) > 2:
            hits.append(
                EntityHit(
                    label="person",
                    text=ent.text.strip(),
                    span_start=ent.start_char,
                    span_end=ent.end_char,
                )
            )
    return hits


def extract_all_entities(text: str) -> list[EntityHit]:
    combined = extract_entities_regex(text) + extract_persons_spacy(text)
    combined.sort(key=lambda h: (h.span_start, -len(h.text)))
    vehicle_hits = [h for h in combined if h.label == "vehicle" and h.text.strip()]

    def overlaps_vehicle_span(hit: EntityHit) -> bool:
        for vh in vehicle_hits:
            if not (hit.span_end <= vh.span_start or vh.span_end <= hit.span_start):
                return True
        return False

    def looks_like_vehicle_phrase(value: str) -> bool:
        words = {w.lower() for w in re.findall(r"[A-Za-z]+", value)}
        return bool(words & _VEHICLE_WORDS)

    filtered: list[EntityHit] = []
    seen: set[tuple[str, str, int, int]] = set()
    for h in combined:
        text_norm = h.text.strip().lower()
        if h.label == "person" and (overlaps_vehicle_span(h) or looks_like_vehicle_phrase(h.text)):
            continue
        key = (h.label, text_norm, h.span_start, h.span_end)
        if key in seen:
            continue
        seen.add(key)
        filtered.append(h)
    return filtered


def summarize_entities(hits: list[EntityHit]) -> EntitySummary:
    s = EntitySummary()
    for h in hits:
        key = h.text.strip()
        if not key:
            continue
        if h.label == "person":
            s.persons[key] += 1
        elif h.label == "vehicle":
            s.vehicles[key] += 1
        elif h.label == "phone":
            norm = _normalize_phone(key) or key
            s.phones[norm] += 1
    return s


def merge_summaries(*summaries: EntitySummary) -> EntitySummary:
    out = EntitySummary()
    for s in summaries:
        out.persons.update(s.persons)
        out.vehicles.update(s.vehicles)
        out.phones.update(s.phones)
    return out


def _entity_key(h: EntityHit) -> str:
    return f"{h.label}:{h.text.strip()}"


def _span_gap_chars(h1: EntityHit, h2: EntityHit) -> int:
    lo1, hi1 = h1.span_start, h1.span_end
    lo2, hi2 = h2.span_start, h2.span_end
    if hi1 < lo2:
        return lo2 - hi1
    if hi2 < lo1:
        return lo1 - hi2
    return 0


def _chunk_pair_min_gaps(hits: list[EntityHit]) -> dict[tuple[str, str], int]:
    """Per chunk, smallest character gap between spans for each unordered entity pair."""
    out: dict[tuple[str, str], int] = {}
    hits_f = [h for h in hits if len(h.text.strip()) > 1]
    for i in range(len(hits_f)):
        for j in range(i + 1, len(hits_f)):
            ka, kb = _entity_key(hits_f[i]), _entity_key(hits_f[j])
            if ka == kb:
                continue
            a, b = (ka, kb) if ka <= kb else (kb, ka)
            g = _span_gap_chars(hits_f[i], hits_f[j])
            k = (a, b)
            if k not in out or g < out[k]:
                out[k] = g
    return out


def cooccurrence_edges(
    chunks_with_entities: list[tuple[str, list[EntityHit], str | None]],
    min_weight: int = 1,
) -> list[tuple[str, str, float, str, str]]:
    """
    Same-chunk co-occurrence edges with deterministic composite strength.

    Each edge is (entity_a, entity_b, strength, strength_label, link_type) where
    strength_label is Strong / Medium / Weak (relative to the strongest edge in
    this result set) and link_type is direct (close spans in at least one chunk)
    or indirect.
    """
    # (a,b) -> freq chunks, sum of proximity scores, source tags, best (min) gap seen
    freq: dict[tuple[str, str], int] = defaultdict(int)
    prox_sum: dict[tuple[str, str], float] = defaultdict(float)
    sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    min_gap: dict[tuple[str, str], int] = defaultdict(lambda: 10**9)

    for _text, hits, source_tag in chunks_with_entities:
        pairs = _chunk_pair_min_gaps(hits)
        if not pairs:
            continue
        tag = (source_tag or "").strip() or "unknown"
        for (a, b), gap in pairs.items():
            freq[(a, b)] += 1
            prox_sum[(a, b)] += 1.0 / (1.0 + gap / 72.0)
            sources[(a, b)].add(tag)
            if gap < min_gap[(a, b)]:
                min_gap[(a, b)] = gap

    raw_scores: dict[tuple[str, str], float] = {}
    for key in freq:
        if freq[key] < min_weight:
            continue
        n_src = len(sources[key])
        cross = 0.24 * max(0, n_src - 1)
        avg_prox = prox_sum[key] / freq[key]
        raw_scores[key] = freq[key] * (0.28 + 0.72 * avg_prox) + cross

    if not raw_scores:
        return []

    max_raw = max(raw_scores.values())
    if max_raw <= 0:
        max_raw = 1.0

    direct_gap_max = 110
    edges: list[tuple[str, str, float, str, str]] = []
    for (a, b), raw in raw_scores.items():
        ratio = raw / max_raw
        if ratio >= 0.56:
            label = "Strong"
        elif ratio >= 0.30:
            label = "Medium"
        else:
            label = "Weak"
        lt = "direct" if min_gap[(a, b)] <= direct_gap_max else "indirect"
        edges.append((a, b, round(raw, 4), label, lt))

    edges.sort(key=lambda e: -e[2])
    return edges
