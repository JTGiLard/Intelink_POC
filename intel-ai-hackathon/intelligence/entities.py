from __future__ import annotations

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

    return hits


_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is required. Run: python -m spacy download en_core_web_sm"
            ) from e
    return _nlp


def extract_persons_spacy(text: str) -> list[EntityHit]:
    nlp = _get_nlp()
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
    filtered: list[EntityHit] = []
    seen: set[tuple[str, str, int, int]] = set()
    for h in combined:
        key = (h.label, h.text.strip().lower(), h.span_start, h.span_end)
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


def cooccurrence_edges(
    texts_with_entities: list[tuple[str, list[EntityHit]]],
    min_weight: int = 1,
) -> list[tuple[str, str, int]]:
    """Return weighted undirected edges as (label_a, label_b, weight) with labels like `person:John Tan`."""
    edge_weights: defaultdict[tuple[str, str], int] = defaultdict(int)
    for _, hits in texts_with_entities:
        names = sorted(
            {f"{h.label}:{h.text.strip()}" for h in hits if len(h.text.strip()) > 1},
            key=lambda x: x.lower(),
        )
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if a > b:
                    a, b = b, a
                edge_weights[(a, b)] += 1
    edges = [(a, b, w) for (a, b), w in edge_weights.items() if w >= min_weight]
    edges.sort(key=lambda e: -e[2])
    return edges
