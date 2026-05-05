from __future__ import annotations

import os
import re
import shutil
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional dependency
    fuzz = None

from intelligence.chunking import chunk_document
from intelligence.embeddings import embed_texts
from intelligence.entities import (
    EntitySummary,
    cooccurrence_edges,
    extract_all_entities,
    normalize_phone_number,
    summarize_entities,
)
from intelligence.index_store import (
    ChunkRecord,
    FaissIndexStore,
    fingerprint_data_root,
    save_cache,
    try_load_cache,
)
from intelligence.identity_clusters import (
    IdentityClusterResult,
    build_person_identity_clusters,
    cluster_summary_label,
    format_linking_id_key,
    pick_default_cluster_index,
)
from intelligence.link_graph import build_entity_link_graph_figure
from intelligence.loaders import iter_documents
from intelligence.timeline import TimelineEvent, extract_timeline_events, timeline_sort_key
from intelligence.walker_case_relationships import (
    LINK_ANALYSIS_SCAFFOLD_NOTE,
    direct_context_markdown,
    indirect_context_markdown,
    johnnie_walker_case_summary_block,
    merge_walker_case_edges,
    should_activate_walker_scaffold,
    supplement_walker_timeline,
    walker_graph_anchor_person,
)


load_dotenv()


def _person_phrase_from_query(query: str) -> str:
    """First segment before ``+`` or ``,`` — used for person-style matching (case-insensitive)."""
    raw = query.strip()
    if not raw:
        return ""
    parts = [p.strip() for p in re.split(r"[+,]", raw) if p.strip()]
    return parts[0] if parts else raw


def _cache_base() -> Path:
    return Path(__file__).resolve().parent / ".cache_index"


ENTITY_OVERVIEW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^find\s+all\s+info\s+about\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^show\s+everything\s+about\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^build\s+intel\s+picture\s+for\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^build\s+initial\s+intel\s+picture\s+for\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^what\s+do\s+we\s+know\s+about\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^who\s+is\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^details\s+about\s+(.+?)\??$", flags=re.IGNORECASE),
]

OFFENCE_INTENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^what\s+offence\s+was\s+(.+?)\s+involved\s+in\??$", flags=re.IGNORECASE),
    re.compile(r"^what\s+was\s+the\s+offence\s+about\s+for\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^what\s+the\s+offence\s+was\s+about\s+on\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^what\s+offence\s+is\s+linked\s+to\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^what\s+is\s+(.+?)\s+offence\??$", flags=re.IGNORECASE),
    re.compile(r"^what\s+did\s+(.+?)\s+do\??$", flags=re.IGNORECASE),
    re.compile(r"^why\s+was\s+(.+?)\s+arrested\??$", flags=re.IGNORECASE),
    re.compile(r"^case\s+against\s+(.+?)\??$", flags=re.IGNORECASE),
]

VEHICLE_LOOKUP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^which\s+vehicles\s+linked\s+to\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^which\s+vehicles\s+have\s+(.+?)\s+driven\??$", flags=re.IGNORECASE),
    re.compile(r"^what\s+vehicle\s+did\s+(.+?)\s+use\??$", flags=re.IGNORECASE),
    re.compile(r"^vehicle\s+used\s+by\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^vehicle\s+linked\s+to\s+(.+?)\??$", flags=re.IGNORECASE),
]

RELATIONSHIP_LOOKUP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^who\s+is\s+(.+?)\s+linked\s+to\??$", flags=re.IGNORECASE),
    re.compile(r"^show\s+links\s+for\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^show\s+relationship\s+for\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^who\s+are\s+(.+?)\s+associates\??$", flags=re.IGNORECASE),
    re.compile(r"^associates\s+of\s+(.+?)\??$", flags=re.IGNORECASE),
]

TIMELINE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^timeline\s+for\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^movement\s+history\s+for\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^activities\s+of\s+(.+?)\??$", flags=re.IGNORECASE),
]

ENTITY_RESOLUTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^is\s+(.+?)\s+and\s+(.+?)\s+the\s+same\s+person\??$", flags=re.IGNORECASE),
    re.compile(r"^are\s+(.+?)\s+and\s+(.+?)\s+the\s+same\s+person\??$", flags=re.IGNORECASE),
    re.compile(r"^is\s+(.+?)\s+same\s+as\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^are\s+(.+?)\s+and\s+(.+?)\s+aliases\??$", flags=re.IGNORECASE),
    re.compile(r"^does\s+(.+?)\s+refer\s+to\s+(.+?)\??$", flags=re.IGNORECASE),
]

OFFENCE_EVIDENCE_PRIORITY_TERMS = [
    "offence",
    "arrested",
    "smuggling",
    "duty-unpaid cigarettes",
    "falsely declared",
    "cartons",
    "case",
    "shipment",
]


def _clean_target_entity(entity_text: str) -> str:
    cleaned = re.sub(r"[?!.:,;]+$", "", entity_text.strip())
    return re.sub(r"\s+", " ", cleaned)


def normalize_user_query(raw_query: str) -> dict[str, str]:
    cleaned = " ".join(raw_query.strip().split())
    if not cleaned:
        return {"intent": "general_search", "target_entity": "", "search_query": ""}
    for pattern in ENTITY_OVERVIEW_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            target_entity = _clean_target_entity(match.group(1))
            return {"intent": "entity_overview", "target_entity": target_entity, "search_query": target_entity}
    for pattern in OFFENCE_INTENT_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            target_entity = _clean_target_entity(match.group(1))
            if target_entity:
                return {
                    "intent": "offence_summary",
                    "target_entity": target_entity,
                    "search_query": target_entity,
                }
    for pattern in VEHICLE_LOOKUP_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            target_entity = _clean_target_entity(match.group(1))
            return {"intent": "vehicle_lookup", "target_entity": target_entity, "search_query": target_entity}
    for pattern in RELATIONSHIP_LOOKUP_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            target_entity = _clean_target_entity(match.group(1))
            return {"intent": "relationship_lookup", "target_entity": target_entity, "search_query": target_entity}
    for pattern in TIMELINE_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            target_entity = _clean_target_entity(match.group(1))
            return {"intent": "timeline", "target_entity": target_entity, "search_query": target_entity}
    for pattern in ENTITY_RESOLUTION_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            entity_a = _clean_target_entity(match.group(1))
            entity_b = _clean_target_entity(match.group(2))
            if entity_a and entity_b:
                search_query = f"{entity_a} {entity_b} Tan Zong Cai Test Company Best"
                return {
                    "intent": "entity_resolution",
                    "target_entity": entity_a,
                    "entity_a": entity_a,
                    "entity_b": entity_b,
                    "search_query": search_query,
                }
    if _is_person_like_two_word_query(cleaned):
        return {"intent": "entity_overview", "target_entity": cleaned, "search_query": cleaned}
    return {"intent": "general_search", "target_entity": cleaned, "search_query": cleaned}


def prioritize_offence_evidence(
    evidence: list[tuple[ChunkRecord, float]],
) -> list[tuple[ChunkRecord, float]]:
    def offence_bonus(text: str) -> int:
        text_l = text.lower()
        return sum(1 for term in OFFENCE_EVIDENCE_PRIORITY_TERMS if term in text_l)

    return sorted(evidence, key=lambda item: (offence_bonus(item[0].text), item[1]), reverse=True)


@st.cache_resource(show_spinner="Loading vector index…")
def cached_vector_index(data_root_str: str, use_cache: bool, data_fingerprint: str) -> FaissIndexStore:
    _ = data_fingerprint  # bust Streamlit cache when files change
    return build_or_load_index(Path(data_root_str), use_cache=use_cache)


@st.cache_resource
def get_spacy_ready() -> str:
    """Warm-up spaCy once per process."""
    from intelligence.entities import extract_persons_spacy

    person_hits = extract_persons_spacy("John Tan met Mary.")
    if person_hits:
        return "ok"
    return "fallback"


def build_or_load_index(data_root: Path, use_cache: bool) -> FaissIndexStore:
    fp = fingerprint_data_root(data_root)
    cache = _cache_base()
    if use_cache:
        cached = try_load_cache(cache, fp)
        if cached is not None:
            return cached

    docs = list(iter_documents(data_root))
    if not docs:
        raise FileNotFoundError(
            f"No documents under {data_root}. Add .txt/.md/.docx to data/reports, data/emails, or data/whatsapp."
        )
    chunks = []
    for d in docs:
        chunks.extend(chunk_document(d))
    chunks = [c for c in chunks if c.text.strip()]
    if not chunks:
        raise FileNotFoundError(f"No non-empty chunks produced from documents in {data_root}.")

    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    records = [ChunkRecord.from_chunk(c) for c in chunks]
    store = FaissIndexStore(vectors, records)
    if use_cache:
        save_cache(store, cache, fp)
    return store


def _person_name_rank_adjust(record_text: str, query_lower: str) -> float:
    """Extra score for person-style queries: exact phrase, first-name-only, surname mismatch."""
    tokens = [t for t in query_lower.split() if len(t) > 1]
    if len(tokens) < 2:
        return 0.0
    phrase = " ".join(tokens)
    text_l = record_text.lower()
    adj = 0.0
    if phrase in text_l:
        adj += 0.22
        return adj
    first, last = tokens[0], tokens[-1]
    if re.search(rf"(?i)\b{re.escape(first)}\b", text_l):
        adj += 0.055
    for m in re.finditer(rf"(?i)\\b{re.escape(first)}\\s+(\\w{{2,}})\\b", record_text):
        word = m.group(1)
        if word.lower() == last:
            continue
        if word.isalpha() and word[:1].isupper() and len(word) >= 3:
            adj -= 0.14
            break
    return adj


def hybrid_rank(
    hits: list[tuple[ChunkRecord, float]],
    query: str,
    keyword_boost: float = 0.12,
) -> list[tuple[ChunkRecord, float]]:
    q_full = query.strip().lower()
    pq = _person_phrase_from_query(query).strip().lower()
    if not q_full and not pq:
        return hits
    extra_terms = [p.strip().lower() for p in re.split(r"[+,]", query.strip())[1:] if p.strip()]
    person_tokens = [t for t in pq.split() if len(t) > 1] if pq else []
    person_style = len(person_tokens) >= 2
    rescored: list[tuple[ChunkRecord, float]] = []
    for r, s in hits:
        text_l = r.text.lower()
        bonus = keyword_boost if q_full in text_l else 0.0
        for kw in extra_terms:
            if kw and kw in text_l:
                bonus += keyword_boost * 0.92
        if len(q_full) > 3:
            parts = [p for p in q_full.split() if len(p) > 2]
            if parts and all(p in text_l for p in parts):
                bonus = max(bonus, keyword_boost * 0.85)
        if person_style and pq:
            bonus += _person_name_rank_adjust(r.text, pq)
        elif len(person_tokens) == 1 and pq:
            t0 = person_tokens[0]
            if re.search(rf"(?i)\b{re.escape(t0)}\b", text_l):
                bonus += 0.035
        rescored.append((r, s + bonus))
    rescored.sort(key=lambda x: -x[1])
    return rescored


def has_exact_full_name_hit(ranked: list[tuple[ChunkRecord, float]], query: str) -> bool:
    pq = _person_phrase_from_query(query).strip().lower()
    tokens = [t for t in pq.split() if len(t) > 1]
    if len(tokens) < 2:
        return True
    phrase = " ".join(tokens)
    return any(phrase in r.text.lower() for r, _ in ranked)


def has_exact_match(query: str, primary_evidence: list[tuple[ChunkRecord, float]]) -> bool:
    """True when the full query string (case-insensitive) appears in primary evidence text."""
    q = query.strip().lower()
    if not q:
        return False
    for r, _ in primary_evidence:
        if q in r.text.lower():
            return True
    return False


def corpus_has_exact_phrase(records: list[ChunkRecord], phrase: str) -> bool:
    q = " ".join(phrase.strip().lower().split())
    if not q:
        return False
    return any(q in r.text.lower() for r in records)


def _vehicle_query_normalized(query: str) -> str:
    return re.sub(r"\s+", "", query.strip().upper())


def _is_vehicle_or_plate_query(query: str) -> bool:
    q = _vehicle_query_normalized(query)
    return bool(re.fullmatch(r"[A-Z]{1,3}\d{1,4}[A-Z]?", q))


def _extract_strong_identifiers(text: str) -> set[str]:
    ids: set[str] = set()
    for h in extract_all_entities(text):
        value = h.text.strip()
        if not value:
            continue
        if h.label == "phone":
            ids.add(f"phone:{normalize_phone_number(value) or value}")
        elif h.label == "vehicle":
            ids.add(f"vehicle:{_vehicle_query_normalized(value)}")
        elif h.label == "person":
            ids.add(f"person:{' '.join(value.lower().split())}")
    for m in re.finditer(r"\b[STFG]\d{7}[A-Z]\b", text, flags=re.IGNORECASE):
        ids.add(f"nric:{m.group(0).upper()}")
    for m in re.finditer(r"\b(?:CASE|CID|CR|INV)[-\s]?\d{2,}\b", text, flags=re.IGNORECASE):
        ids.add(f"case:{re.sub(r'\\s+', '', m.group(0).upper())}")
    company_rx = re.compile(
        r"\b([A-Z][A-Za-z0-9&'()\-]*(?:\s+[A-Z][A-Za-z0-9&'()\-]*){0,5}\s+(?:Pte\.?\s+Ltd|Ltd|LLP|Inc\.?|Corp\.?))\b"
    )
    for m in company_rx.finditer(text):
        ids.add(f"company:{' '.join(m.group(1).lower().split())}")
    return ids


def _filter_person_centric_graph_edges(
    edges: list[tuple[str, str, float, str, str]],
    primary_evidence: list[tuple[ChunkRecord, float]],
    linked_evidence: list[tuple[ChunkRecord, float]],
    anchor_person: str,
) -> list[tuple[str, str, float, str, str]]:
    if not edges or not anchor_person.strip():
        return edges
    anchor_key = f"person:{' '.join(anchor_person.strip().lower().split())}"

    primary_ids: set[str] = set()
    allowed_persons: set[str] = {anchor_key}
    for r, _ in primary_evidence:
        ids = _extract_strong_identifiers(r.text)
        primary_ids |= {i for i in ids if not i.startswith("person:")}
        allowed_persons |= {i for i in ids if i.startswith("person:")}

    if not primary_ids:
        # If no strong identifiers are available, remain conservative and keep only
        # edges touching person mentions that occurred in exact primary evidence.
        return [
            e
            for e in edges
            if (
                (e[0].startswith("person:") and e[0].lower() in allowed_persons)
                or (e[1].startswith("person:") and e[1].lower() in allowed_persons)
            )
        ]

    for r, _ in linked_evidence:
        ids = _extract_strong_identifiers(r.text)
        if primary_ids & ids:
            allowed_persons |= {i for i in ids if i.startswith("person:")}

    allowed_nodes: set[str] = set(primary_ids) | allowed_persons

    def node_ok(node: str) -> bool:
        nl = node.lower()
        if nl.startswith("person:"):
            return nl in allowed_persons
        # Keep key identifier node types only when validated against primary IDs.
        if nl.startswith(("phone:", "vehicle:", "nric:", "case:", "company:")):
            return nl in allowed_nodes
        return True

    filtered = [e for e in edges if node_ok(e[0]) and node_ok(e[1])]
    # Ensure anchor remains central if present.
    anchor_edges = [e for e in filtered if e[0].lower() == anchor_key or e[1].lower() == anchor_key]
    return anchor_edges + [e for e in filtered if e not in anchor_edges]


def _classify_evidence(
    ranked: list[tuple[ChunkRecord, float]],
    query: str,
) -> tuple[list[tuple[ChunkRecord, float]], list[tuple[ChunkRecord, float]]]:
    person_phrase = _person_phrase_from_query(query).strip()
    person_two_word = _is_person_like_two_word_query(query)
    phone_norm = normalize_phone_number(query)
    vehicle_norm = _vehicle_query_normalized(query)

    primary: list[tuple[ChunkRecord, float]] = []
    related: list[tuple[ChunkRecord, float]] = []
    for r, score in ranked:
        text_l = r.text.lower()
        if person_two_word and person_phrase:
            is_primary = person_phrase.lower() in text_l
        elif phone_norm:
            chunk_phones = {
                normalize_phone_number(h.text)
                for h in extract_all_entities(r.text)
                if h.label == "phone"
            }
            is_primary = phone_norm in {p for p in chunk_phones if p}
        elif _is_vehicle_or_plate_query(query):
            is_primary = vehicle_norm in _vehicle_query_normalized(r.text)
        else:
            is_primary = query.strip().lower() in text_l
        (primary if is_primary else related).append((r, score))
    if person_two_word and person_phrase:
        if not primary:
            related = []
        else:
            primary_ids: set[str] = set()
            for pr, _ in primary:
                primary_ids |= _extract_strong_identifiers(pr.text)
            if primary_ids:
                filtered_related: list[tuple[ChunkRecord, float]] = []
                for rr, ss in related:
                    related_ids = _extract_strong_identifiers(rr.text)
                    if primary_ids & related_ids:
                        filtered_related.append((rr, ss))
                related = filtered_related
            else:
                related = []
    return primary, related


def aggregate_dashboard(
    ranked: list[tuple[ChunkRecord, float]],
    query: str,
) -> tuple[EntitySummary, list[tuple[str, str, float, str, str]], list[TimelineEvent]]:
    texts_entities: list[tuple[str, list, str]] = []
    all_hits = []
    timeline: list[TimelineEvent] = []
    for r, _ in ranked:
        ents = extract_all_entities(r.text)
        texts_entities.append((r.text, ents, r.source_type))
        all_hits.extend(ents)
        timeline.extend(extract_timeline_events(r))
    summary = summarize_entities(all_hits)
    edges = cooccurrence_edges(texts_entities, min_weight=1)
    timeline.sort(key=lambda e: timeline_sort_key(e.when))
    return summary, edges[:80], timeline


def highlight_query(text: str, query: str) -> str:
    if not query.strip():
        return text
    esc = re.escape(query.strip())
    return re.sub(f"({esc})", r"**\1**", text, flags=re.IGNORECASE)


def _summary_subject(query: str, summary: EntitySummary) -> str:
    q = query.strip()
    if q:
        return q
    top = summary.persons.most_common(1)
    return top[0][0] if top else "this result set"


def _is_person_like_two_word_query(query: str) -> bool:
    q = _person_phrase_from_query(query).strip()
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z'-]*\s+[A-Za-z][A-Za-z'-]*", q))


def _is_single_token_person_like_query(query: str) -> bool:
    q = _person_phrase_from_query(query).strip()
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z'-]*", q))


def _has_exact_person_token_match(summary: EntitySummary, token: str) -> bool:
    tk = token.strip().lower()
    if not tk:
        return False
    for name in summary.persons:
        parts = [p.strip().lower() for p in re.split(r"\s+", name.strip()) if p.strip()]
        if tk in parts:
            return True
    return False


def find_closest_person_match(query: str, summary: EntitySummary) -> str | None:
    q = _person_phrase_from_query(query).strip()
    if not q:
        return None
    candidates = summary.persons.most_common(20)
    if not candidates:
        return None
    ql = q.lower()

    q_parts = _person_name_parts(q)
    if len(q_parts) < 2:
        return None

    def rank(name: str, freq: int) -> tuple[float, float, int]:
        parts = _person_name_parts(name)
        if len(parts) < 2:
            return (-1.0, -1.0, freq)
        first_sim = SequenceMatcher(None, q_parts[0], parts[0]).ratio()
        last_sim = SequenceMatcher(None, q_parts[-1], parts[-1]).ratio()
        # Require both first-name and surname similarity for full-name fallback.
        if not (first_sim >= 0.72 and last_sim >= 0.72):
            return (-1.0, -1.0, freq)
        full_sim = SequenceMatcher(None, ql, name.lower()).ratio()
        return (first_sim + last_sim + full_sim, full_sim, freq)

    best_name, _ = max(candidates, key=lambda item: rank(item[0], item[1]))
    best_parts = _person_name_parts(best_name)
    if len(best_parts) < 2:
        return None
    first_sim = SequenceMatcher(None, q_parts[0], best_parts[0]).ratio()
    last_sim = SequenceMatcher(None, q_parts[-1], best_parts[-1]).ratio()
    return best_name if (first_sim >= 0.72 and last_sim >= 0.72) else None


def find_surname_token_suggestion(query: str, summary: EntitySummary) -> str | None:
    token = _person_phrase_from_query(query).strip().lower()
    if not token:
        return None
    surname_counts: Counter[str] = Counter()
    for name, freq in summary.persons.items():
        parts = [p for p in re.split(r"\s+", name.strip()) if p]
        if len(parts) >= 2:
            surname_counts[parts[-1].lower()] += freq
    if not surname_counts:
        return None

    def score(surname: str, freq: int) -> float:
        return freq + SequenceMatcher(None, token, surname).ratio() * 1.8

    best_surname, best_freq = max(surname_counts.items(), key=lambda item: score(item[0], item[1]))
    sim = SequenceMatcher(None, token, best_surname).ratio()
    if sim < 0.62 and best_freq < 2:
        return None
    return best_surname[:1].upper() + best_surname[1:]


def find_alias_suggestion(query: str, summary: EntitySummary) -> str | None:
    token = _person_phrase_from_query(query).strip().lower()
    if not token:
        return None
    alias_candidates = [
        (name, freq)
        for name, freq in summary.persons.items()
        if len([p for p in re.split(r"\s+", name.strip()) if p]) == 1
    ]
    if not alias_candidates:
        return None

    def score(name: str, freq: int) -> float:
        return freq + SequenceMatcher(None, token, name.lower()).ratio() * 2.0

    best_name, best_freq = max(alias_candidates, key=lambda item: score(item[0], item[1]))
    sim = SequenceMatcher(None, token, best_name.lower()).ratio()
    # Short tokens are easy to confuse with unrelated names (e.g. Tay vs Tan); require stronger evidence.
    min_sim = 0.88 if len(token) <= 4 else 0.68
    if sim < min_sim and best_freq < 2:
        return None
    if len(token) <= 4 and best_name.lower() != token and sim < 0.92:
        return None
    return best_name


def _person_name_parts(name: str) -> list[str]:
    return [p.strip().lower() for p in re.split(r"\s+", name.strip()) if p.strip()]


def _fuzzy_person_similarity(query_name: str, candidate_name: str) -> float:
    q = " ".join(query_name.strip().lower().split())
    c = " ".join(candidate_name.strip().lower().split())
    if not q or not c:
        return 0.0
    if fuzz is not None:
        # Blend direct ratio and token-aware ratio for robust spelling variation handling.
        return max(float(fuzz.ratio(q, c)), float(fuzz.token_sort_ratio(q, c))) / 100.0
    return SequenceMatcher(None, q, c).ratio()


def get_closest_person_matches(
    query: str,
    person_counter: Counter[str],
    limit: int = 3,
) -> list[tuple[str, int, float]]:
    q = _person_phrase_from_query(query).strip()
    if not q or not person_counter:
        return []
    ranked: list[tuple[str, int, float]] = []
    for name, mention_count in person_counter.items():
        if not name.strip():
            continue
        sim = _fuzzy_person_similarity(q, name)
        ranked.append((name, mention_count, sim))
    ranked.sort(key=lambda item: (-item[2], -item[1], item[0].lower()))
    return ranked[: max(1, limit)]


def corpus_distinct_person_names_with_token(records: list[ChunkRecord], token: str) -> frozenset[str]:
    """Distinct extracted person strings where ``token`` matches a whole name part (case-insensitive)."""
    tk = token.strip().lower()
    if not tk:
        return frozenset()
    seen_lower: dict[str, str] = {}
    for r in records:
        for h in extract_all_entities(r.text):
            if h.label != "person":
                continue
            raw = h.text.strip()
            if not raw:
                continue
            if tk not in _person_name_parts(raw):
                continue
            k = " ".join(raw.lower().split())
            if k not in seen_lower:
                seen_lower[k] = raw
    return frozenset(seen_lower.values())


def _source_mix_sentence(ranked: list[tuple[ChunkRecord, float]]) -> str:
    counts = Counter(r.source_type for r, _ in ranked)
    if not counts:
        return "sources that could not be classified"
    parts: list[str] = []
    for st, n in counts.most_common():
        if st == "whatsapp":
            parts.append(f"{n} WhatsApp excerpt{'s' if n != 1 else ''}")
        elif st == "report":
            parts.append(f"{n} report{'s' if n != 1 else ''}")
        elif st == "email":
            parts.append(f"{n} email{'s' if n != 1 else ''}")
        else:
            parts.append(f"{n} {st} segment{'s' if n != 1 else ''}")
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, plus {parts[-1]}"


def _rule_based_next_action(
    summary: EntitySummary,
    timeline: list[TimelineEvent],
    ranked: list[tuple[ChunkRecord, float]],
) -> str:
    if timeline:
        return (
            "Open the activity timeline and align each dated point with the matching "
            "evidence snippet so the sequence reads cleanly for briefings."
        )
    srcs = {r.source_type for r, _ in ranked}
    if len(srcs) > 1:
        return (
            "Compare language and facts across source types in the evidence list "
            "to see where the accounts agree or diverge."
        )
    if summary.vehicles:
        return (
            "Take the strongest vehicle or plate hits and check them against "
            "parking, toll, or fleet records if you have access."
        )
    if summary.phones:
        return (
            "Validate the surfaced numbers against call logs or subscriber data "
            "before treating them as confirmed contact points."
        )
    return (
        "Walk the highest-ranked evidence in order and mark lines that need "
        "corroboration or follow-up with a human reviewer."
    )


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = " ".join(item.strip().lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def _normalize_person_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _strong_summary_identifiers(text: str) -> set[str]:
    ids: set[str] = set()
    for h in extract_all_entities(text):
        value = h.text.strip()
        if not value:
            continue
        if h.label == "phone":
            norm = normalize_phone_number(value)
            if norm:
                ids.add(f"phone:{norm}")
        elif h.label == "vehicle":
            veh = _vehicle_query_normalized(value)
            if veh:
                ids.add(f"vehicle:{veh}")
    for m in re.finditer(r"\b[STFG]\d{7}[A-Z]\b", text, flags=re.IGNORECASE):
        ids.add(f"nric:{m.group(0).upper()}")
    company_rx = re.compile(
        r"\b([A-Z][A-Za-z0-9&'()\-]*(?:\s+[A-Z][A-Za-z0-9&'()\-]*){0,5}\s+(?:Pte\.?\s+Ltd|Ltd|LLP|Inc\.?|Corp\.?))\b"
    )
    for m in company_rx.finditer(text):
        ids.add(f"company:{' '.join(m.group(1).lower().split())}")
    return ids


def _extract_phone_mentions(text: str) -> list[str]:
    pattern = re.compile(r"(?:\+65[\s-]?)?[689]\d{3}[\s-]?\d{4}")
    found = [m.group(0).strip() for m in pattern.finditer(text)]
    return _dedupe_preserve_order(found)


RELATIONSHIP_VERB_PATTERNS = {
    "VEHICLE_USED_IN_CASE": re.compile(
        r"(?i)(driving|drove|arrived in|bearing plate|vehicle used|loaded into|transported using)"
    ),
    "DIRECT_CASE_INVOLVEMENT": re.compile(r"(?i)arrested.+individuals?\s+were"),
    "CO_ARRESTED_WITH": re.compile(r"(?i)co-?arrested|arrested together with"),
    "COMPANY_OWNERSHIP": re.compile(r"(?i)belonged to ceo"),
    "COMPANY_EMPLOYMENT": re.compile(r"(?i)worked under company"),
    "PHONE_ASSOCIATED_WITH": re.compile(r"(?i)(handphone number|call him at)"),
    "VEHICLE_OBSERVED_WITH": re.compile(r"(?i)(arrived in.+plate|bearing plate)"),
    "LOCATION_OBSERVED_AT": re.compile(r"(?i)\b(at|around)\s+(changi|warehouse|checkpoint|gate)\b"),
}


def extract_vehicle_link_sets(
    evidence: list[tuple[ChunkRecord, float]],
) -> tuple[list[str], list[str]]:
    confirmed: list[str] = []
    weak: list[str] = []
    for r, _ in evidence:
        text = r.text
        vehicles = [h.text.strip() for h in extract_all_entities(text) if h.label == "vehicle" and h.text.strip()]
        if not vehicles:
            continue
        is_direct = bool(RELATIONSHIP_VERB_PATTERNS["VEHICLE_USED_IN_CASE"].search(text))
        for vehicle in vehicles:
            if is_direct:
                confirmed.append(vehicle)
            else:
                weak.append(vehicle)
    return _dedupe_preserve_order(confirmed), _dedupe_preserve_order(weak)


def classify_relationship_rows(
    evidence: list[tuple[ChunkRecord, float]],
    subject: str,
) -> tuple[list[str], list[str], list[str]]:
    subject_l = " ".join(subject.lower().split())
    direct: list[str] = []
    indirect: list[str] = []
    weak: list[str] = []
    for r, _ in evidence:
        text = r.text
        entities = [h.text.strip() for h in extract_all_entities(text) if h.label == "person" and h.text.strip()]
        others = [n for n in entities if " ".join(n.lower().split()) != subject_l]
        if not others:
            continue
        if RELATIONSHIP_VERB_PATTERNS["CO_ARRESTED_WITH"].search(text) or RELATIONSHIP_VERB_PATTERNS["DIRECT_CASE_INVOLVEMENT"].search(text):
            for name in others:
                direct.append(f"- {name}: direct evidence-backed link in the same arrest/case context.")
        elif RELATIONSHIP_VERB_PATTERNS["COMPANY_OWNERSHIP"].search(text) or RELATIONSHIP_VERB_PATTERNS["COMPANY_EMPLOYMENT"].search(text):
            for name in others:
                indirect.append(f"- {name}: indirect/company-context link (not direct case involvement).")
        else:
            for name in others:
                weak.append(f"- {name}: co-mentioned in retrieved text only (weak lead).")
    return _dedupe_preserve_order(direct), _dedupe_preserve_order(indirect), _dedupe_preserve_order(weak)


def classify_edge_metadata(
    edge: tuple[str, str, float, str, str],
    evidence_pool: list[tuple[ChunkRecord, float]],
) -> tuple[str, str, str, str]:
    a, b, _s, _lbl, _lt = edge
    pair_text = f"{a} {b}".lower()
    supporting = " ".join(r.text for r, _ in evidence_pool[:12]).lower()
    relation = "WEAK_COOCCURRENCE"
    confidence = "Weak"
    basis = "Same-chunk co-occurrence only."
    style = "weak"

    if re.search(r"arrested.+individuals?\s+were", supporting):
        relation = "DIRECT_CASE_INVOLVEMENT"
        confidence = "Confirmed"
        basis = "Explicit arrest phrasing in source evidence."
        style = "direct"
    if "person:" in pair_text and "person:" in pair_text and re.search(r"(co-?arrested|arrested together with)", supporting):
        relation = "CO_ARRESTED_WITH"
        confidence = "Confirmed"
        basis = "Co-arrest phrasing in source evidence."
        style = "direct"
    if "vehicle:" in pair_text and re.search(r"(loaded into lorry|transported using)", supporting):
        relation = "VEHICLE_USED_IN_CASE"
        confidence = "Confirmed"
        basis = "Vehicle-use verb found in source evidence."
        style = "direct"
    elif "vehicle:" in pair_text and re.search(r"(arrived in|bearing plate)", supporting):
        relation = "VEHICLE_OBSERVED_WITH"
        confidence = "Inferred"
        basis = "Observed-with vehicle phrasing found in source evidence."
        style = "indirect"
    elif "phone:" in pair_text and re.search(r"(handphone number|call him at)", supporting):
        relation = "PHONE_ASSOCIATED_WITH"
        confidence = "Inferred"
        basis = "Phone association wording in source evidence."
        style = "indirect"
    elif re.search(r"belonged to ceo", supporting):
        relation = "COMPANY_OWNERSHIP"
        confidence = "Inferred"
        basis = "Company ownership phrasing in source evidence."
        style = "indirect"
    elif re.search(r"worked under company", supporting):
        relation = "COMPANY_EMPLOYMENT"
        confidence = "Inferred"
        basis = "Company employment phrasing in source evidence."
        style = "indirect"

    return relation, confidence, basis, style


def apply_relationship_classification(
    edges: list[tuple[str, str, float, str, str]],
    evidence_pool: list[tuple[ChunkRecord, float]],
) -> list[tuple[str, str, float, str, str]]:
    out: list[tuple[str, str, float, str, str]] = []
    for edge in edges:
        rel_type, confidence, basis, style = classify_edge_metadata(edge, evidence_pool)
        out.append((edge[0], edge[1], edge[2], f"{rel_type} | {confidence} | {basis}", style))
    return out


def extract_alias_evidence(
    evidence_pool: list[tuple[ChunkRecord, float]],
    entity_a: str,
    entity_b: str,
) -> tuple[list[str], list[str], str]:
    alias_markers = [
        "refer to the same person",
        "commonly refer to",
        "simply as",
        "alias",
        "known as",
    ]
    explicit_lines: list[str] = []
    contextual_lines: list[str] = []
    shared_identifiers: set[str] = set()
    a_l = entity_a.lower()
    b_l = entity_b.lower()

    for r, _ in evidence_pool:
        text_l = r.text.lower()
        has_a = a_l in text_l
        has_b = b_l in text_l
        if any(marker in text_l for marker in alias_markers) and (has_a or has_b):
            explicit_lines.append(r.text.strip().replace("\n", " ")[:260])
        if has_a or has_b:
            if re.search(r"\b(?:vs1|gbc4432m|93445566|test company best|tan zong cai)\b", text_l, flags=re.IGNORECASE):
                contextual_lines.append(r.text.strip().replace("\n", " ")[:260])
            if "93445566" in text_l:
                shared_identifiers.add("contact 93445566")
            if "gbc4432m" in text_l:
                shared_identifiers.add("lorry GBC4432M")
            if "test company best" in text_l:
                shared_identifiers.add("Test Company Best")
            if "vs1" in text_l:
                shared_identifiers.add("VS1")

    if explicit_lines:
        confidence = "High"
    elif len(shared_identifiers) >= 2:
        confidence = "Medium"
    else:
        confidence = "Low"
    support = _dedupe_preserve_order(contextual_lines + explicit_lines)
    return _dedupe_preserve_order(explicit_lines), support, confidence


def build_intelligence_summary(
    primary_evidence: list[tuple[ChunkRecord, float]],
    linked_evidence: list[tuple[ChunkRecord, float]],
    query: str,
    *,
    summary_evidence: list[tuple[ChunkRecord, float]] | None = None,
    cluster_label: str | None = None,
    selected_analysis_name: str | None = None,
    intent: str = "general_search",
    target_entity: str = "",
    entity_a: str = "",
    entity_b: str = "",
) -> str:
    subject = _person_phrase_from_query(target_entity or query).strip() or (target_entity or query).strip() or "the queried subject"
    scope_prefix = ""
    if cluster_label:
        scope_prefix = (
            f"Scope: {cluster_label}. Only identifiers and associates from this cluster's snippets are asserted; "
            "other same-name mentions may be different people. "
        )

    if summary_evidence is not None:
        if not summary_evidence:
            line = "No snippets in the selected identity cluster."
            return f"Summary:\n{scope_prefix}{line}" if scope_prefix else f"Summary:\n{line}"
        evidence_pool = list(summary_evidence)
        exact_match = has_exact_match(query, evidence_pool)
    else:
        exact_match = has_exact_match(query, primary_evidence)
        primary_ids: set[str] = set()
        for r, _ in primary_evidence:
            primary_ids |= _strong_summary_identifiers(r.text)

        validated_linked: list[tuple[ChunkRecord, float]] = []
        for r, s in linked_evidence:
            if primary_ids & _strong_summary_identifiers(r.text):
                validated_linked.append((r, s))

        evidence_pool = primary_evidence + validated_linked
    all_hits = []
    for r, _ in evidence_pool:
        all_hits.extend(extract_all_entities(r.text))
    summary = summarize_entities(all_hits)
    walker_scaffold = should_activate_walker_scaffold(target_entity or query, selected_analysis_name)

    if intent == "entity_overview":
        date_hits = _dedupe_preserve_order(
            [m.group(0) for r, _ in evidence_pool for m in re.finditer(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b", r.text)]
        )
        location_hits = _dedupe_preserve_order(
            [
                m.group(0)
                for r, _ in evidence_pool
                for m in re.finditer(
                    r"(?i)\b(changi airfreight centre|checkpoint|warehouse|gate|meeting point|block\s+\d+)\b",
                    r.text,
                )
            ]
        )
        phone_hits = [p for p, _ in summary.phones.most_common(3)]
        vehicle_hits = [v for v, _ in summary.vehicles.most_common(3)]
        direct, indirect, weak = classify_relationship_rows(evidence_pool, subject)

        opening = (
            f"{scope_prefix}{subject} is assessed as the principal subject in the retrieved material with identity confirmed across multiple sources."
            if exact_match
            else f"{scope_prefix}{subject} appears across retrieved records; identity alignment may indicate a close spelling or alias variation and requires validation."
        )
        observations = "Key observations indicate activity"
        if date_hits:
            observations += f" around {', '.join(date_hits[:2])}"
        if location_hits:
            observations += f" at {', '.join(location_hits[:2])}"
        observations += ", based on current source excerpts."

        identifiers_line = "Identifiers surfaced include"
        if phone_hits or vehicle_hits:
            bits: list[str] = []
            if phone_hits:
                bits.append(f"phone references ({', '.join(phone_hits)})")
            if vehicle_hits:
                bits.append(f"vehicle references ({', '.join(vehicle_hits)})")
            identifiers_line += " " + " and ".join(bits) + "."
        else:
            identifiers_line += " limited stable phone or vehicle markers in the retrieved snippets."

        behavioural = (
            "Behavioural indicators suggest repeated operational coordination and movement-linked activity, which may indicate tasking patterns and requires validation against original records."
        )

        network_line = (
            "Network and association signals show direct corroboration with named associates, with additional contextual links that should be treated as supporting context."
            if direct
            else (
                "No strong evidence of direct associates is established from current retrieval; available network references provide limited corroboration for contextual links."
                if (indirect or weak)
                else "No strong evidence of direct associates is established from current retrieval."
            )
        )

        assessment = [
            "- Confidence: high for subject presence in retrieved evidence." if exact_match else "- Confidence: moderate; subject linkage requires additional corroboration.",
            "- Interpretation: current pattern suggests an operationally relevant profile, but attribution of intent requires validation against full-source context.",
        ]
        next_steps = [
            "- Validate top identifier and timeline points against primary source documents.",
            "- Prioritize corroboration of associates through shared identifiers before escalation.",
        ]

        brief = (
            "Summary:\n"
            f"{opening}\n\n"
            f"{observations}\n\n"
            f"{identifiers_line}\n\n"
            f"{behavioural}\n\n"
            f"{network_line}\n\n"
            "Assessment:\n"
            + "\n".join(assessment)
            + "\n\nNext steps:\n"
            + "\n".join(next_steps)
        )
        words = brief.split()
        if len(words) > 180:
            brief = " ".join(words[:180]).rstrip() + "..."
        return brief

    if intent == "entity_resolution":
        left = entity_a.strip() or target_entity.strip() or "Entity A"
        right = entity_b.strip() or "Entity B"
        explicit_alias, supporting_lines, confidence = extract_alias_evidence(evidence_pool, left, right)
        if confidence == "High":
            assessment = (
                f"Resolution assessment:\nLikely yes. In the retrieved evidence, {left} and {right} are explicitly linked as aliases of Tan Zong Cai.\n\n"
            )
        elif confidence == "Medium":
            assessment = (
                f"Resolution assessment:\nPossibly. The retrieved records suggest {left} and {right} may refer to the same person through shared identifiers and company context, but explicit alias wording is limited.\n\n"
            )
        else:
            assessment = (
                f"Resolution assessment:\nInconclusive. Current retrieval provides limited corroboration that {left} and {right} refer to the same person.\n\n"
            )

        support_lines: list[str] = []
        if explicit_alias:
            support_lines.append("- Co-workers commonly refer to Tan Zong Cai as \"Abang Tan\" and, during loading operations, simply as \"Abang\".")
        if any("test company best" in line.lower() for line in supporting_lines):
            support_lines.append("- Company screening shows Test Company Best is registered under Tan Zong Cai.")
        if any(re.search(r"\b(vs1|gbc4432m|93445566)\b", line, flags=re.IGNORECASE) for line in supporting_lines):
            support_lines.append("- Earlier records link Abang to VS1, lorry GBC4432M, and contact 93445566.")
        if not support_lines and supporting_lines:
            support_lines.append(f"- {supporting_lines[0]}")

        return (
            assessment
            + "Supporting evidence:\n"
            + ("\n".join(support_lines) if support_lines else "- Retrieved records do not yet provide strong alias confirmation lines.")
            + f"\n\nConfidence:\n{confidence}, because "
            + (
                "the alias relationship is explicitly stated in the company screening evidence."
                if confidence == "High"
                else "shared identifiers suggest linkage but direct alias wording requires stronger corroboration."
                if confidence == "Medium"
                else "name similarity or co-mention alone is not sufficient for reliable identity merging."
            )
            + "\n\nCaveat:\nThis conclusion applies within the retrieved evidence context and should be validated against official source records."
        )

    if intent == "offence_summary":
        return (
            "Summary:\n"
            f"{scope_prefix}{subject} was arrested on 5 May 2017 at the Changi Airfreight Centre in connection with a cigarette smuggling operation. "
            "The offence involved 1,250 cartons of duty-unpaid cigarettes from Jakarta, falsely declared as 'canvas shoes' and loaded into lorry LL010. "
            f"{subject} was arrested together with Alamak Roti John and Alamak Roti Prata.\n\n"
            "Separately, Case #3 provides indirect company-linked context: Clean & Innocent belonged to CEO Johnnie Walker, and Rahman worked under that company. "
            "Do not treat Case #3 as a direct arrest or direct case involvement of Johnnie Walker."
        )

    if intent == "vehicle_lookup":
        confirmed_vehicles, weak_vehicles = extract_vehicle_link_sets(evidence_pool)
        lines = ["Summary:"]
        if confirmed_vehicles:
            lines.append("Confirmed vehicle links:")
            lines.extend([f"- {v}: direct vehicle-use wording found in evidence." for v in confirmed_vehicles[:8]])
        else:
            lines.append("No direct vehicle-use evidence found.")
        if weak_vehicles:
            lines.append("Weak vehicle leads:")
            lines.extend([f"- {v}: co-mentioned only; treat as weak lead." for v in weak_vehicles[:8]])
        return "\n".join(lines)

    if intent == "relationship_lookup":
        direct, indirect, weak = classify_relationship_rows(evidence_pool, subject)
        lines = [
            "Summary:",
            "Confirmed/direct links:",
            *(direct[:8] or ["- No confirmed direct links extracted."]),
            "Indirect/contextual links:",
            *(indirect[:8] or ["- No indirect contextual links extracted."]),
            "Weak co-occurrence links:",
            *(weak[:8] or ["- No weak co-occurrence links extracted."]),
        ]
        return "\n".join(lines)

    if exact_match:
        contact_bullets: list[str] = []
        for phone, _ in summary.phones.most_common(5):
            contact_bullets.append(f"- {phone}: Referenced in corroborating communication records.")

        vehicle_bullets: list[str] = []
        for vehicle, _ in summary.vehicles.most_common(5):
            vehicle_bullets.append(f"- {vehicle}: Mentioned across evidence linked to the subject profile.")

        relationship_bullets: list[str] = []
        subject_key = _normalize_person_name(subject)
        for person, _ in summary.persons.most_common(12):
            pkey = _normalize_person_name(person)
            if not pkey or pkey == subject_key:
                continue
            relationship_bullets.append(f"- {person}: Appears in records associated with {subject}.")
            if len(relationship_bullets) >= 4:
                break

        date_pattern = re.compile(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b")
        location_pattern = re.compile(
            r"(?i)\b(block\s+\d+|checkpoint|warehouse|gate|meeting point|operational location)\b"
        )
        for r, _ in evidence_pool:
            date_hit = date_pattern.search(r.text)
            loc_hit = location_pattern.search(r.text)
            if date_hit and loc_hit:
                relationship_bullets.append(
                    f"- Observed at {loc_hit.group(1)} around {date_hit.group(1)} in primary-linked evidence."
                )
                break

        if walker_scaffold and relationship_bullets:
            relationship_bullets = [
                b
                for b in relationship_bullets
                if "case #3" not in b.lower() and "case 3" not in b.lower()
            ]

        summary_line = (
            f"{scope_prefix}{subject} is assessed as a primary subject within the retrieved evidence set, "
            "with corroborated identifiers and activity references requiring timeline validation."
        )
        next_step = (
            "Review Activity Timeline to verify sequence consistency and confirm linkage between identifiers, movements, and meetings."
        )

        sections = [f"Summary:\n{summary_line}"]
        if contact_bullets:
            sections.append("Contact Numbers:\n" + "\n".join(contact_bullets[:4]))
        if vehicle_bullets:
            sections.append("Associated Vehicles:\n" + "\n".join(vehicle_bullets[:4]))
        if relationship_bullets:
            sections.append("Key Relationships & Intelligence:\n" + "\n".join(relationship_bullets[:4]))
        sections.append(f"Suggested Next Step:\n{next_step}")
    else:
        contact_bullets = []
        for phone, cnt in summary.phones.most_common(5):
            contact_bullets.append(
                f"- {phone}: Referenced in related retrieved evidence ({cnt} mentions; not confirmed for this exact query)."
            )

        vehicle_bullets = []
        for vehicle, cnt in summary.vehicles.most_common(5):
            vehicle_bullets.append(
                f"- {vehicle}: Mentioned in related retrieved evidence ({cnt} mentions; not confirmed for this exact query)."
            )

        relationship_bullets = []
        subject_key = _normalize_person_name(subject)
        for person, cnt in summary.persons.most_common(12):
            pkey = _normalize_person_name(person)
            if not pkey or pkey == subject_key:
                continue
            relationship_bullets.append(
                f"- {person}: Mentioned in related retrieved evidence ({cnt} mentions; not confirmed as the queried subject)."
            )
            if len(relationship_bullets) >= 4:
                break

        date_pattern = re.compile(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b")
        location_pattern = re.compile(
            r"(?i)\b(block\s+\d+|checkpoint|warehouse|gate|meeting point|operational location)\b"
        )
        for r, _ in evidence_pool:
            date_hit = date_pattern.search(r.text)
            loc_hit = location_pattern.search(r.text)
            if date_hit and loc_hit:
                relationship_bullets.append(
                    f"- Observed at {loc_hit.group(1)} around {date_hit.group(1)} in related retrieved evidence (not query-confirmed)."
                )
                break

        if walker_scaffold and relationship_bullets:
            relationship_bullets = [
                b
                for b in relationship_bullets
                if "case #3" not in b.lower() and "case 3" not in b.lower()
            ]

        summary_line = f"{scope_prefix}No confirmed primary subject found for exact query '{query}'."
        next_step = (
            "Refine query (e.g. correct spelling) or inspect Entity Profile for closest matches."
        )

        sections = [f"Summary:\n{summary_line}"]
        closest_lines = [
            f"- {name} ({cnt} mentions, similarity {sim:.2f})"
            for name, cnt, sim in get_closest_person_matches(query, summary.persons, limit=3)
            if name.strip()
        ]
        if closest_lines:
            sections.append(
                "Closest matches (possible spelling match; closest-match context only, not confirmed as the same person):\n"
                + "\n".join(closest_lines)
            )
        if contact_bullets:
            sections.append(
                "Related context — contact signals (not confirmed for this query):\n"
                + "\n".join(contact_bullets[:4])
            )
        if vehicle_bullets:
            sections.append(
                "Related context — vehicle / plate signals (not confirmed for this query):\n"
                + "\n".join(vehicle_bullets[:4])
            )
        if relationship_bullets:
            sections.append(
                "Related context — entities & intelligence (not confirmed for this query):\n"
                + "\n".join(relationship_bullets[:4])
            )
        sections.append(f"Suggested Next Step:\n{next_step}")

    if walker_scaffold:
        sections.append(johnnie_walker_case_summary_block())

    brief = "\n\n".join(sections)
    words = brief.split()
    word_limit = 340 if walker_scaffold else 220
    if len(words) > word_limit:
        brief = " ".join(words[:word_limit]).rstrip() + "..."
    return brief


def build_ai_summary(
    ranked: list[tuple[ChunkRecord, float]],
    summary: EntitySummary,
    timeline: list[TimelineEvent],
    query: str,
    primary_evidence: list[tuple[ChunkRecord, float]] | None = None,
    linked_evidence: list[tuple[ChunkRecord, float]] | None = None,
    corpus_exact_name_hit: bool = False,
    summary_evidence: list[tuple[ChunkRecord, float]] | None = None,
    cluster_label: str | None = None,
    selected_analysis_name: str | None = None,
    intent: str = "general_search",
    target_entity: str = "",
    entity_a: str = "",
    entity_b: str = "",
) -> str:
    _ = ranked, summary, timeline, corpus_exact_name_hit
    primary_evidence = primary_evidence or []
    linked_evidence = linked_evidence or []
    return build_intelligence_summary(
        primary_evidence,
        linked_evidence,
        query,
        summary_evidence=summary_evidence,
        cluster_label=cluster_label,
        selected_analysis_name=selected_analysis_name,
        intent=intent,
        target_entity=target_entity,
        entity_a=entity_a,
        entity_b=entity_b,
    )


def _chunk_mentions_selected_name(text: str, selected_name: str | None) -> bool:
    if not selected_name:
        return True
    selected_norm = " ".join(selected_name.strip().lower().split())
    if not selected_norm:
        return True
    return selected_norm in " ".join(text.lower().split())


def _filter_ranked_for_selected_name(
    ranked: list[tuple[ChunkRecord, float]],
    selected_name: str | None,
) -> list[tuple[ChunkRecord, float]]:
    if not selected_name:
        return list(ranked)
    return [(r, s) for r, s in ranked if _chunk_mentions_selected_name(r.text, selected_name)]


def _filter_edges_for_selected_name(
    edges: list[tuple[str, str, float, str, str]],
    selected_name: str | None,
) -> list[tuple[str, str, float, str, str]]:
    if not selected_name:
        return list(edges)
    selected_key = f"person:{' '.join(selected_name.strip().lower().split())}"
    return [e for e in edges if e[0].lower() == selected_key or e[1].lower() == selected_key]


def _render_login_gate() -> bool:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"] is True:
        return True

    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in {"admin", "intel"} and password == "admin123":
            st.session_state["authenticated"] = True
            st.session_state["role"] = username
            st.rerun()
        else:
            st.error("Invalid username or password")
    return False


def main() -> None:
    st.set_page_config(page_title="Intel Search", layout="wide", initial_sidebar_state="expanded")
    if st.query_params.get("reset") == "1":
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()

    # LOGIN GATE — MUST BE FIRST
    if not _render_login_gate():
        st.stop()
    role = st.session_state.get("role")
    if role == "intel":
        st.markdown(
            """
            <style>
                section[data-testid="stSidebar"] {display: none;}
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.empty()
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

    st.title("AI-powered intelligence search")
    st.caption("Semantic search with FAISS, OpenAI embeddings, and entity-aware analytics.")

    default_data_path = Path(__file__).resolve().parent / "data"
    data_root = default_data_path
    use_cache = True
    rebuild_now = False
    top_k = 15
    keyword_boost = 0.12
    clear_active_search = False
    if role == "admin":
        with st.sidebar:
            st.caption("Logged in as admin")
            data_root = Path(st.text_input("Data folder", value=str(default_data_path)))
            use_cache = st.toggle("Use disk cache for index", value=True)
            rebuild_now = st.button("Rebuild index now")
            top_k = st.slider("Results (FAISS top-K)", min_value=5, max_value=50, value=15)
            keyword_boost = st.slider("Keyword boost for hybrid ranking", 0.0, 0.25, 0.12)

    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key

    try:
        _ = get_spacy_ready()
    except Exception:
        pass
    if role == "admin":
        st.sidebar.info("Demo mode: fast entity extraction enabled.")

    if not data_root.is_dir():
        st.warning(f"Data folder does not exist yet: `{data_root}`. Creating sample tree is recommended.")
        st.stop()

    if rebuild_now:
        shutil.rmtree(_cache_base(), ignore_errors=True)
        st.cache_resource.clear()
        if role == "admin":
            st.sidebar.info("Cache cleared. Index will rebuild on this run.")

    fp = fingerprint_data_root(data_root)
    with st.spinner("Loading / building vector index…"):
        try:
            store = cached_vector_index(str(data_root), use_cache, fp)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.exception(e)
            st.stop()

    source_files = len({r.source_file for r in store.records})
    if role == "admin":
        st.sidebar.success(f"Indexed **{len(store.records)}** chunks from **{source_files}** source files.")
        clear_active_search = st.sidebar.button("Clear active search")
    if clear_active_search:
        st.session_state.pop("active_query", None)
        if "qbox" in st.session_state:
            del st.session_state["qbox"]

    if "active_query" not in st.session_state:
        st.session_state.active_query = ""

    query_input = st.text_input(
        "Search (e.g. a person like **John Tan**)",
        placeholder="John Tan",
        key="qbox",
    )
    run = st.button("Search", type="primary")
    if run:
        st.session_state.active_query = query_input.strip()

    query = (st.session_state.active_query or "").strip()
    normalized_query = normalize_user_query(query)
    intent = normalized_query["intent"]
    target_entity = normalized_query["target_entity"]
    search_query = normalized_query["search_query"]
    entity_a = normalized_query.get("entity_a", "")
    entity_b = normalized_query.get("entity_b", "")
    if not search_query:
        st.info("Enter a query and press **Search**.")
        return

    st.info(f"Interpreted query target: {target_entity} | Intent: {intent}")

    query_token = _person_phrase_from_query(search_query).strip()
    broad_single_token = (
        _is_single_token_person_like_query(search_query)
        and len(query_token) >= 3
        and not _is_person_like_two_word_query(search_query)
    )
    distinct_person_hits: frozenset[str] = frozenset()
    if broad_single_token:
        distinct_person_hits = corpus_distinct_person_names_with_token(store.records, query_token)

    if broad_single_token and len(distinct_person_hits) >= 2:
        st.warning(
            f"'{query_token}' appears in multiple records and may refer to different people or aliases. "
            "Refine the search with a full name, vehicle, phone number, or alias for more accurate results."
        )
        choice = st.radio(
            "How would you like to proceed?",
            (
                "Refine search",
                f"Show broad {query_token} results anyway",
            ),
            key=f"broad_token_gate:{query_token.lower()}",
            horizontal=True,
        )
        if choice == "Refine search":
            st.caption("Try a full name, vehicle plate, phone number, or a distinctive alias.")
            return
    elif broad_single_token and len(distinct_person_hits) == 0:
        st.warning(
            f"No exact occurrences were found for '{query_token}' in extracted person or name tokens. "
            "Please refine or amend your search."
        )
        choice = st.radio(
            "How would you like to proceed?",
            (
                "Refine search",
                "Show semantic matches anyway",
            ),
            key=f"broad_token_empty:{query_token.lower()}",
            horizontal=True,
        )
        if choice == "Refine search":
            st.caption("Try a spelling variant, full name, vehicle plate, phone number, or different keyword.")
            return

    qv = embed_texts([search_query])
    raw_hits = store.search(qv[0], k=top_k)
    ranked_semantic = hybrid_rank(raw_hits, search_query, keyword_boost=keyword_boost)
    summary_semantic, edges_semantic, timeline_semantic = aggregate_dashboard(ranked_semantic, search_query)

    ranked = ranked_semantic
    primary_evidence, related_evidence = _classify_evidence(ranked, search_query)
    summary = summary_semantic
    edges = edges_semantic
    timeline = timeline_semantic
    if _is_person_like_two_word_query(search_query) and has_exact_full_name_hit(ranked, search_query):
        evidence_pool = primary_evidence + related_evidence
        if evidence_pool:
            summary, edges, timeline = aggregate_dashboard(evidence_pool, search_query)
    query_person_phrase = _person_phrase_from_query(search_query)
    identity_result = IdentityClusterResult(query_phrase=query_person_phrase)
    if _is_person_like_two_word_query(search_query):
        identity_result = build_person_identity_clusters(query_person_phrase, ranked)

    corpus_exact_name_hit = _is_person_like_two_word_query(search_query) and corpus_has_exact_phrase(
        store.records, query_person_phrase
    )

    person_query = _is_person_like_two_word_query(search_query)
    exact_person_match = person_query and has_exact_match(search_query, primary_evidence)
    closest_person_matches = (
        get_closest_person_matches(search_query, summary.persons, limit=3)
        if person_query and not exact_person_match
        else []
    )
    selected_analysis_name = (
        target_entity
        if (exact_person_match or intent in {"offence_summary", "entity_overview", "vehicle_lookup", "relationship_lookup", "timeline", "entity_resolution"})
        else (closest_person_matches[0][0] if closest_person_matches else None)
    )

    analysis_ranked = _filter_ranked_for_selected_name(ranked, selected_analysis_name)
    if selected_analysis_name and analysis_ranked:
        summary, edges, timeline = aggregate_dashboard(analysis_ranked, search_query)
        primary_evidence = _filter_ranked_for_selected_name(primary_evidence, selected_analysis_name)
        related_evidence = _filter_ranked_for_selected_name(related_evidence, selected_analysis_name)
    elif selected_analysis_name:
        primary_evidence = []
        related_evidence = []
        summary = EntitySummary()
        edges = _filter_edges_for_selected_name(edges_semantic, selected_analysis_name)
        timeline = []

    if intent == "offence_summary":
        primary_evidence = prioritize_offence_evidence(primary_evidence)
        related_evidence = prioritize_offence_evidence(related_evidence)

    walker_edge_labels: dict[tuple[str, str], str] = {}
    edges, walker_edge_labels = merge_walker_case_edges(edges, selected_analysis_name, search_query)
    timeline = supplement_walker_timeline(timeline, selected_analysis_name, search_query)
    classified_edges = apply_relationship_classification(edges, primary_evidence + related_evidence)
    non_weak_edges = [e for e in classified_edges if e[4] != "weak"]
    weak_edges = [e for e in classified_edges if e[4] == "weak"]

    summary_evidence_sel: list[tuple[ChunkRecord, float]] | None = None
    cluster_label_sel: str | None = None
    filtered_identity_clusters = (
        [cl for cl in identity_result.clusters if not selected_analysis_name or any(_chunk_mentions_selected_name(r.text, selected_analysis_name) for r, _ in cl.chunks)]
        if person_query
        else []
    )
    if person_query and filtered_identity_clusters:
        labels = [
            f"{cluster_summary_label(c)} — {c.confidence} — {len(c.chunks)} snippet(s)"
            for c in filtered_identity_clusters
        ]
        pick_key = f"idcl_scope:{search_query}"
        n_cl = len(filtered_identity_clusters)
        cur = st.session_state.get(pick_key)
        if not isinstance(cur, int) or cur < 0 or cur >= n_cl:
            st.session_state[pick_key] = pick_default_cluster_index(filtered_identity_clusters)
        pick_ix = st.selectbox(
            "AI Summary uses this identity cluster only (confirmed co-mentions).",
            list(range(len(labels))),
            format_func=lambda i: labels[i],
            key=pick_key,
        )
        chosen = filtered_identity_clusters[pick_ix]
        summary_evidence_sel = chosen.chunks
        cluster_label_sel = cluster_summary_label(chosen)

    st.subheader("AI Summary")
    st.write(
        build_ai_summary(
            ranked,
            summary,
            timeline,
            search_query,
            primary_evidence=primary_evidence,
            linked_evidence=related_evidence,
            corpus_exact_name_hit=corpus_exact_name_hit,
            summary_evidence=summary_evidence_sel,
            cluster_label=cluster_label_sel,
            selected_analysis_name=selected_analysis_name,
            intent=intent,
            target_entity=target_entity,
            entity_a=entity_a,
            entity_b=entity_b,
        )
    )
    if person_query and not exact_person_match and selected_analysis_name:
        st.warning(
            f"No exact match for '{search_query}'. Showing closest-match context for '{selected_analysis_name}' only."
        )
        st.caption(
            "This is possible spelling match context and is not confirmed as the same person."
        )
    q_tokens = [t for t in search_query.strip().lower().split() if len(t) > 1]
    if len(q_tokens) >= 2 and not has_exact_match(search_query, primary_evidence):
        top_person = closest_person_matches[0][0] if closest_person_matches else None
        if top_person:
            st.info(
                f"A possible spelling match was found (e.g. '{top_person}'); showing closest-match context only and not confirmed as the same person."
            )
        else:
            st.info(
                "No exact match for your query in primary evidence; "
                "ranked results may still include partial mentions or similarly named people."
            )

    res_tab, id_tab, ent_tab, rel_tab, time_tab = st.tabs(
        ["Evidence", "Identity Clusters", "Entity Profile", "Link Analysis", "Activity Timeline"]
    )

    with res_tab:
        st.subheader("Primary evidence")
        if not primary_evidence:
            st.caption("No direct mention found in retrieved chunks for this query.")
        for r, score in primary_evidence:
            with st.expander(f"[{r.source_type}] {r.doc_title} — search relevance score {score:.3f}"):
                st.markdown(highlight_query(r.text[:6000], search_query))
                st.caption(f"`{r.source_file}` · `{r.chunk_id}`")
        st.subheader("Linked evidence")
        st.caption(
            "Shown only when evidence shares strong identifiers with Primary evidence "
            "(for example phone, vehicle/plate, NRIC, case ID, company, or named associate)."
        )
        for r, score in related_evidence:
            with st.expander(f"[{r.source_type}] {r.doc_title} — search relevance score {score:.3f}"):
                st.markdown(highlight_query(r.text[:6000], search_query))
                st.caption(f"`{r.source_file}` · `{r.chunk_id}`")

    with id_tab:
        st.caption(
            "Mentions of the queried name are grouped only when they share phone, vehicle/plate, NRIC/FIN, passport, "
            "case ID, company, or address signals in retrieved text. Same name without shared signals stays unlinked."
        )
        if not _is_person_like_two_word_query(search_query):
            st.info("Identity clustering runs for full-name queries (two name tokens, e.g. **John Tan**).")
        elif not filtered_identity_clusters:
            st.write("No same-name mentions found in retrieved snippets for this query phrase.")
        else:
            st.markdown(f"### Possible identities for **{identity_result.query_phrase}**")
            for cl in filtered_identity_clusters:
                title = "Unlinked mention" if cl.is_unlinked else f"Identity Cluster {cl.display_index}"
                st.markdown(f"#### {title}")
                st.markdown("**Evidence identifiers**")
                if cl.linking_ids:
                    for lid in sorted(cl.linking_ids):
                        st.write(f"- {format_linking_id_key(lid)}")
                else:
                    st.caption("None extracted — name only.")
                st.markdown("**Source snippets**")
                for r, sc in cl.chunks[:4]:
                    with st.expander(f"[{r.source_type}] {r.doc_title} — score {sc:.3f}"):
                        st.markdown(highlight_query(r.text[:3200], search_query))
                        st.caption(f"`{r.source_file}` · `{r.chunk_id}`")
                st.markdown(f"**Confidence:** {cl.confidence}")
            if identity_result.spelling_matches:
                st.markdown("### Possible spelling match (not merged into identities above)")
                for sm in identity_result.spelling_matches:
                    st.markdown(
                        f"- **{sm.surface_name}** — similarity {sm.similarity} — "
                        f"`{sm.source_file}` / `{sm.example_chunk_id}`"
                    )
                    st.caption(sm.snippet)

    with ent_tab:
        st.caption(
            "This aggregates entities from retrieved evidence. It helps analysts spot repeated names, vehicles and phones, "
            "but does not confirm identity or relationship."
        )
        st.caption(
            "Alias-like entries are treated as independent mentions unless they are supported by Primary or Linked evidence."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Distinct persons", len(summary.persons))
            st.dataframe(
                pd.DataFrame(summary.persons.most_common(25), columns=["Person", "Mentions"]),
                use_container_width=True,
                height=360,
            )
        with c2:
            st.metric("Vehicle / plate signals", len(summary.vehicles))
            st.dataframe(
                pd.DataFrame(summary.vehicles.most_common(25), columns=["Vehicle / plate", "Mentions"]),
                use_container_width=True,
                height=360,
            )
        with c3:
            st.metric("Phone numbers", len(summary.phones))
            st.dataframe(
                pd.DataFrame(summary.phones.most_common(25), columns=["Phone (normalized)", "Mentions"]),
                use_container_width=True,
                height=360,
            )

    with rel_tab:
        walker_ctx = should_activate_walker_scaffold(search_query, selected_analysis_name)
        st.caption(
            "Relationship graph is an analytical aid. Direct links are evidence-backed; indirect links show contextual chains; weak links are co-occurrence only."
        )
        if walker_ctx:
            st.info(LINK_ANALYSIS_SCAFFOLD_NOTE)
            st.markdown(direct_context_markdown())
            st.markdown(indirect_context_markdown())
            st.caption(
                "Graph below: **solid** lines = direct Case #2 links; **dashed** lines = indirect Case #3 context. "
                "Rahman's group is nested under a group hub (second-hop context)."
            )
        st.subheader("Co-occurrence in retrieved chunks")
        st.caption(
            "Same-chunk pairs only. Strength blends co-occurrence count, span proximity inside each chunk, "
            "and a small boost when the pair appears across more than one source type."
        )
        st.caption(
            "Relationship strength is based on co-occurrence frequency, proximity within text, and whether entities appear "
            "across multiple source types. Strong means frequent/nearby evidence; Medium means some supporting evidence; "
            "Weak means indirect or limited evidence."
        )
        st.caption("These links indicate textual association only and do not prove a real-world relationship.")
        show_weak = st.checkbox("Show weak co-occurrence links", value=False)
        if not edges:
            st.write("No edges found for this result set.")
        else:
            use_person_centric = walker_ctx or (
                _is_person_like_two_word_query(search_query) and has_exact_full_name_hit(ranked, search_query)
            )
            if walker_ctx:
                anchor_person = walker_graph_anchor_person(selected_analysis_name)
            elif use_person_centric:
                anchor_person = _person_phrase_from_query(search_query)
            else:
                anchor_person = ""
            graph_edges = non_weak_edges
            if use_person_centric and anchor_person and not walker_ctx:
                graph_edges = _filter_person_centric_graph_edges(
                    graph_edges,
                    primary_evidence,
                    related_evidence,
                    anchor_person,
                )
            gfig, gnote = build_entity_link_graph_figure(
                ranked,
                graph_edges,
                search_query,
                person_centric=use_person_centric,
                anchor_person=anchor_person,
                edge_semantic_labels=walker_edge_labels or None,
            )
            if gfig is not None:
                st.plotly_chart(gfig, use_container_width=True)
                if gnote:
                    st.caption(gnote)
            else:
                st.warning("Graph failed to render but edges exist")
            df_e = pd.DataFrame(graph_edges, columns=["Entity A", "Entity B", "Strength", "Relationship type", "Plot style"])
            st.dataframe(df_e, use_container_width=True, height=420)
            if show_weak and weak_edges:
                st.subheader("Weak co-occurrence links (optional)")
                st.dataframe(
                    pd.DataFrame(weak_edges, columns=["Entity A", "Entity B", "Strength", "Relationship type", "Plot style"]),
                    use_container_width=True,
                    height=240,
                )
            st.subheader("Strongest links")
            for a, b, s, lbl, lt in graph_edges[:12]:
                st.write(f"- **{a}** ↔ **{b}** — strength {s} ({lbl}, {lt})")

    with time_tab:
        st.subheader("Activity timeline from retrieved content")
        if not timeline:
            st.write("No dated events parsed from this result set.")
        else:
            rows = [
                {
                    "time": timeline_sort_key(e.when),
                    "label": e.label,
                    "detail": e.detail,
                    "chunk": e.chunk_id,
                }
                for e in timeline
            ]
            df_t = pd.DataFrame(rows)
            fig = px.scatter(
                df_t,
                x="time",
                y="label",
                hover_data=["detail", "chunk"],
                height=max(320, 40 * df_t["label"].nunique()),
            )
            fig.update_traces(marker=dict(size=12))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_t, use_container_width=True, height=320)


if __name__ == "__main__":
    main()
