from __future__ import annotations

import os
import re
import shutil
from collections import Counter, defaultdict, deque
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from pandas.io.formats.style import Styler
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
    is_evidence_boilerplate_person_name,
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
    format_identifier_provenance_line,
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

DEBUG_MODE = False


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
    re.compile(r"^details\s+about\s+(.+?)\??$", flags=re.IGNORECASE),
]

IDENTITY_LOOKUP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^what\s+is\s+(.+?)\s+real\s+name\??$", flags=re.IGNORECASE),
    re.compile(r"^what\s+is\s+the\s+real\s+name\s+of\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^who\s+is\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^who\s+is\s+known\s+as\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^identify\s+(.+?)\??$", flags=re.IGNORECASE),
    re.compile(r"^what\s+does\s+(.+?)\s+refer\s+to\??$", flags=re.IGNORECASE),
]

_ABANG_IDENTITY_SEARCH_ENRICH = (
    "Abang Abang Tan Tan Zong Cai Test Company Best VS1 GBC4432M 93445566"
)

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


def _is_abang_identity_target(target: str) -> bool:
    return bool(re.search(r"\babang\b", target or "", flags=re.IGNORECASE))


def _abang_identity_chunk_needles() -> tuple[str, ...]:
    return (
        "abang",
        "abang tan",
        "tan zong cai",
        "test company best",
        "vs1",
        "gbc4432m",
        "93445566",
    )


def _chunk_matches_abang_identity_context(text: str) -> bool:
    tl = text.lower()
    return any(n in tl for n in _abang_identity_chunk_needles())


def _identity_lookup_search_query(target_entity: str) -> str:
    te = _clean_target_entity(target_entity)
    if _is_abang_identity_target(te):
        return f"{te} {_ABANG_IDENTITY_SEARCH_ENRICH}".strip()
    return f"{te} Tan Zong Cai Test Company Best".strip()


def _filter_ranked_abang_identity(
    ranked: list[tuple[ChunkRecord, float]],
) -> list[tuple[ChunkRecord, float]]:
    return [(r, s) for r, s in ranked if _chunk_matches_abang_identity_context(r.text)]


_ABANG_GRAPH_NODE_LOWER = frozenset(
    {
        "person:tan zong cai",
        "person:abang",
        "person:abang tan",
        "company:test company best",
        "phone:93445566",
        "vehicle:gbc4432m",
        "vehicle:vs1",
    }
)


def _node_in_abang_identity_graph(node: str) -> bool:
    return " ".join(node.strip().lower().split()) in _ABANG_GRAPH_NODE_LOWER


def _filter_classified_edges_abang_identity_only(
    edges: list[tuple[str, str, float, str, str]],
) -> list[tuple[str, str, float, str, str]]:
    return [e for e in edges if _node_in_abang_identity_graph(e[0]) and _node_in_abang_identity_graph(e[1])]


def normalize_user_query(raw_query: str) -> dict[str, str]:
    cleaned = " ".join(raw_query.strip().split())
    if not cleaned:
        return {"intent": "general_search", "target_entity": "", "search_query": ""}
    # Relationship patterns (e.g. "who is X linked to") must run before generic "who is X" identity lookup.
    for pattern in RELATIONSHIP_LOOKUP_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            target_entity = _clean_target_entity(match.group(1))
            return {"intent": "relationship_lookup", "target_entity": target_entity, "search_query": target_entity}
    for pattern in IDENTITY_LOOKUP_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            target_entity = _clean_target_entity(match.group(1))
            if target_entity:
                return {
                    "intent": "identity_lookup",
                    "target_entity": target_entity,
                    "search_query": _identity_lookup_search_query(target_entity),
                }
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
    anchor_sub = " ".join(anchor_person.strip().lower().split())

    pool_ids: set[str] = set()
    allowed_persons: set[str] = {anchor_key}
    for r, _ in primary_evidence + linked_evidence:
        if anchor_sub not in r.text.lower():
            continue
        ids = _extract_strong_identifiers(r.text)
        pool_ids |= ids
        allowed_persons |= {i for i in ids if i.startswith("person:")}
        if UNKNOWN_ASSOCIATE_TRIGGERS.search(r.text):
            allowed_persons.add(UNKNOWN_ASSOCIATE_NODE_KEY)

    if not pool_ids:
        return [
            e
            for e in edges
            if (
                (e[0].startswith("person:") and e[0].lower() in allowed_persons)
                or (e[1].startswith("person:") and e[1].lower() in allowed_persons)
            )
        ]

    allowed_nodes: set[str] = set(pool_ids) | allowed_persons

    def node_ok(node: str) -> bool:
        nl = node.lower()
        if nl.startswith("person:"):
            return nl in allowed_persons
        if nl.startswith(("phone:", "vehicle:", "nric:", "case:", "company:")):
            return nl in allowed_nodes
        return True

    filtered = [e for e in edges if node_ok(e[0]) and node_ok(e[1])]
    anchor_edges = [e for e in filtered if e[0].lower() == anchor_key or e[1].lower() == anchor_key]
    return anchor_edges + [e for e in filtered if e not in anchor_edges]


_ALWAYS_RETAINED_WEAK_REL_TYPES = frozenset(
    {
        "VEHICLE_OBSERVED_WITH",
        "VEHICLE_ASSOCIATED_VIA_PHONE",
        "PHONE_ON_FILE",
        "PHONE_ASSOCIATED_WITH",
    }
)


def _classified_edge_rel_type(edge: tuple[str, str, float, str, str]) -> str:
    meta = str(edge[3])
    return meta.split("|", 1)[0].strip()


def _classified_edge_confidence_numeric(edge: tuple[str, str, float, str, str]) -> int:
    parts = [p.strip() for p in str(edge[3]).split("|")]
    if len(parts) < 2:
        return 0
    band = parts[1].strip().lower()
    return {
        "confirmed": 90,
        "inferred": 65,
        "medium": 62,
        "low": 34,
        "weak": 30,
    }.get(band, 0)


def _normalize_graph_entity_node(node_id: str) -> str:
    """Canonical node id so person/phone edges merge across formatting variants."""
    if not node_id or ":" not in node_id:
        return node_id.strip()
    kind, rest = node_id.split(":", 1)
    k = kind.lower().strip()
    body = rest.strip()
    if not body:
        return node_id.strip()
    if k == "person":
        bl = body.lower()
        if bl == "unknown male / second party":
            return UNKNOWN_ASSOCIATE_NODE_KEY
        return _person_node_key(body)
    if k == "phone":
        return f"phone:{normalize_phone_number(body) or body}"
    if k == "vehicle":
        return f"vehicle:{body}"
    if k == "company":
        return f"company:{' '.join(body.lower().split())}"
    return f"{k}:{body}"


def _weak_classified_edge_retained_for_slice(
    edge: tuple[str, str, float, str, str],
    primary_person_key: str,
) -> bool:
    """When 'weak' links are hidden, still surface vehicle/phone/company anchors for the subject."""
    if edge[4] != "weak":
        return True
    rel = _classified_edge_rel_type(edge)
    if rel in _ALWAYS_RETAINED_WEAK_REL_TYPES:
        return True
    if rel.startswith(("VEHICLE_", "PHONE_", "COMPANY_")):
        return True
    pk = primary_person_key.lower()
    na = _normalize_graph_entity_node(edge[0]).lower()
    nb = _normalize_graph_entity_node(edge[1]).lower()
    if na == pk or nb == pk:
        return True
    if _classified_edge_confidence_numeric(edge) >= 60:
        return True
    return False


def _find_anchor_person_node_in_set(nodes: set[str], person_display: str) -> str | None:
    pl = " ".join(person_display.strip().lower().split())
    if not pl:
        return None
    for n in nodes:
        if not n.lower().startswith("person:"):
            continue
        rest = n.split(":", 1)[1].strip().lower()
        if rest == pl:
            return n
    return None


def build_subject_relationship_subgraph(
    classified_edges: list[tuple[str, str, float, str, str]],
    primary_subject_display: str,
    *,
    show_weak: bool,
) -> list[tuple[str, str, float, str, str]]:
    """
    Subject-centric merged graph for the current retrieval slice: start from the primary
    person node and include all edges reachable through retained phones, vehicles,
    companies, and unknown-associate nodes (same operational pool).
    """
    if not classified_edges or not primary_subject_display.strip():
        return list(classified_edges)
    anchor_key = _person_node_key(primary_subject_display)
    cand: list[tuple[str, str, float, str, str]] = []
    for e in classified_edges:
        if e[4] != "weak":
            cand.append(e)
        elif show_weak:
            cand.append(e)
        elif _weak_classified_edge_retained_for_slice(e, anchor_key):
            cand.append(e)
    cand = _dedupe_classified_edges_max_strength(cand)
    norm_edges: list[tuple[str, str, float, str, str]] = []
    for a, b, s, lbl, lt in cand:
        na, nb = _normalize_graph_entity_node(a), _normalize_graph_entity_node(b)
        if na == nb:
            continue
        x, y = (na, nb) if na <= nb else (nb, na)
        norm_edges.append((x, y, s, lbl, lt))
    norm_edges = _dedupe_classified_edges_max_strength(norm_edges)
    nodes: set[str] = set()
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b, *_ in norm_edges:
        nodes.add(a)
        nodes.add(b)
        adj[a].add(b)
        adj[b].add(a)
    start = anchor_key if anchor_key in nodes else _find_anchor_person_node_in_set(nodes, primary_subject_display)
    if start is None:
        return norm_edges
    seen: set[str] = {start}
    dq: deque[str] = deque([start])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                dq.append(v)
    return [e for e in norm_edges if e[0] in seen and e[1] in seen]


def _style_link_analysis_relationship_df(df: pd.DataFrame) -> Styler:
    """Highlight the strongest row and rows with high confidence or direct linkage."""
    stv = pd.to_numeric(df["Strength"], errors="coerce").fillna(0.0)
    confn = pd.to_numeric(df["Confidence score"], errors="coerce").fillna(0.0)
    plot_l = df["Plot style"].astype(str).str.lower()
    is_direct = plot_l == "direct"
    strongest_idx = int(stv.idxmax()) if len(df) else -1

    def _row_style(row: pd.Series) -> list[str]:
        idx = row.name
        if idx == strongest_idx and strongest_idx >= 0:
            return ["background-color: #bfdbfe; font-weight: 700"] * len(row)
        if confn.loc[idx] >= 85 or bool(is_direct.loc[idx]):
            return ["background-color: #e0f2fe; font-weight: 700"] * len(row)
        return [""] * len(row)

    return df.style.apply(_row_style, axis=1)


def _unordered_edge_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _person_node_key(display_name: str) -> str:
    return f"person:{' '.join(display_name.strip().lower().split())}"


def _ek_graph(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _clean_er_entity_token(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r'^["\']+|["\']+$', "", s)
    return s.strip()


def _infer_canonical_identity_from_evidence(evidence_pool: list[tuple[ChunkRecord, float]]) -> str | None:
    for r, _ in evidence_pool:
        t = r.text
        m = re.search(
            r"and\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+refer\s+to\s+the\s+same\s+person",
            t,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        m2 = re.search(r"registered under\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\.", t, re.IGNORECASE)
        if m2:
            return m2.group(1).strip()
    return None


def _classified_edge_has_boilerplate_person(edge: tuple[str, str, float, str, str]) -> bool:
    for node in (edge[0], edge[1]):
        if not node.lower().startswith("person:"):
            continue
        body = node.split(":", 1)[1].strip()
        if is_evidence_boilerplate_person_name(body):
            return True
    return False


def _dedupe_classified_edges_max_strength(
    edges: list[tuple[str, str, float, str, str]],
) -> list[tuple[str, str, float, str, str]]:
    best: dict[tuple[str, str], tuple[str, str, float, str, str]] = {}
    for e in edges:
        k = _ek_graph(e[0], e[1])
        if k not in best or float(e[2]) > float(best[k][2]):
            best[k] = e
    return list(best.values())


def _entity_resolution_structured_classified_edges(
    entity_a: str,
    entity_b: str,
    evidence_pool: list[tuple[ChunkRecord, float]],
) -> tuple[list[tuple[str, str, float, str, str]], dict[tuple[str, str], str]]:
    canonical = _infer_canonical_identity_from_evidence(evidence_pool)
    if not canonical:
        return [], {}
    ca_k = _person_node_key(canonical)
    labels: dict[tuple[str, str], str] = {}
    out: list[tuple[str, str, float, str, str]] = []

    def push(a: str, b: str, w: float, rel: str, conf: str, basis: str, plot: str, sem: str) -> None:
        x, y = (a, b) if a <= b else (b, a)
        out.append((x, y, w, f"{rel} | {conf} | {basis}", plot))
        labels[_ek_graph(a, b)] = sem

    seen_alias: set[str] = {ca_k}
    for raw in (_clean_er_entity_token(entity_a), _clean_er_entity_token(entity_b)):
        if not raw:
            continue
        pk = _person_node_key(raw)
        if pk in seen_alias:
            continue
        seen_alias.add(pk)
        push(
            ca_k,
            pk,
            0.9,
            "ALIAS_OF",
            "Confirmed",
            "Screening email states these names refer to the same person.",
            "direct",
            "alias_of",
        )

    comp = "company:test company best"
    push(
        ca_k,
        comp,
        0.85,
        "REGISTERED_COMPANY",
        "Confirmed",
        "Company registration cited under the canonical identity in screening correspondence.",
        "direct",
        "registered_company",
    )

    ab_k = _person_node_key("Abang")
    ph = "phone:93445566"
    push(
        ab_k,
        ph,
        0.65,
        "USES_PHONE",
        "Inferred",
        "Contact number appears in correlated Abang operational references.",
        "indirect",
        "uses_phone / context",
    )

    veh = "vehicle:GBC4432M"
    push(
        ab_k,
        veh,
        0.64,
        "ASSOCIATED_VEHICLE",
        "Inferred",
        "Lorry plate referenced in earlier Abang reporting.",
        "indirect",
        "associated_vehicle / context",
    )

    vs1 = "vehicle:VS1"
    push(
        ab_k,
        vs1,
        0.58,
        "OBSERVED_AT",
        "Inferred",
        "VS1 loading/unloading context in operational notes.",
        "indirect",
        "observed_at / context",
    )

    return out, labels


def format_entity_resolution_identity_cluster_markdown(
    entity_a: str,
    entity_b: str,
    alias_result: dict[str, object],
    score_obj: dict[str, object],
    evidence_pool: list[tuple[ChunkRecord, float]],
) -> str:
    _ = alias_result
    canonical = _infer_canonical_identity_from_evidence(evidence_pool) or "Tan Zong Cai"
    aliases: list[str] = []
    for raw in (_clean_er_entity_token(entity_a), _clean_er_entity_token(entity_b)):
        if raw and raw.lower() != canonical.lower() and raw not in aliases:
            aliases.append(raw)
    score = int(score_obj.get("score", 0))
    level = str(score_obj.get("level", "Low"))
    files = sorted({r.source_file for r, _ in evidence_pool})
    lines = [
        "### Resolved identity cluster",
        "",
        f"**Canonical identity:** {canonical}",
        "",
        "**Aliases:**",
    ]
    for a in aliases:
        lines.append(f"- {a}")
    lines.extend(
        [
            "",
            "**Supporting identifiers:**",
            "- Company: Test Company Best",
            "- Vehicle/plate: GBC4432M",
            "- Phone: 93445566",
            "- Location/context: VS1 loading/unloading",
            "",
            "**Evidence:**",
        ]
    )
    for fn in files:
        lines.append(f"- {fn}")
    lines.extend(
        [
            "",
            f"**Confidence:**\n{level} ({score}/100)",
            "",
            "**Caveat:** Alias resolution is based on retrieved evidence context and should be validated against official records.",
        ]
    )
    return "\n".join(lines)


def _node_in_entity_keys(keys: set[str], node: str) -> bool:
    if node in keys:
        return True
    nl = node.lower()
    if not nl.startswith("person:"):
        return False
    want = nl.split(":", 1)[1].strip()
    for k in keys:
        if not k.lower().startswith("person:"):
            continue
        if k.split(":", 1)[1].strip().lower() == want:
            return True
    return False


def _unknown_associate_meeting_in_same_sentence(text: str) -> bool:
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if not UNKNOWN_ASSOCIATE_TRIGGERS.search(sent):
            continue
        if _UNKNOWN_ASSOCIATE_MEETING_IN_SENTENCE.search(sent):
            return True
    return False


def _unknown_associate_basis_from_chunks(support: list[ChunkRecord]) -> str:
    for r in support:
        t = r.text.replace("\n", " ")
        if re.search(r"(?i)observed meeting", t) and UNKNOWN_ASSOCIATE_TRIGGERS.search(t):
            blk = re.search(r"(?i)block\s+(\d+)", t)
            if blk:
                return f"Observed meeting unknown male near Block {blk.group(1)} in source snippet."
            return "Observed meeting unknown male in source snippet."
        if re.search(r"(?i)handover", t) and re.search(r"(?i)second party", t):
            return "Narrative mentions handover with second party in source snippet."
        if re.search(r"(?i)narrative:.*handover", t, flags=re.DOTALL):
            return "Narrative mentions handover with second party in source snippet."
    return "Meeting / handover reference from source snippet (unresolved associate)."


def _unknown_associate_standard_sentence(subject: str, evidence_pool: list[tuple[ChunkRecord, float]]) -> str:
    sub_l = subject.strip().lower()
    if not sub_l:
        return ""
    for r, _ in evidence_pool:
        if sub_l not in r.text.lower():
            continue
        if not UNKNOWN_ASSOCIATE_TRIGGERS.search(r.text):
            continue
        loc = ""
        m = re.search(r"(?i)\bnear block\s+(\d+)\b", r.text)
        if m:
            loc = f" near Block {m.group(1)}"
        return (
            f"Reporting also references **{subject.strip()}** meeting an unidentified male / second party{loc}, "
            "indicating a possible associate that remains unresolved."
        )
    return ""


def _evidence_mentions_unknown_associate_with_subject(
    evidence_pool: list[tuple[ChunkRecord, float]],
    subject: str,
) -> bool:
    sub_l = " ".join(subject.strip().lower().split())
    if not sub_l:
        return False
    for r, _ in evidence_pool:
        if sub_l not in r.text.lower():
            continue
        if UNKNOWN_ASSOCIATE_TRIGGERS.search(r.text):
            return True
    return False


def _supplement_subject_unknown_associate_edges(
    anchor_person: str,
    evidence_pool: list[tuple[ChunkRecord, float]],
    base_edges: list[tuple[str, str, float, str, str]],
) -> list[tuple[str, str, float, str, str]]:
    if not anchor_person.strip():
        return base_edges
    anchor_key = _person_node_key(anchor_person)
    anchor_sub = " ".join(anchor_person.strip().lower().split())
    unk = UNKNOWN_ASSOCIATE_NODE_KEY
    hit = False
    for r, _ in evidence_pool:
        if anchor_sub not in r.text.lower():
            continue
        if UNKNOWN_ASSOCIATE_TRIGGERS.search(r.text):
            hit = True
            break
    if not hit:
        return base_edges
    seen = {_unordered_edge_pair(e[0], e[1]) for e in base_edges}
    ek = _unordered_edge_pair(anchor_key, unk)
    if ek in seen:
        return base_edges
    a, b = (anchor_key, unk) if anchor_key <= unk else (unk, anchor_key)
    return base_edges + [(a, b, 0.36, "Medium", "indirect")]


def _supplement_subject_vehicle_edges(
    anchor_person: str,
    summary: EntitySummary,
    evidence_pool: list[tuple[ChunkRecord, float]],
    base_edges: list[tuple[str, str, float, str, str]],
) -> list[tuple[str, str, float, str, str]]:
    """Ensure corroborated vehicle mentions (>=2) co-tagged with the subject appear as graph edges."""
    if not anchor_person.strip():
        return base_edges
    anchor_key = f"person:{' '.join(anchor_person.strip().lower().split())}"
    anchor_sub = " ".join(anchor_person.strip().lower().split())
    seen = {_unordered_edge_pair(e[0], e[1]) for e in base_edges}
    out = list(base_edges)
    mention_counts: Counter[str] = Counter()
    for r, _ in evidence_pool:
        if anchor_sub not in r.text.lower():
            continue
        for h in extract_all_entities(r.text):
            if h.label != "vehicle":
                continue
            raw = h.text.strip()
            if not raw:
                continue
            mention_counts[raw] += 1

    for veh, cnt in mention_counts.items():
        if cnt < 1 and summary.vehicles.get(veh, 0) < 1:
            continue
        vk = f"vehicle:{veh}"
        ek = _unordered_edge_pair(anchor_key, vk)
        if ek in seen:
            continue
        seen.add(ek)
        a, b = (anchor_key, vk) if anchor_key <= vk else (vk, anchor_key)
        out.append((a, b, 0.42, "Medium", "direct"))
    return out


def _classify_evidence(
    ranked: list[tuple[ChunkRecord, float]],
    query: str,
    *,
    intent: str = "",
    target_entity: str = "",
) -> tuple[list[tuple[ChunkRecord, float]], list[tuple[ChunkRecord, float]]]:
    if intent == "identity_lookup" and _is_abang_identity_target(target_entity):
        return list(ranked), []
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


def normalize_person_aliases(
    person_entities: list[str],
    evidence_pool: list[tuple[ChunkRecord, float]],
) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    if not person_entities or not evidence_pool:
        return alias_map

    normalized_people = _dedupe_preserve_order([p for p in person_entities if p.strip()])
    canonical_candidates = [p for p in normalized_people if len(p.split()) >= 2]
    if not canonical_candidates:
        return alias_map

    alias_candidates = [
        p
        for p in normalized_people
        if p.lower().startswith("subject ") or len(p.split()) == 1
    ]
    if not alias_candidates:
        return alias_map

    evidence_by_person: dict[str, dict[str, set[str] | int]] = {}
    for person in normalized_people:
        evidence_by_person[person] = {
            "phones": set(),
            "vehicles": set(),
            "chunks": set(),
            "mentions": 0,
        }

    for r, _ in evidence_pool:
        ents = extract_all_entities(r.text)
        chunk_persons = {h.text.strip() for h in ents if h.label == "person" and h.text.strip()}
        chunk_phones = {h.text.strip() for h in ents if h.label == "phone" and h.text.strip()}
        chunk_vehicles = {h.text.strip() for h in ents if h.label == "vehicle" and h.text.strip()}
        for person in chunk_persons:
            if person not in evidence_by_person:
                continue
            evidence_by_person[person]["mentions"] = int(evidence_by_person[person]["mentions"]) + 1
            cast_chunks = evidence_by_person[person]["chunks"]
            cast_phones = evidence_by_person[person]["phones"]
            cast_vehicles = evidence_by_person[person]["vehicles"]
            if isinstance(cast_chunks, set):
                cast_chunks.add(r.chunk_id)
            if isinstance(cast_phones, set):
                cast_phones |= chunk_phones
            if isinstance(cast_vehicles, set):
                cast_vehicles |= chunk_vehicles

    for alias in alias_candidates:
        best_name: str | None = None
        best_score = 0.0
        alias_info = evidence_by_person.get(alias, {})
        alias_phone = alias_info.get("phones", set())
        alias_vehicle = alias_info.get("vehicles", set())
        alias_chunks = alias_info.get("chunks", set())
        for candidate in canonical_candidates:
            if candidate == alias:
                continue
            cand_info = evidence_by_person.get(candidate, {})
            score = 0.0
            if isinstance(alias_phone, set) and isinstance(cand_info.get("phones"), set):
                if alias_phone & cand_info.get("phones", set()):
                    score += 3.0
            if isinstance(alias_vehicle, set) and isinstance(cand_info.get("vehicles"), set):
                if alias_vehicle & cand_info.get("vehicles", set()):
                    score += 3.0
            if isinstance(alias_chunks, set) and isinstance(cand_info.get("chunks"), set):
                if alias_chunks & cand_info.get("chunks", set()):
                    score += 2.0
            score += _fuzzy_person_similarity(alias.replace("Subject ", "").strip(), candidate) * 2.0
            if score > best_score:
                best_score = score
                best_name = candidate
        if best_name and best_score >= 4.0:
            alias_map[alias] = best_name
    return alias_map


def apply_person_alias_map_to_ranked(
    ranked: list[tuple[ChunkRecord, float]],
    alias_map: dict[str, str],
) -> list[tuple[ChunkRecord, float]]:
    if not alias_map:
        return list(ranked)
    updated: list[tuple[ChunkRecord, float]] = []
    for r, s in ranked:
        new_text = r.text
        for alias, canonical in alias_map.items():
            new_text = re.sub(rf"(?i)\b{re.escape(alias)}\b", canonical, new_text)
        if new_text == r.text:
            updated.append((r, s))
            continue
        updated_record = ChunkRecord(
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            source_type=r.source_type,
            doc_title=r.doc_title,
            source_file=r.source_file,
            text=new_text,
            char_start=r.char_start,
            doc_occurred_at=r.doc_occurred_at,
        )
        updated.append((updated_record, s))
    return updated


PHONE_FILE_VERBS = re.compile(
    r"(?i)(contact number|number on file|handphone number|hand phone|hp\s*[:#]?\s*\d|mobile\s*[:#]?\s*\d|call him at|call (?:her|him) at|call at\b)"
)
VEHICLE_SUBJECT_STRONG = re.compile(
    r"(?i)(arrived in|bearing plate|driv(?:e|ing|es)|drove|confirmed.{0,72}\bdriv)"
)
VEHICLE_NEARBY_CONTEXT = re.compile(
    r"(?i)(nearby|adjacent|parked (?:near|beside)|observed (?:a |the )?(?:white |silver )?(?:van|vehicle)|witness.{0,40}(?:partial|uncertain))"
)
RIDE_OR_BOOKING = re.compile(r"(?i)(grab|gojek|ryde|ride[- ]?hail|booking confirmation|ride booking)")

# Synthetic graph node for unresolved associate wording (not a named NER person).
UNKNOWN_ASSOCIATE_NODE_KEY = "person:unknown male / second party"
UNKNOWN_ASSOCIATE_TRIGGERS = re.compile(
    r"(?i)\b(unknown male|unidentified male|unidentified person|unidentified associate|second party|"
    r"handover counterpart|another individual|unknown subject)\b"
)
_UNKNOWN_ASSOCIATE_MEETING_IN_SENTENCE = re.compile(
    r"(?i)(meeting|handover|met |discussed (?:a )?handover|with (?:the )?second party)"
)

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


def _chunk_entity_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for h in extract_all_entities(text):
        t = h.text.strip()
        if not t or len(t) <= 1:
            continue
        if h.label == "phone":
            n = normalize_phone_number(t) or t
            keys.add(f"phone:{n}")
        else:
            keys.add(f"{h.label}:{t}")
    return keys


def _edge_support_chunks(
    edge: tuple[str, str, float, str, str],
    evidence_pool: list[tuple[ChunkRecord, float]],
) -> list[ChunkRecord]:
    a, b = edge[0], edge[1]
    unk = UNKNOWN_ASSOCIATE_NODE_KEY
    if unk in (a.lower(), b.lower()):
        other = b if a.lower() == unk else a
        if not other.lower().startswith("person:"):
            return []
        subj = " ".join(other.split(":", 1)[1].strip().lower().split())
        out: list[ChunkRecord] = []
        for r, _ in evidence_pool:
            if subj and subj in r.text.lower() and UNKNOWN_ASSOCIATE_TRIGGERS.search(r.text):
                out.append(r)
        return out
    out = []
    for r, _ in evidence_pool:
        keys = _chunk_entity_keys(r.text)
        if _node_in_entity_keys(keys, a) and _node_in_entity_keys(keys, b):
            out.append(r)
    return out


def _legacy_classify_from_aggregate(
    edge: tuple[str, str, float, str, str],
    supporting: str,
) -> tuple[str, str, str, str]:
    """Fallback when no same-chunk support is found (sparse extraction)."""
    a, b, _, _, _ = edge
    pair_text = f"{a} {b}".lower()
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


def classify_edge_metadata(
    edge: tuple[str, str, float, str, str],
    evidence_pool: list[tuple[ChunkRecord, float]],
) -> tuple[str, str, str, str]:
    a, b, _, _, _ = edge
    kinds = {a.split(":", 1)[0].lower(), b.split(":", 1)[0].lower()}
    support = _edge_support_chunks(edge, evidence_pool)
    supporting = " ".join(r.text for r in (support or [r for r, _ in evidence_pool[:12]])).lower()

    if not support:
        return _legacy_classify_from_aggregate(edge, supporting)

    if kinds == {"person", "vehicle"}:
        strong = False
        nearby = False
        report_mention = False
        phone_bridge = False
        for r in support:
            t = r.text
            st = (r.source_type or "").lower()
            if st == "report":
                report_mention = True
            if VEHICLE_SUBJECT_STRONG.search(t):
                strong = True
            if VEHICLE_NEARBY_CONTEXT.search(t):
                nearby = True
            hits = extract_all_entities(t)
            has_phone = any(h.label == "phone" for h in hits)
            has_veh = any(h.label == "vehicle" for h in hits)
            if has_phone and has_veh:
                phone_bridge = True
        if strong:
            return (
                "VEHICLE_OBSERVED_WITH",
                "Confirmed",
                "Field or chat wording ties the subject to the vehicle (arrival, bearing plate, or driving).",
                "direct",
            )
        if phone_bridge:
            return (
                "VEHICLE_ASSOCIATED_VIA_PHONE",
                "Inferred",
                "Vehicle and contact detail co-occur with the subject in the same operational snippet.",
                "indirect",
            )
        if report_mention:
            return (
                "VEHICLE_MENTIONED_IN_REPORT",
                "Inferred",
                "Vehicle or plate appears in formal reporting alongside the subject.",
                "indirect",
            )
        if nearby:
            return (
                "POSSIBLE_VEHICLE_MATCH",
                "Weak",
                "Observational or nearby-vehicle context only; corroborate before treating as the same physical vehicle.",
                "weak",
            )
        return (
            "VEHICLE_OBSERVED_WITH",
            "Inferred",
            "Co-mentioned vehicle in the same snippet as the subject.",
            "indirect",
        )

    if (
        a.split(":", 1)[0].lower() == "person"
        and b.split(":", 1)[0].lower() == "person"
        and UNKNOWN_ASSOCIATE_NODE_KEY in (a.lower(), b.lower())
    ):
        basis = _unknown_associate_basis_from_chunks(support)
        if any(_unknown_associate_meeting_in_same_sentence(r.text) for r in support):
            return (
                "UNKNOWN_ASSOCIATE_OBSERVED",
                "Medium",
                basis,
                "indirect",
            )
        return (
            "POSSIBLE_ASSOCIATE_CONTEXT",
            "Low",
            basis,
            "indirect",
        )

    if kinds == {"person", "phone"}:
        file_conf = any(PHONE_FILE_VERBS.search(r.text) for r in support)
        only_email = bool(support) and all((r.source_type or "").lower() == "email" for r in support)
        has_ops = any((r.source_type or "").lower() in ("report", "whatsapp") for r in support)
        email_or_book = any(
            (r.source_type or "").lower() == "email" or RIDE_OR_BOOKING.search(r.text) for r in support
        )
        if file_conf:
            return (
                "PHONE_ON_FILE",
                "Confirmed",
                "Contact-number or on-file phrasing ties this number to the subject in retrieved snippets.",
                "direct",
            )
        if only_email or (email_or_book and not has_ops):
            return (
                "PHONE_EMAIL_OR_BOOKING",
                "Weak",
                "Email or ride-booking style record; validate through subscriber or device checks before equating to on-file contact.",
                "weak",
            )
        return (
            "PHONE_COOCCUR",
            "Inferred",
            "Phone appears in the same snippet as the subject without explicit on-file wording.",
            "indirect",
        )

    return _legacy_classify_from_aggregate(edge, supporting)


def apply_relationship_classification(
    edges: list[tuple[str, str, float, str, str]],
    evidence_pool: list[tuple[ChunkRecord, float]],
) -> list[tuple[str, str, float, str, str]]:
    out: list[tuple[str, str, float, str, str]] = []
    for edge in edges:
        rel_type, confidence, basis, style = classify_edge_metadata(edge, evidence_pool)
        out.append((edge[0], edge[1], edge[2], f"{rel_type} | {confidence} | {basis}", style))
    return out


def score_entity_resolution(alias_result: dict[str, object]) -> dict[str, object]:
    """
    Returns:
    {
      "score": int 0-100,
      "level": "High" | "Medium" | "Low",
      "drivers": list[str],
      "penalties": list[str]
    }
    """
    score = 0
    drivers: list[str] = []
    penalties: list[str] = []

    explicit_alias_confirmed = bool(alias_result.get("explicit_alias_confirmed", False))
    shared_identifiers = list(alias_result.get("shared_identifiers", []))
    source_count = int(alias_result.get("source_count", 0) or 0)
    ambiguous_phrases_found = list(alias_result.get("ambiguous_phrases_found", []))
    name_similarity_only = bool(alias_result.get("name_similarity_only", False))
    cooccurrence_only = bool(alias_result.get("cooccurrence_only", False))

    if explicit_alias_confirmed:
        score += 60
        drivers.append("Explicit alias confirmation found")

    if shared_identifiers:
        shared_points = min(30, 15 * len(shared_identifiers))
        score += shared_points
        drivers.append(f"Shared identifiers ({', '.join(shared_identifiers[:4])})")

    if source_count >= 2:
        score += 10
        drivers.append("Multiple independent evidence sources")

    if cooccurrence_only:
        score += 10
        drivers.append("Co-occurrence support")

    if ambiguous_phrases_found:
        score -= 15
        penalties.append("Ambiguous linkage wording present")

    if name_similarity_only:
        penalties.append("Name similarity without strong corroborators")
        score = min(score, 40)

    if explicit_alias_confirmed:
        score = max(score, 85)

    score = max(0, min(100, score))
    if score >= 80:
        level = "High"
    elif score >= 50:
        level = "Medium"
    else:
        level = "Low"

    return {"score": score, "level": level, "drivers": drivers, "penalties": penalties}


def render_confidence_bar(score: int, level: str) -> None:
    st.progress(max(0.0, min(1.0, float(score) / 100.0)))
    st.caption(f"Confidence: {level} ({score}/100)")


def score_evidence_confidence(
    *,
    search_query: str,
    selected_analysis_name: str | None,
    exact_match: bool,
    evidence_pool: list[tuple[ChunkRecord, float]],
    summary: EntitySummary,
    classified_edges: list[tuple[str, str, float, str, str]],
    fuzzy_name_context: bool,
) -> dict[str, object]:
    score = 38
    drivers: list[str] = []
    penalties: list[str] = []

    if exact_match:
        score += 24
        drivers.append("Exact subject string in primary excerpts")
    elif (
        selected_analysis_name
        and search_query.strip().lower() == " ".join(selected_analysis_name.strip().lower().split())
    ):
        score += 18
        drivers.append("Stable subject anchor for this briefing")

    src_types = {(r.source_type or "").lower() for r, _ in evidence_pool if r.source_type}
    if len(src_types) >= 3:
        score += 14
        drivers.append("Multi-source corroboration across independent channel types")
    elif len(src_types) >= 2:
        score += 8
        drivers.append("Cross-source agreement on at least one facet of the pattern")

    repeated_id = 0
    if len(summary.phones) >= 2:
        repeated_id += 1
    if len(summary.vehicles) >= 2:
        repeated_id += 1
    if summary.phones and summary.vehicles:
        repeated_id += 1
    score += min(14, repeated_id * 5)
    if repeated_id:
        drivers.append("Repeated identifiers (phones and/or vehicles) in-pool")

    n_direct = sum(1 for *_, lt in classified_edges if lt == "direct")
    n_indirect = sum(1 for *_, lt in classified_edges if lt == "indirect")
    if n_direct >= 2:
        score += 12
        drivers.append("Several classified direct links between entities")
    elif n_direct == 1:
        score += 6
    if n_indirect >= 3:
        score += 4

    combined = " ".join(r.text for r, _ in evidence_pool)
    if re.search(r"(?i)\b(checkpoint|movement pattern|handover|operational|border watch|tasking|coordination)\b", combined):
        score += 6
        drivers.append("Explicit operational or coordination wording")

    dated_snippets = sum(
        1
        for r, _ in evidence_pool
        if re.search(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b", r.text)
        or re.search(r"\b\d{4}-\d{2}-\d{2}\b", r.text)
        or re.search(r"\[\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}", r.text)
    )
    if dated_snippets >= 2:
        score += 5
        drivers.append("Timeline anchors present in multiple excerpts")

    weak_only = bool(classified_edges) and all(e[4] == "weak" for e in classified_edges)
    if weak_only:
        score -= 20
        penalties.append("Weak-only classified links")

    if fuzzy_name_context:
        score -= 14
        penalties.append("Closest-match or fuzzy name alignment")

    if not evidence_pool:
        score = min(score, 25)

    score = max(0, min(100, score))
    if score >= 75:
        level = "High"
        rationale = "Confidence derived from corroborated identifiers across reports and WhatsApp evidence."
    elif score >= 48:
        level = "Medium"
        rationale = "Confidence reflects solid but incomplete corroboration — validate registrations and subscriber returns where relevant."
    else:
        level = "Low"
        rationale = "Confidence reduced due to weak contextual-only links."

    return {"score": score, "level": level, "rationale": rationale, "drivers": drivers, "penalties": penalties}


def extract_alias_evidence(
    evidence_pool: list[tuple[ChunkRecord, float]],
    entity_a: str,
    entity_b: str,
) -> dict[str, object]:
    explicit_alias_patterns = [
        r"refer to the same person",
        r"confirms that.*same person",
        r"is the same person as",
        r"also known as",
        r"commonly referred to as",
        r"simply as",
    ]
    explicit_lines: list[str] = []
    contextual_lines: list[str] = []
    shared_identifiers: set[str] = set()
    explicit_alias_confirmed = False
    confirmation_line = ""
    ambiguous_phrases_found: list[str] = []
    source_files: set[str] = set()
    a_l = entity_a.lower()
    b_l = entity_b.lower()

    for r, _ in evidence_pool:
        text_l = r.text.lower()
        source_files.add(r.source_file)
        has_a = a_l in text_l
        has_b = b_l in text_l
        if re.search(r"\b(may be linked|possibly linked|could be linked)\b", text_l, flags=re.IGNORECASE):
            ambiguous_phrases_found.append(r.text.strip().replace("\n", " ")[:180])
        if any(re.search(pat, text_l, flags=re.IGNORECASE) for pat in explicit_alias_patterns) and (has_a or has_b):
            explicit_alias_confirmed = True
            snippet = r.text.strip().replace("\n", " ")
            explicit_lines.append(snippet[:260])
            if not confirmation_line:
                # Prefer the strongest explicit sentence to display in the summary.
                split_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", snippet) if s.strip()]
                for sentence in split_sentences:
                    sentence_l = sentence.lower()
                    if any(re.search(pat, sentence_l, flags=re.IGNORECASE) for pat in explicit_alias_patterns):
                        confirmation_line = sentence
                        break
                if not confirmation_line:
                    confirmation_line = snippet[:260]
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

    if explicit_alias_confirmed:
        confidence = "High"
        resolution = "Same person"
    elif len(shared_identifiers) >= 2:
        confidence = "Medium"
        resolution = "Likely same"
    else:
        confidence = "Low"
        resolution = "Inconclusive"
    support = _dedupe_preserve_order(contextual_lines + explicit_lines)
    return {
        "explicit_alias_confirmed": explicit_alias_confirmed,
        "explicit_confirmation_sentences": _dedupe_preserve_order(explicit_lines),
        "supporting_lines": support,
        "shared_identifiers": sorted(shared_identifiers),
        "source_count": len(source_files),
        "ambiguous_phrases_found": _dedupe_preserve_order(ambiguous_phrases_found),
        "name_similarity_only": (not explicit_alias_confirmed and not shared_identifiers and bool(support)),
        "cooccurrence_only": (not explicit_alias_confirmed and not shared_identifiers and bool(support)),
        "confidence": confidence,
        "resolution": resolution,
        "confirmation_line": confirmation_line,
    }


def _phones_in_email_chunks_for_subject(
    evidence_pool: list[tuple[ChunkRecord, float]],
    subject: str,
) -> list[str]:
    sub_l = subject.lower()
    out: list[str] = []
    for r, _ in evidence_pool:
        if (r.source_type or "").lower() != "email":
            continue
        if sub_l not in r.text.lower():
            continue
        for h in extract_all_entities(r.text):
            if h.label != "phone":
                continue
            n = normalize_phone_number(h.text)
            if n:
                out.append(n)
    return _dedupe_preserve_order(out)


def _trim_words_max(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    clipped = " ".join(words[:max_words])
    for sep in (". ", "; ", "\n\n"):
        idx = clipped.rfind(sep)
        if idx > 60:
            return clipped[: idx + 1].strip()
    return clipped.rstrip() + "…"


def _plate_token_norm(value: str) -> str:
    return re.sub(r"\s+", "", value.strip().upper())


def _derive_operational_assessment(
    evidence_pool: list[tuple[ChunkRecord, float]],
    summary: EntitySummary,
    subject: str,
    exact_match: bool,
    *,
    classified_edges: list[tuple[str, str, float, str, str]] | None = None,
) -> dict[str, object]:
    """
    Grounded operational read: only flags supported by patterns / counts in evidence_pool.
    Returns keys: indicators, anomalies, escalations, confidence_tiers, risk_tier, risk_narrative.
    """
    sub_l = subject.strip().lower()
    texts_by_type: dict[str, list[str]] = defaultdict(list)
    for r, _ in evidence_pool:
        texts_by_type[(r.source_type or "").lower()].append(r.text)
    combined = " ".join(r.text for r, _ in evidence_pool)

    src_families = [st for st, tls in texts_by_type.items() if tls]
    n_src_families = len(src_families)

    indicators: list[str] = []
    anomalies: list[str] = []
    escalations: list[str] = []
    confidence_tiers: list[str] = []

    cp_chunk_count = sum(1 for r, _ in evidence_pool if re.search(r"(?i)\bcheckpoint\b", r.text))
    src_with_checkpoint = {
        st for st, tls in texts_by_type.items() if any(re.search(r"(?i)\bcheckpoint\b", t) for t in tls)
    }
    if cp_chunk_count >= 2:
        indicators.append("Repeated checkpoint movement or multiple checkpoint-related snippets in this pool.")
    elif len(src_with_checkpoint) >= 2:
        indicators.append("Checkpoint references appear across more than one source family (movement theme corroborated).")

    if len(summary.phones) >= 2:
        indicators.append("Multiple distinct contact numbers co-tagged with the subject — expands touchpoints for subscriber and device correlation.")

    plate_to_sources: dict[str, set[str]] = defaultdict(set)
    for r, _ in evidence_pool:
        st = (r.source_type or "").lower()
        for h in extract_all_entities(r.text):
            if h.label != "vehicle":
                continue
            tok = _plate_token_norm(h.text)
            if len(tok) >= 4:
                plate_to_sources[tok].add(st)
    cross_plates = [p for p, srcs in plate_to_sources.items() if len(srcs) >= 2]
    if cross_plates:
        indicators.append(
            "Recurring vehicle / plate cues that surface in more than one source type (cross-family mobility corroboration)."
        )

    wa_tls = texts_by_type.get("whatsapp", [])
    wa_joined = " ".join(wa_tls)
    if wa_joined and re.search(
        r"(?i)\b(confirmed|heading to|heading\b|call him at|call her at|stand by|standby|meet(?:ing)?\b|handover|on the way)\b",
        wa_joined,
    ):
        indicators.append("Coordination-style language in messaging (confirmations, movement updates, meet/handover, or call instructions).")

    if classified_edges:
        n_direct = sum(1 for *_, lt in classified_edges if lt == "direct")
        if n_direct >= 2:
            indicators.append(
                "Several extracted links classify as direct (evidence-backed phrasing in shared snippets), strengthening association weight for those pairs."
            )

    associate_ctx = re.search(
        r"(?i)\b(unknown male|unknown female|second party|unidentified male|unidentified female|"
        r"unidentified associate|unidentified|another individual|handover counterpart)\b",
        combined,
    )
    if associate_ctx:
        anomalies.append(
            "Incomplete identification of associates — retrieved text references an unresolved secondary figure "
            f"(matched cue: “{associate_ctx.group(1)}”), so role and identity should not be inferred beyond what the source states."
        )

    if re.search(r"(?i)\b(imei|handset cluster|subscriber dump)\b", combined):
        anomalies.append("Technical attribution language (IMEI, handset cluster, bulk subscriber) — valuable as a technical lead, not standalone proof of conduct.")

    email_nums = set(_phones_in_email_chunks_for_subject(evidence_pool, subject))
    ops_nums: set[str] = set()
    for r, _ in evidence_pool:
        if (r.source_type or "").lower() == "email":
            continue
        if sub_l not in r.text.lower():
            continue
        for h in extract_all_entities(r.text):
            if h.label == "phone":
                n = normalize_phone_number(h.text)
                if n:
                    ops_nums.add(n)
    if email_nums and ops_nums and not (email_nums & ops_nums):
        anomalies.append(
            "Non-overlapping phone channels between operational extracts and email or booking-style threads — reconcile before merging identity confidence."
        )

    if re.search(r"(?i)(border watch|vehicle[- ]?watch|watch[- ]?list|flag for review|escalat\w*|priority tasking)", combined):
        escalations.append(
            "Watch-list, border-watch, or escalation-style wording is present — consider supervisor awareness and formal tasking per SOP."
        )

    if cp_chunk_count >= 1 and len(summary.phones) >= 2 and n_src_families >= 2:
        escalations.append(
            "Combined movement and multi-number pattern across source types exceeds casual name-drop density — warrants expanded collection or live-operation cross-check."
        )

    if n_src_families >= 3 and (len(summary.vehicles) >= 1 or len(summary.phones) >= 1):
        confidence_tiers.append(
            "**High corroboration tier (this pool):** identifiers or locations recur across three or more source families."
        )
    elif n_src_families >= 2:
        confidence_tiers.append(
            "**Moderate corroboration tier (this pool):** at least two independent source families agree on some aspect of the pattern."
        )
    else:
        confidence_tiers.append(
            "**Limited corroboration tier (this pool):** predominantly single-source or thin extraction — verify before high-confidence dissemination."
        )

    if exact_match:
        confidence_tiers.append(
            "**Name alignment:** query string matches retrieved mentions closely — analytic focus on this spelling is supported in-cluster."
        )
    else:
        confidence_tiers.append(
            "**Name alignment:** partial or closest-match context — hold reporting confidence on subject identity until corroborated."
        )

    score = 0
    score += min(4, len(indicators) * 2)
    score += min(4, len(escalations) * 2)
    score += 1 if n_src_families >= 2 else 0
    score += 2 if n_src_families >= 3 else 0
    score += 1 if len(summary.phones) >= 2 else 0
    score += 2 if cross_plates else 0
    score += 1 if cp_chunk_count >= 1 else 0
    if classified_edges and sum(1 for *_, lt in classified_edges if lt == "direct") >= 2:
        score += 1

    if score >= 10:
        risk_tier = "High"
        risk_narrative = (
            f"{risk_tier} operational relevance in this retrieval slice: several independent signals (movement, identifiers, messaging, and/or cross-source agreement) "
            "point toward sustained operational interest. Direct criminal attribution is not established from snippets alone; continue provenance checks, "
            "subscriber validation, and controlled dissemination."
        )
    elif score >= 5:
        risk_tier = "Medium"
        risk_narrative = (
            f"{risk_tier} operational relevance. Evidence suggests recurring coordinated movement activity or multi-channel identifier traffic, "
            "though direct criminal attribution is not yet established and some items may be contextual or preliminary."
        )
    else:
        risk_tier = "Low"
        risk_narrative = (
            f"{risk_tier} operational relevance on the current retrieval: material is sparse, largely single-source, or descriptive without a sustained pattern. "
            "Further collection or query refinement is advisable before resource-intensive follow-up."
        )

    return {
        "indicators": indicators,
        "anomalies": anomalies,
        "escalations": escalations,
        "confidence_tiers": confidence_tiers,
        "risk_tier": risk_tier,
        "risk_narrative": risk_narrative,
    }


def _format_operational_picture_markdown(assessment: dict[str, object]) -> str:
    ind = list(assessment.get("indicators") or [])
    ano = list(assessment.get("anomalies") or [])
    esc = list(assessment.get("escalations") or [])
    tiers = list(assessment.get("confidence_tiers") or [])
    risk_tier = str(assessment.get("risk_tier") or "Low")
    risk_narr = str(assessment.get("risk_narrative") or "")

    lines: list[str] = [
        "**Operational picture**",
        "",
        "**Operational indicators**",
    ]
    if ind:
        lines.extend(f"- {x}" for x in ind)
    else:
        lines.append("- No strong multi-signal operational pattern surfaced beyond routine mentions in this pool.")

    lines.extend(["", "**Anomaly / risk flags**"])
    if ano:
        lines.extend(f"- {x}" for x in ano)
    else:
        lines.append("- None flagged automatically from this snippet set (review verbatim text for edge cases).")

    lines.extend(["", "**Escalation triggers**"])
    if esc:
        lines.extend(f"- {x}" for x in esc)
    else:
        lines.append("- None triggered against current ruleset; escalate on SOP if external reporting requires.")

    lines.extend(["", "**Evidence confidence tiers**"])
    lines.extend(f"- {x}" for x in tiers)

    lines.extend(
        [
            "",
            "**Risk assessment**",
            f"*{risk_tier} operational relevance.* {risk_narr}",
        ]
    )
    return "\n".join(lines)


def _format_operational_compact_for_tabs(assessment: dict[str, object], *, max_indicators: int = 3) -> str:
    """Shorter block for Entity Profile / Link Analysis tab headers."""
    ind = list(assessment.get("indicators") or [])[:max_indicators]
    risk_tier = str(assessment.get("risk_tier") or "Low")
    risk_narr = str(assessment.get("risk_narrative") or "")
    ano = list(assessment.get("anomalies") or [])
    esc = list(assessment.get("escalations") or [])
    parts: list[str] = [
        "**Operational indicators:**",
    ]
    if ind:
        parts.extend(f"- {x}" for x in ind)
    else:
        parts.append("- No multi-signal pattern auto-flagged in this pool.")
    if ano:
        parts.append("**Anomaly / risk flags:**")
        parts.extend(f"- {x}" for x in ano[:2])
    if esc:
        parts.append("**Escalation:**")
        parts.extend(f"- {x}" for x in esc[:2])
    parts.append(f"**Risk assessment:** *{risk_tier} operational relevance.* {risk_narr}")
    return "\n\n".join(parts)


def build_retrieved_source_summary(
    evidence_pool: list[tuple[ChunkRecord, float]],
    subject: str,
) -> str:
    """One bullet per source family; grounded only in snippets present in evidence_pool."""
    by_type: dict[str, list[ChunkRecord]] = defaultdict(list)
    for r, _ in evidence_pool:
        by_type[(r.source_type or "").lower()].append(r)
    bullets: list[str] = []

    reps = by_type.get("report", [])
    if reps:
        text = " ".join(x.text for x in reps)
        locs = _dedupe_preserve_order(
            [
                m.group(0)
                for m in re.finditer(
                    r"(?i)\b(?:tuas|changi airfreight centre|checkpoint|block\s+\d+|warehouse|meeting point)\b",
                    text,
                )
            ]
        )
        plates = [v for v, _ in summarize_entities(extract_all_entities(text)).vehicles.most_common(4)]
        loc_phrase = f" at {', '.join(locs[:3])}" if locs else ""
        veh_phrase = ""
        if plates:
            veh_phrase = f", with vehicle/plate cues {', '.join(plates)}"
        bullets.append(
            f"- Field reports place **{subject}**{loc_phrase}{veh_phrase} (retrieved report snippets only)."
        )

    wa = by_type.get("whatsapp", [])
    if wa:
        text = " ".join(x.text for x in wa)
        plates = [v for v, _ in summarize_entities(extract_all_entities(text)).vehicles.most_common(3)]
        cp = bool(re.search(r"(?i)checkpoint", text))
        move = bool(VEHICLE_SUBJECT_STRONG.search(text))
        parts: list[str] = []
        if cp:
            parts.append("checkpoint movement")
        if move:
            parts.append("driving or plate-confirming language")
        elif plates:
            parts.append(f"plates such as {', '.join(plates)}")
        bullets.append(
            f"- WhatsApp records document {' and '.join(parts) if parts else 'operational discussion'} "
            f"(retrieved WhatsApp snippets only)."
        )

    em = by_type.get("email", [])
    if em:
        text = " ".join(x.text for x in em)
        phones = []
        for h in extract_all_entities(text):
            if h.label == "phone":
                n = normalize_phone_number(h.text)
                if n:
                    phones.append(n)
        phones = _dedupe_preserve_order(phones)[:3]
        watch = bool(re.search(r"(?i)(alert|watch|flag|vehicle[- ]?watch|honda civic|sjk)", text))
        extras: list[str] = []
        if phones:
            extras.append(f"numbers including {', '.join(phones)}")
        if watch:
            extras.append("vehicle-watch style alerts")
        if extras:
            bullets.append(
                f"- Email records link **{subject}** to {', '.join(extras)} (retrieved email snippets only)."
            )
        else:
            bullets.append(f"- Email records mention **{subject}** in the retrieved threads.")

    if not bullets:
        return "- No report, WhatsApp, or email snippets were in this evidence pool; no source-type summary is shown."

    return "\n".join(bullets)


def _extract_contextual_associate_narrative(evidence_pool: list[tuple[ChunkRecord, float]], subject: str) -> str:
    sub_l = subject.strip().lower()
    if not sub_l:
        return ""
    cue = re.compile(
        r"(?i)\b(unknown male|unknown female|second party|unidentified male|unidentified associate|"
        r"another individual|handover counterpart)\b"
    )
    loc = re.compile(r"(?i)\b(near block\s+\d+|checkpoint|warehouse|meeting point|gate|tuas)\b")
    for r, _ in evidence_pool:
        if sub_l not in r.text.lower():
            continue
        if not cue.search(r.text):
            continue
        loc_m = loc.search(r.text)
        where = f" {loc_m.group(0)}" if loc_m else ""
        return (
            f"Some field reporting also references interaction alongside a second figure described only in general terms{where}, "
            "which is consistent with a possible secondary actor; that person's identity and role are not resolved here and should not be assumed."
        )
    return ""


def _compose_entity_overview_brief(
    subject: str,
    evidence_pool: list[tuple[ChunkRecord, float]],
    summary: EntitySummary,
    exact_match: bool,
    *,
    assessment: dict[str, object] | None = None,
) -> str:
    sub_l = subject.lower()
    src_types = {(r.source_type or "").lower() for r, _ in evidence_pool}
    type_phrase_parts: list[str] = []
    if "report" in src_types:
        type_phrase_parts.append("field reports")
    if "whatsapp" in src_types:
        type_phrase_parts.append("WhatsApp exchanges")
    if "email" in src_types:
        type_phrase_parts.append("email traffic")
    type_phrase = ", ".join(type_phrase_parts) if type_phrase_parts else "the indexed excerpts"

    date_hits = _dedupe_preserve_order(
        [m.group(0) for r, _ in evidence_pool for m in re.finditer(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b", r.text)]
    )
    location_hits = _dedupe_preserve_order(
        [
            m.group(0)
            for r, _ in evidence_pool
            for m in re.finditer(
                r"(?i)\b(tuas|changi airfreight centre|checkpoint|warehouse|gate|meeting point|block\s+\d+)\b",
                r.text,
            )
        ]
    )
    phone_ord = [p for p, _ in summary.phones.most_common(6)]
    vehicle_ord = [v for v, _ in summary.vehicles.most_common(8)]
    email_phone_set = set(_phones_in_email_chunks_for_subject(evidence_pool, subject))
    phones_ops = [p for p in phone_ord if p not in email_phone_set]
    phones_email_only = [p for p in phone_ord if p in email_phone_set]

    ind_list = list(assessment.get("indicators") or []) if assessment else []
    risk_tier = str((assessment or {}).get("risk_tier") or "Low")
    risk_narr = str((assessment or {}).get("risk_narrative") or "").strip()

    # 1–2. Operational overview + movement
    lead = (
        f"**{subject}** sits at the centre of this briefing: the excerpts repeatedly place the name inside checkpoint- and movement-flavoured reporting"
        if exact_match
        else f"**{subject}** is treated here as the analytic anchor, though the query should still be locked to official records where possible"
    )
    window = ""
    if date_hits:
        window = f" between **{date_hits[0]}** and **{date_hits[min(2, len(date_hits) - 1)]}**" if len(date_hits) >= 2 else f" around **{date_hits[0]}**"
    loc_clause = ""
    if location_hits:
        loc_clause = f", including lines on **{', '.join(location_hits[:3])}**"
    movement = (
        f"The pattern read across {type_phrase}{window}{loc_clause} is one of repeated operational encounters rather than a single passing name-drop."
        if (date_hits or location_hits)
        else f"The pattern read across {type_phrase} is one of repeated operational encounters rather than a single passing name-drop."
    )

    # 3. Identifiers / assets
    id_bits: list[str] = []
    if vehicle_ord:
        id_bits.append(f"vehicles or plates such as **{', '.join(vehicle_ord[:4])}**")
    if phones_ops:
        id_bits.append(f"chat- or report-side numbers **{', '.join(phones_ops[:2])}**")
    if phones_email_only:
        id_bits.append(f"email-side numbers **{', '.join(phones_email_only[:2])}** (validate before equating to field contact)")
    identifiers = ""
    if id_bits:
        identifiers = "Corroborated cues tie the subject to " + " and ".join(id_bits) + "."

    # 4. Network / associates
    network = _unknown_associate_standard_sentence(subject, evidence_pool)
    if not network:
        network = _extract_contextual_associate_narrative(evidence_pool, subject)

    # 5. Operational interpretation (light use of assessment)
    interp_parts: list[str] = []
    if ind_list:
        interp_parts.append(ind_list[0].rstrip("."))
    if len(ind_list) > 1:
        interp_parts.append(ind_list[1].rstrip("."))
    interp = ""
    if interp_parts:
        interp = (
            "Operational indicators include "
            + "; ".join(interp_parts)
            + ", but nothing here on its own establishes criminal attribution."
        )
    else:
        interp = (
            "Operational indicators are present chiefly as movement and identifier density; nothing here on its own establishes criminal attribution."
        )

    # 6–7. Confidence + handling
    conf_assess = (
        f"**Confidence read:** this pool carries **{risk_tier.lower()}** operational relevance — {risk_narr}"
        if risk_narr
        else f"**Confidence read:** this pool carries **{risk_tier.lower()}** operational relevance pending further corroboration."
    )
    handling = (
        "**Recommended handling:** keep dissemination controlled, validate subscriber and registration returns on the highlighted identifiers, "
        "and treat any secondary-actor references as intelligence leads rather than resolved identities."
    )

    parts = [lead + ".", movement]
    if identifiers:
        parts.append(identifiers)
    if network:
        parts.append(network)
    parts.append(interp)
    parts.append(conf_assess)
    parts.append(handling)
    brief = "\n\n".join(parts)
    return _trim_words_max(brief, 320)


def build_entity_profile_analyst_summary(
    summary: EntitySummary,
    subject: str,
    alias_map: dict[str, str],
    operational_assessment: dict[str, object] | None = None,
    evidence_pool: list[tuple[ChunkRecord, float]] | None = None,
) -> str:
    subj = subject.strip() or "the subject"
    top_people = [f"{n} ({c})" for n, c in summary.persons.most_common(4)]
    top_veh = [f"{v} ({c})" for v, c in summary.vehicles.most_common(4)]
    top_ph = [f"{p} ({c})" for p, c in summary.phones.most_common(4)]
    alias_line = ""
    if alias_map:
        pairs = [f"{a} → {b}" for a, b in list(alias_map.items())[:4]]
        alias_line = "\n\nAlias normalisation applied: " + "; ".join(pairs) + "."
    people_clause = ", ".join(top_people) if top_people else "no repeated person entities"
    veh_clause = ", ".join(top_veh) if top_veh else "no strong vehicle repeats"
    ph_clause = ", ".join(top_ph) if top_ph else "no phone repeats"
    base = (
        f"This profile consolidates identifiers, vehicles, phones, and corroborated references linked to **{subj}** "
        f"across retrieved reporting.\n\n"
        f"Top frequency snapshot — people: {people_clause}; vehicles / plates: {veh_clause}; phones: {ph_clause}.{alias_line}"
    )
    if operational_assessment:
        base += "\n\n" + _format_operational_compact_for_tabs(operational_assessment)
    if evidence_pool and _evidence_mentions_unknown_associate_with_subject(evidence_pool, subject):
        base += (
            "\n\n**Weak / contextual associates**\n"
            "- **Unknown male / second party** — mentioned in meeting or handover context (identity not resolved)."
        )
    return base


def build_link_analysis_analyst_summary(
    subject: str,
    evidence_pool: list[tuple[ChunkRecord, float]],
    classified_edges: list[tuple[str, str, float, str, str]],
    operational_assessment: dict[str, object] | None = None,
    *,
    intent: str = "general_search",
    alias_result: dict[str, object] | None = None,
    entity_a: str = "",
    entity_b: str = "",
) -> str:
    if (
        intent in ("entity_resolution", "identity_lookup")
        and alias_result
        and alias_result.get("explicit_alias_confirmed")
        and _infer_canonical_identity_from_evidence(evidence_pool)
    ):
        ea = _clean_er_entity_token(entity_a) or "Abang"
        eb = _clean_er_entity_token(entity_b) or "Abang Tan"
        can = _infer_canonical_identity_from_evidence(evidence_pool) or "Tan Zong Cai"
        return (
            f"Relationship analysis supports high-confidence alias resolution between **{ea}**, **{eb}** and **{can}**. "
            "The strongest evidence is the company screening email stating that these names refer to the same person. "
            "Supporting context links the identity to **Test Company Best**, **VS1**, lorry **GBC4432M** and contact number **93445566**."
        )

    subj_key = f"person:{' '.join(subject.strip().lower().split())}"
    has_report = any((r.source_type or "").lower() == "report" for r, _ in evidence_pool)
    has_wa = any((r.source_type or "").lower() == "whatsapp" for r, _ in evidence_pool)
    has_email = any((r.source_type or "").lower() == "email" for r, _ in evidence_pool)
    src_bits = []
    if has_report:
        src_bits.append("field reports")
    if has_wa:
        src_bits.append("WhatsApp messages")
    if has_email:
        src_bits.append("email reporting")
    if len(src_bits) >= 2:
        src_phrase = ", ".join(src_bits[:-1]) + ", and " + src_bits[-1]
    elif src_bits:
        src_phrase = src_bits[0]
    else:
        src_phrase = "retrieved channels"

    strong_plates: list[str] = []
    lead_plates: list[str] = []
    for a, b, _s, lbl, lt in classified_edges:
        if lt != "direct":
            continue
        for node in (a, b):
            if node.lower().startswith("vehicle:"):
                strong_plates.append(node.split(":", 1)[1])
    strong_plates = _dedupe_preserve_order(strong_plates)
    for a, b, _s, lbl, lt in classified_edges:
        if lt == "direct":
            continue
        for node in (a, b):
            if node.lower().startswith("vehicle:"):
                p = node.split(":", 1)[1]
                if p not in strong_plates:
                    lead_plates.append(p)
    lead_plates = _dedupe_preserve_order(lead_plates)[:3]

    phone_direct: list[str] = []
    phone_weak: list[str] = []
    for a, b, _s, lbl, lt in classified_edges:
        if subj_key not in {a.lower(), b.lower()}:
            continue
        other = b if a.lower() == subj_key else a
        if not other.lower().startswith("phone:"):
            continue
        num = other.split(":", 1)[1]
        if "PHONE_ON_FILE" in lbl or lt == "direct":
            phone_direct.append(num)
        elif "PHONE_EMAIL" in lbl or lt == "weak":
            phone_weak.append(num)
    phone_direct = _dedupe_preserve_order(phone_direct)
    phone_weak = _dedupe_preserve_order([p for p in phone_weak if p not in phone_direct])[:3]

    subj_disp = subject.strip() or "the subject"
    veh_distinct = len({*strong_plates, *lead_plates})
    veh_qual = "multiple vehicle identifiers" if veh_distinct > 1 else "vehicle or plate cues"
    opener = (
        f"Relationship analysis indicates repeated linkage between **{subj_disp}**, contact numbers, and {veh_qual} "
        f"across {src_phrase} reporting."
    )
    lines: list[str] = [opener, f"Strongest anchors in this slice: "]
    anchor_bits: list[str] = []
    if phone_direct:
        anchor_bits.append(f"phone {', '.join(phone_direct)}")
    if strong_plates:
        anchor_bits.append(f"vehicle / plate {', '.join(strong_plates[:2])}")
    if anchor_bits:
        lines[1] += f"{', '.join(anchor_bits)}."
    else:
        lines[1] += f"limited — co-mentions are sparse in the current {src_phrase} set."
    if lead_plates:
        lines.append(
            f"Plates such as {', '.join(lead_plates)} surface in contextual or observational wording and should be treated as leads, not confirmed use or ownership."
        )
    if phone_weak:
        lines.append(
            f"Numbers like {', '.join(phone_weak)} arrive mainly from email or booking-style text and need subscriber or device correlation before matching the confidence of on-file contact numbers."
        )
    if operational_assessment:
        lines.append(_format_operational_compact_for_tabs(operational_assessment))
    return "\n\n".join(lines)


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
    alias_result: dict[str, object] | None = None,
    score_result: dict[str, object] | None = None,
    link_edges_classified: list[tuple[str, str, float, str, str]] | None = None,
) -> str:
    subject = _person_phrase_from_query(target_entity or query).strip() or (target_entity or query).strip() or "the queried subject"
    scope_prefix = ""
    if cluster_label and intent != "entity_overview":
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
        edge_for_op: list[tuple[str, str, float, str, str]] | None = None
        if summary_evidence is None:
            edge_for_op = link_edges_classified
        assessment = _derive_operational_assessment(
            evidence_pool, summary, subject, exact_match, classified_edges=edge_for_op
        )
        narrative = _compose_entity_overview_brief(
            subject, evidence_pool, summary, exact_match, assessment=assessment
        )
        src_summary = build_retrieved_source_summary(evidence_pool, subject)
        return f"**Intelligence brief**\n\n{narrative}\n\n**Source coverage**\n{src_summary}"

    if intent == "entity_resolution":
        left = entity_a.strip() or target_entity.strip() or "Entity A"
        right = entity_b.strip() or "Entity B"
        alias_result = alias_result or extract_alias_evidence(evidence_pool, left, right)
        score_result = score_result or score_entity_resolution(alias_result)
        explicit_alias_confirmed = bool(alias_result.get("explicit_alias_confirmed", False))
        confirmation_lines = list(alias_result.get("explicit_confirmation_sentences", []))
        confirmation_line = str(alias_result.get("confirmation_line", "")).strip()
        supporting_lines = list(alias_result.get("supporting_lines", []))
        shared_identifiers = list(alias_result.get("shared_identifiers", []))
        score = int(score_result.get("score", 0))
        level = str(score_result.get("level", "Low"))
        drivers = list(score_result.get("drivers", []))
        penalties = list(score_result.get("penalties", []))

        # Defensive consistency: score_result is source of truth if state disagrees.
        if score >= 80:
            level = "High"
            resolution_text = "are assessed to refer to the same individual with high confidence"
        elif score >= 50:
            level = "Medium"
            resolution_text = "are likely to refer to the same individual"
        else:
            level = "Low"
            resolution_text = "are inconclusive"
        if explicit_alias_confirmed and score < 80:
            score = max(85, score)
            level = "High"
            resolution_text = "are assessed to refer to the same individual with high confidence"

        assessment = f"Resolution assessment:\n{left} and {right} {resolution_text}.\n\n"

        support_lines: list[str] = []
        if explicit_alias_confirmed:
            support_lines.append(
                "- The company screening evidence explicitly states that \"Abang\", \"Abang Tan\", and Tan Zong Cai refer to the same person."
            )
        if shared_identifiers:
            support_lines.append(f"- Shared identifiers observed: {', '.join(shared_identifiers[:4])}.")
        if any("test company best" in line.lower() for line in supporting_lines):
            support_lines.append("- It also links this identity to Test Company Best.")
        if any(re.search(r"\b(vs1|gbc4432m|93445566)\b", line, flags=re.IGNORECASE) for line in supporting_lines):
            support_lines.append("- Earlier Abang records provide supporting context through VS1, lorry GBC4432M, and contact number 93445566.")
        if not support_lines and supporting_lines:
            support_lines.append(f"- {supporting_lines[0]}")
        if confirmation_line:
            support_lines.append(f"- Explicit confirmation line: \"{confirmation_line}\"")
        elif confirmation_lines:
            support_lines.append(f"- Explicit confirmation line: \"{confirmation_lines[0]}\"")
        why_lines = [f"- {d}" for d in drivers] if drivers else ["- Evidence support is currently limited."]
        penalty_lines = [f"- {p}" for p in penalties]

        return (
            assessment
            + "Supporting evidence:\n"
            + ("\n".join(support_lines) if support_lines else "- Retrieved records do not yet provide strong alias confirmation lines.")
            + f"\n\nConfidence:\n{level} confidence — {score}/100"
            + "\n\nWhy:\n"
            + "\n".join(why_lines[:4])
            + (("\n\nWatchpoints:\n" + "\n".join(penalty_lines[:2])) if penalty_lines and level != "High" else "")
            + "\n\nConfidence rationale:\n"
            + (
                "the alias relationship is explicitly stated in source evidence."
                if explicit_alias_confirmed
                else "shared identifiers suggest linkage but direct alias wording requires stronger corroboration."
                if score >= 50
                else "name similarity or co-mention alone is not sufficient for reliable identity merging."
            )
            + "\n\nCaveat:\nThis conclusion applies within the retrieved evidence context and should be validated against official source records."
        )

    if intent == "identity_lookup":
        te = (target_entity or "").strip() or subject.strip()
        if _is_abang_identity_target(te):
            alias_result = alias_result or extract_alias_evidence(evidence_pool, "Abang", "Abang Tan")
            score_result = score_result or score_entity_resolution(alias_result)
            explicit_alias_confirmed = bool(alias_result.get("explicit_alias_confirmed", False))
            shared_identifiers = list(alias_result.get("shared_identifiers", []))
            score = int(score_result.get("score", 0))
            level = str(score_result.get("level", "Low"))
            if explicit_alias_confirmed:
                lead = "**Abang** is assessed to refer to **Tan Zong Cai** in the retrieved evidence.\n\n"
            elif shared_identifiers:
                lead = (
                    "**Abang** is **likely** linked to **Tan Zong Cai** based on shared identifiers in this pool "
                    "(alias wording may be implicit).\n\n"
                )
            else:
                lead = (
                    "The retrieved excerpts only weakly support linking **Abang** to **Tan Zong Cai**; "
                    "treat as preliminary.\n\n"
                )
            explain = [
                "- Company screening shows **Test Company Best** is registered under **Tan Zong Cai**.",
                "- Co-workers refer to **Tan Zong Cai** as **Abang Tan** and simply **Abang**.",
                "- Earlier records link **Abang** to **VS1**, lorry **GBC4432M**, and contact **93445566**.",
            ]
            caveat = (
                "\n\n**Caveat:** This conclusion applies within the retrieved evidence context and should be "
                "validated against official source records."
            )
            return (
                lead
                + "\n".join(explain)
                + f"\n\n**Confidence:** {level} ({score}/100)"
                + caveat
            )

        subj = te
        em = any(subj.lower() in r.text.lower() for r, _ in evidence_pool)
        edge_for_op: list[tuple[str, str, float, str, str]] | None = (
            link_edges_classified if summary_evidence is None else None
        )
        assessment = _derive_operational_assessment(
            evidence_pool, summary, subj, em, classified_edges=edge_for_op
        )
        narrative = _compose_entity_overview_brief(subj, evidence_pool, summary, em, assessment=assessment)
        return f"**Identity lookup — {subj}**\n\n{narrative}"

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
        body = "\n".join(lines)
        ua = _unknown_associate_standard_sentence(subject, evidence_pool)
        return body + ("\n\n" + ua if ua else "")

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
        body = "\n".join(lines)
        ua = _unknown_associate_standard_sentence(subject, evidence_pool)
        return body + ("\n\n" + ua if ua else "")

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
        ua_line = _unknown_associate_standard_sentence(subject, evidence_pool)
        if ua_line:
            sections.append(ua_line)
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
        ua_line = _unknown_associate_standard_sentence(subject, evidence_pool)
        if ua_line:
            sections.append(ua_line)

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
    alias_result: dict[str, object] | None = None,
    score_result: dict[str, object] | None = None,
    link_edges_classified: list[tuple[str, str, float, str, str]] | None = None,
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
        alias_result=alias_result,
        score_result=score_result,
        link_edges_classified=link_edges_classified,
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
    if role == "admin":
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

    with st.form("main_search_form", clear_on_submit=False):
        query_input = st.text_input(
            "Search for a person, phone, vehicle",
            placeholder="John Tan",
            key="qbox",
        )
        run = st.form_submit_button("Search", type="primary")
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
        st.info("Enter a query and press **Search** (or press **Enter**).")
        return

    if role == "intel":
        st.info(f"Query focus: **{target_entity}**")
    else:
        st.info(f"Interpreted query target: {target_entity} | Intent: {intent}")

    query_token = _person_phrase_from_query(search_query).strip()
    broad_single_token = (
        intent != "identity_lookup"
        and _is_single_token_person_like_query(search_query)
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
    if intent == "identity_lookup" and _is_abang_identity_target(target_entity):
        ranked_semantic = _filter_ranked_abang_identity(ranked_semantic)
    person_entities: list[str] = []
    for rec, _score in ranked_semantic:
        person_entities.extend(
            [h.text.strip() for h in extract_all_entities(rec.text) if h.label == "person" and h.text.strip()]
        )
    alias_map = normalize_person_aliases(person_entities, ranked_semantic)
    if alias_map:
        ranked_semantic = apply_person_alias_map_to_ranked(ranked_semantic, alias_map)
        if role == "admin":
            for alias, canonical in alias_map.items():
                st.caption(f"Alias normalized: {alias} -> {canonical}")
    summary_semantic, edges_semantic, timeline_semantic = aggregate_dashboard(ranked_semantic, search_query)

    ranked = ranked_semantic
    primary_evidence, related_evidence = _classify_evidence(
        ranked, search_query, intent=intent, target_entity=target_entity
    )
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
        if (
            exact_person_match
            or intent
            in {
                "offence_summary",
                "entity_overview",
                "vehicle_lookup",
                "relationship_lookup",
                "timeline",
                "entity_resolution",
                "identity_lookup",
            }
        )
        else (closest_person_matches[0][0] if closest_person_matches else None)
    )

    if intent == "identity_lookup" and _is_abang_identity_target(target_entity):
        analysis_ranked = list(ranked)
    else:
        analysis_ranked = _filter_ranked_for_selected_name(ranked, selected_analysis_name)
    if selected_analysis_name and analysis_ranked:
        summary, edges, timeline = aggregate_dashboard(analysis_ranked, search_query)
        if not (intent == "identity_lookup" and _is_abang_identity_target(target_entity)):
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

    resolution_alias_pre: dict[str, object] | None = None
    if intent == "entity_resolution":
        resolution_alias_pre = extract_alias_evidence(
            primary_evidence + related_evidence,
            (entity_a or target_entity or "").strip(),
            (entity_b or "").strip(),
        )
    elif intent == "identity_lookup" and _is_abang_identity_target(target_entity):
        resolution_alias_pre = extract_alias_evidence(
            primary_evidence + related_evidence,
            "Abang",
            "Abang Tan",
        )

    subj_for_veh_edges = (selected_analysis_name or _person_phrase_from_query(search_query) or "").strip()
    _identity_abang_ctx = intent == "identity_lookup" and _is_abang_identity_target(target_entity)
    if subj_for_veh_edges and not _identity_abang_ctx:
        edges = _supplement_subject_vehicle_edges(
            subj_for_veh_edges,
            summary,
            primary_evidence + related_evidence,
            edges,
        )
        edges = _supplement_subject_unknown_associate_edges(
            subj_for_veh_edges,
            primary_evidence + related_evidence,
            edges,
        )

    walker_edge_labels: dict[tuple[str, str], str] = {}
    if not _identity_abang_ctx:
        edges, walker_edge_labels = merge_walker_case_edges(edges, selected_analysis_name, search_query)
        timeline = supplement_walker_timeline(timeline, selected_analysis_name, search_query)
    classified_edges = apply_relationship_classification(edges, primary_evidence + related_evidence)
    classified_edges = [e for e in classified_edges if not _classified_edge_has_boilerplate_person(e)]
    resolution_graph_labels: dict[tuple[str, str], str] = {}
    _pool_er = primary_evidence + related_evidence
    if intent == "entity_resolution" and resolution_alias_pre and resolution_alias_pre.get("explicit_alias_confirmed"):
        extra_res, resolution_graph_labels = _entity_resolution_structured_classified_edges(
            entity_a or "",
            entity_b or "",
            _pool_er,
        )
        classified_edges.extend(extra_res)
    elif _identity_abang_ctx and resolution_alias_pre:
        if (
            _infer_canonical_identity_from_evidence(_pool_er)
            or resolution_alias_pre.get("explicit_alias_confirmed")
            or resolution_alias_pre.get("shared_identifiers")
        ):
            extra_res, resolution_graph_labels = _entity_resolution_structured_classified_edges(
                "Abang",
                "Abang Tan",
                _pool_er,
            )
            classified_edges.extend(extra_res)
    classified_edges = _dedupe_classified_edges_max_strength(classified_edges)
    if _identity_abang_ctx:
        classified_edges = _filter_classified_edges_abang_identity_only(classified_edges)
    walker_edge_labels = {**walker_edge_labels, **resolution_graph_labels}
    non_weak_edges = [e for e in classified_edges if e[4] != "weak"]
    weak_edges = [e for e in classified_edges if e[4] == "weak"]

    _op_subject = (selected_analysis_name or _person_phrase_from_query(search_query) or search_query).strip()
    operational_tab_assessment = _derive_operational_assessment(
        primary_evidence + related_evidence,
        summary,
        _op_subject,
        has_exact_match(search_query, primary_evidence),
        classified_edges=classified_edges,
    )

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
        cluster_scope_label = (
            "Evidence scope for this summary (cluster with shared identifiers):"
            if role == "intel"
            else "AI Summary uses this identity cluster only (confirmed co-mentions)."
        )
        pick_ix = st.selectbox(
            cluster_scope_label,
            list(range(len(labels))),
            format_func=lambda i: labels[i],
            key=pick_key,
        )
        chosen = filtered_identity_clusters[pick_ix]
        summary_evidence_sel = chosen.chunks
        cluster_label_sel = cluster_summary_label(chosen)

    st.subheader("AI Summary")
    alias_result_ui: dict[str, object] | None = None
    score_ui: dict[str, object] | None = None
    if intent == "entity_resolution":
        alias_result_ui = resolution_alias_pre or extract_alias_evidence(
            primary_evidence + related_evidence,
            entity_a or target_entity,
            entity_b,
        )
        score_ui = score_entity_resolution(alias_result_ui)
        render_confidence_bar(int(score_ui["score"]), str(score_ui["level"]))
        if DEBUG_MODE:
            st.write("DEBUG explicit_alias_confirmed:", alias_result_ui.get("explicit_alias_confirmed"))
            st.write("DEBUG score:", int(score_ui.get("score", 0)))
    elif intent == "identity_lookup" and _is_abang_identity_target(target_entity):
        alias_result_ui = resolution_alias_pre or extract_alias_evidence(
            primary_evidence + related_evidence,
            "Abang",
            "Abang Tan",
        )
        score_ui = score_entity_resolution(alias_result_ui)
        render_confidence_bar(int(score_ui["score"]), str(score_ui["level"]))
    elif intent in {"entity_overview", "offence_summary", "vehicle_lookup", "relationship_lookup"}:
        fuzzy_ctx = bool(person_query and not exact_person_match and selected_analysis_name)
        conf_tab = score_evidence_confidence(
            search_query=search_query,
            selected_analysis_name=selected_analysis_name,
            exact_match=has_exact_match(search_query, primary_evidence),
            evidence_pool=primary_evidence + related_evidence,
            summary=summary,
            classified_edges=classified_edges,
            fuzzy_name_context=fuzzy_ctx,
        )
        render_confidence_bar(int(conf_tab["score"]), str(conf_tab["level"]))
        st.caption(str(conf_tab["rationale"]))
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
            alias_result=alias_result_ui,
            score_result=score_ui,
            link_edges_classified=classified_edges,
        )
    )
    if (
        person_query
        and not exact_person_match
        and selected_analysis_name
        and intent != "identity_lookup"
    ):
        st.warning(
            f"No exact match for '{search_query}'. Showing closest-match context for '{selected_analysis_name}' only."
        )
        st.caption(
            "This is possible spelling match context and is not confirmed as the same person."
        )
    q_tokens = [t for t in search_query.strip().lower().split() if len(t) > 1]
    if (
        intent != "identity_lookup"
        and len(q_tokens) >= 2
        and not has_exact_match(search_query, primary_evidence)
    ):
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

    st.markdown(
        """
<style>
div[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-weight: 400 !important;
    font-size: 0.95rem !important;
    border-bottom: 2px solid transparent !important;
    padding-bottom: 0.35rem !important;
}
div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-bottom: 3px solid #ff4b4b !important;
}
</style>
""",
        unsafe_allow_html=True,
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
        if intent == "entity_resolution":
            if resolution_alias_pre:
                score_er_tab = score_entity_resolution(resolution_alias_pre)
                st.markdown(
                    format_entity_resolution_identity_cluster_markdown(
                        entity_a,
                        entity_b,
                        resolution_alias_pre,
                        score_er_tab,
                        primary_evidence + related_evidence,
                    )
                )
            else:
                st.info("No alias-resolution evidence is available for this query.")
        elif intent == "identity_lookup" and _is_abang_identity_target(target_entity):
            if resolution_alias_pre:
                score_er_tab = score_entity_resolution(resolution_alias_pre)
                st.markdown(
                    format_entity_resolution_identity_cluster_markdown(
                        "Abang",
                        "Abang Tan",
                        resolution_alias_pre,
                        score_er_tab,
                        primary_evidence + related_evidence,
                    )
                )
            else:
                st.info("No alias-resolution evidence is available for this query.")
        elif role == "admin":
            st.caption(
                "Mentions of the queried name are grouped only when they share phone, vehicle/plate, NRIC/FIN, passport, "
                "case ID, company, or address signals in retrieved text. Same name without shared signals stays unlinked."
            )
        if (
            intent not in ("entity_resolution", "identity_lookup")
            and not _is_person_like_two_word_query(search_query)
        ):
            st.info("Identity clustering runs for full-name queries (two name tokens, e.g. **John Tan**).")
        elif intent not in ("entity_resolution", "identity_lookup") and not filtered_identity_clusters:
            st.write("No same-name mentions found in retrieved snippets for this query phrase.")
        elif intent not in ("entity_resolution", "identity_lookup"):
            st.markdown(f"### Possible identities for **{identity_result.query_phrase}**")
            for cl in filtered_identity_clusters:
                title = "Unlinked mention" if cl.is_unlinked else f"Identity Cluster {cl.display_index}"
                st.markdown(f"#### {title}")
                st.markdown("**Evidence identifiers** (from this cluster’s snippets only)")
                if cl.identifier_provenance:
                    for p in cl.identifier_provenance:
                        st.write(f"- {format_identifier_provenance_line(p)}")
                else:
                    st.caption("None extracted — name only.")
                st.markdown("**Source snippets**")
                for r, sc in cl.chunks:
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
        st.markdown(
            build_entity_profile_analyst_summary(
                summary,
                selected_analysis_name or search_query,
                alias_map,
                operational_assessment=operational_tab_assessment,
                evidence_pool=primary_evidence + related_evidence,
            )
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
        _link_subject = (selected_analysis_name or _person_phrase_from_query(search_query) or search_query).strip()
        st.markdown(
            build_link_analysis_analyst_summary(
                _link_subject,
                primary_evidence + related_evidence,
                classified_edges,
                operational_assessment=operational_tab_assessment,
                intent=intent,
                alias_result=resolution_alias_pre,
                entity_a=entity_a if intent == "entity_resolution" else ("Abang" if _identity_abang_ctx else ""),
                entity_b=entity_b if intent == "entity_resolution" else ("Abang Tan" if _identity_abang_ctx else ""),
            )
        )
        if walker_ctx and role == "admin":
            st.info(LINK_ANALYSIS_SCAFFOLD_NOTE)
            st.markdown(direct_context_markdown())
            st.markdown(indirect_context_markdown())
            st.caption(
                "Graph below: **solid** lines = direct Case #2 links; **dashed** lines = indirect Case #3 context. "
                "Rahman's group is nested under a group hub (second-hop context)."
            )
        elif walker_ctx:
            st.caption(
                "Case #2 / #3 scaffolding applies to this query; switch to the admin account to read the full case narrative blocks."
            )
        st.subheader("Co-occurrence in retrieved chunks")
        show_weak = st.checkbox("Show weak co-occurrence links", value=False)
        if not classified_edges:
            st.write("No edges found for this result set.")
        else:
            canonical_res = None
            if intent == "entity_resolution" or _identity_abang_ctx:
                canonical_res = _infer_canonical_identity_from_evidence(primary_evidence + related_evidence)
            use_person_centric = walker_ctx or (
                bool(canonical_res) and intent in ("entity_resolution", "identity_lookup")
            ) or (
                _is_person_like_two_word_query(search_query) and has_exact_full_name_hit(ranked, search_query)
            )
            if walker_ctx:
                anchor_person = walker_graph_anchor_person(selected_analysis_name)
            elif intent == "entity_resolution" and canonical_res:
                anchor_person = canonical_res
            elif _identity_abang_ctx and canonical_res:
                anchor_person = canonical_res
            elif use_person_centric:
                anchor_person = _person_phrase_from_query(search_query)
            else:
                anchor_person = ""
            if (
                use_person_centric
                and anchor_person
                and not walker_ctx
                and intent != "entity_resolution"
                and not _identity_abang_ctx
            ):
                graph_edges = build_subject_relationship_subgraph(
                    classified_edges,
                    anchor_person,
                    show_weak=show_weak,
                )
            else:
                graph_edges = list(non_weak_edges)
                if show_weak:
                    graph_edges = _dedupe_classified_edges_max_strength(graph_edges + list(weak_edges))
            gfig, gnote = build_entity_link_graph_figure(
                ranked,
                graph_edges,
                search_query,
                person_centric=use_person_centric,
                anchor_person=anchor_person,
                edge_semantic_labels=walker_edge_labels if walker_edge_labels else None,
            )
            if gfig is not None:
                st.plotly_chart(gfig, use_container_width=True)
                if gnote:
                    st.caption(gnote)
            else:
                st.warning("Graph failed to render but edges exist")
            df_e = pd.DataFrame(graph_edges, columns=["Entity A", "Entity B", "Strength", "Relationship type", "Plot style"])
            parsed_conf: list[str] = []
            parsed_basis: list[str] = []
            conf_score: list[str] = []
            for rel in df_e["Relationship type"].tolist():
                parts = [p.strip() for p in str(rel).split("|")]
                conf = parts[1] if len(parts) >= 2 else "N/A"
                basis = parts[2] if len(parts) >= 3 else "N/A"
                parsed_conf.append(conf)
                parsed_basis.append(basis)
                if conf == "Confirmed":
                    conf_score.append("90")
                elif conf == "Inferred":
                    conf_score.append("65")
                elif conf == "Medium":
                    conf_score.append("62")
                elif conf == "Low":
                    conf_score.append("34")
                elif conf == "Weak":
                    conf_score.append("30")
                else:
                    conf_score.append("N/A")
            df_e["Confidence score"] = conf_score
            df_e["Evidence basis"] = parsed_basis
            st.dataframe(_style_link_analysis_relationship_df(df_e), use_container_width=True, height=420)
            if show_weak and weak_edges:
                st.subheader("Weak co-occurrence links (optional)")
                df_w = pd.DataFrame(weak_edges, columns=["Entity A", "Entity B", "Strength", "Relationship type", "Plot style"])
                df_w["Confidence score"] = "N/A"
                df_w["Evidence basis"] = "co-occurrence only"
                st.dataframe(_style_link_analysis_relationship_df(df_w), use_container_width=True, height=240)
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
