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
from intelligence.link_graph import build_entity_link_graph_figure
from intelligence.loaders import iter_documents
from intelligence.timeline import TimelineEvent, extract_timeline_events, timeline_sort_key


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


def _classify_evidence(
    ranked: list[tuple[ChunkRecord, float]],
    query: str,
) -> tuple[list[tuple[ChunkRecord, float]], list[tuple[ChunkRecord, float]]]:
    person_phrase = _person_phrase_from_query(query).strip()
    person_two_word = _is_person_like_two_word_query(query)
    phone_norm = normalize_phone_number(query)
    vehicle_norm = _vehicle_query_normalized(query)

    def extract_strong_identifiers(text: str) -> set[str]:
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
                primary_ids |= extract_strong_identifiers(pr.text)
            if primary_ids:
                filtered_related: list[tuple[ChunkRecord, float]] = []
                for rr, ss in related:
                    related_ids = extract_strong_identifiers(rr.text)
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


def build_intelligence_summary(
    primary_evidence: list[tuple[ChunkRecord, float]],
    linked_evidence: list[tuple[ChunkRecord, float]],
    query: str,
    *,
    summary_evidence: list[tuple[ChunkRecord, float]] | None = None,
    cluster_label: str | None = None,
) -> str:
    subject = _person_phrase_from_query(query).strip() or query.strip() or "the queried subject"
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

        summary_line = f"{scope_prefix}No confirmed primary subject found for exact query '{query}'."
        next_step = (
            "Refine query (e.g. correct spelling) or inspect Entity Profile for closest matches."
        )

        sections = [f"Summary:\n{summary_line}"]
        closest_lines = [
            f"- {name} ({cnt} mentions)" for name, cnt in summary.persons.most_common(3) if name.strip()
        ]
        if closest_lines:
            sections.append("Closest matches:\n" + "\n".join(closest_lines))
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

    brief = "\n\n".join(sections)
    words = brief.split()
    if len(words) > 220:
        brief = " ".join(words[:220]).rstrip() + "..."
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
    )


def _render_login_gate() -> bool:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"] is True:
        return True

    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state["authenticated"] = True
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

    if not _render_login_gate():
        st.stop()

    st.sidebar.caption("Logged in as admin")
    if st.sidebar.button("Logout"):
        st.session_state.pop("authenticated", None)
        st.session_state.pop("active_query", None)
        st.session_state.pop("qbox", None)
        st.rerun()

    st.title("AI-powered intelligence search")
    st.caption("Semantic search with FAISS, OpenAI embeddings, and entity-aware analytics.")

    default_data_path = Path(__file__).resolve().parent / "data"
    data_root = Path(st.sidebar.text_input("Data folder", value=str(default_data_path)))
    use_cache = st.sidebar.toggle("Use disk cache for index", value=True)
    rebuild_now = st.sidebar.button("Rebuild index now")
    top_k = st.sidebar.slider("Results (FAISS top-K)", min_value=5, max_value=50, value=15)
    keyword_boost = st.sidebar.slider("Keyword boost for hybrid ranking", 0.0, 0.25, 0.12)

    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key

    try:
        _ = get_spacy_ready()
    except Exception:
        pass
    st.sidebar.info("Demo mode: fast entity extraction enabled.")

    if not data_root.is_dir():
        st.warning(f"Data folder does not exist yet: `{data_root}`. Creating sample tree is recommended.")
        st.stop()

    if rebuild_now:
        shutil.rmtree(_cache_base(), ignore_errors=True)
        st.cache_resource.clear()
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
    st.sidebar.success(f"Indexed **{len(store.records)}** chunks from **{source_files}** source files.")
    if st.sidebar.button("Clear active search"):
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
    if not query:
        st.info("Enter a query and press **Search**.")
        return

    query_token = _person_phrase_from_query(query).strip()
    broad_single_token = (
        _is_single_token_person_like_query(query)
        and len(query_token) >= 3
        and not _is_person_like_two_word_query(query)
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

    qv = embed_texts([query])
    raw_hits = store.search(qv[0], k=top_k)
    ranked_semantic = hybrid_rank(raw_hits, query, keyword_boost=keyword_boost)
    summary_semantic, edges_semantic, timeline_semantic = aggregate_dashboard(ranked_semantic, query)

    ranked = ranked_semantic
    summary = summary_semantic
    edges = edges_semantic
    timeline = timeline_semantic
    primary_evidence, related_evidence = _classify_evidence(ranked, query)
    query_person_phrase = _person_phrase_from_query(query)
    corpus_exact_name_hit = _is_person_like_two_word_query(query) and corpus_has_exact_phrase(
        store.records, query_person_phrase
    )

    st.subheader("AI Summary")
    st.write(
        build_ai_summary(
            ranked,
            summary,
            timeline,
            query,
            primary_evidence=primary_evidence,
            linked_evidence=related_evidence,
            corpus_exact_name_hit=corpus_exact_name_hit,
        )
    )
    q_tokens = [t for t in query.strip().lower().split() if len(t) > 1]
    if len(q_tokens) >= 2 and not has_exact_match(query, primary_evidence):
        top_person = summary.persons.most_common(1)[0][0] if summary.persons else None
        if top_person:
            st.info(
                f"A close match was found (e.g. '{top_person}'), but no exact match for your query."
            )
        else:
            st.info(
                "No exact match for your query in primary evidence; "
                "ranked results may still include partial mentions or similarly named people."
            )

    res_tab, ent_tab, rel_tab, time_tab = st.tabs(
        ["Evidence", "Entity Profile", "Link Analysis", "Activity Timeline"]
    )

    with res_tab:
        st.subheader("Primary evidence")
        if not primary_evidence:
            st.caption("No direct mention found in retrieved chunks for this query.")
        for r, score in primary_evidence:
            with st.expander(f"[{r.source_type}] {r.doc_title} — search relevance score {score:.3f}"):
                st.markdown(highlight_query(r.text[:6000], query))
                st.caption(f"`{r.source_file}` · `{r.chunk_id}`")
        st.subheader("Linked evidence")
        st.caption(
            "Shown only when evidence shares strong identifiers with Primary evidence "
            "(for example phone, vehicle/plate, NRIC, case ID, company, or named associate)."
        )
        for r, score in related_evidence:
            with st.expander(f"[{r.source_type}] {r.doc_title} — search relevance score {score:.3f}"):
                st.markdown(highlight_query(r.text[:6000], query))
                st.caption(f"`{r.source_file}` · `{r.chunk_id}`")

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
        if not edges:
            st.write("No edges found for this result set.")
        else:
            use_person_centric = _is_person_like_two_word_query(query) and has_exact_full_name_hit(ranked, query)
            anchor_person = _person_phrase_from_query(query) if use_person_centric else ""
            gfig, gnote = build_entity_link_graph_figure(
                ranked,
                edges,
                query,
                person_centric=use_person_centric,
                anchor_person=anchor_person,
            )
            if gfig is not None:
                st.plotly_chart(gfig, use_container_width=True)
                if gnote:
                    st.caption(gnote)
            else:
                st.warning("Graph failed to render but edges exist")
            df_e = pd.DataFrame(
                edges,
                columns=["Entity A", "Entity B", "Strength", "Strength label", "Link type"],
            )
            st.dataframe(df_e, use_container_width=True, height=420)
            st.subheader("Strongest links")
            for a, b, s, lbl, lt in edges[:12]:
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
