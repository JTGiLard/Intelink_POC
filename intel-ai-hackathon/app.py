from __future__ import annotations

import os
import re
from collections import Counter
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
            f"No documents under {data_root}. Add .txt/.md to data/reports, data/emails, or data/whatsapp."
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


def build_ai_summary(
    ranked: list[tuple[ChunkRecord, float]], summary: EntitySummary, timeline: list[TimelineEvent], query: str
) -> str:
    if not ranked:
        return "Nothing came back for this query, so there is no profile to summarize yet."

    subject = _summary_subject(query, summary)
    n = len(ranked)
    mix = _source_mix_sentence(ranked)
    vehicles = [name for name, _ in summary.vehicles.most_common(2)]
    phones = [name for name, _ in summary.phones.most_common(2)]

    lead = (
        f"The main subject is {subject}. "
        f"This view is built from {n} evidence record{'s' if n != 1 else ''}, "
        f"drawn from {mix}."
    )

    detail_parts: list[str] = []
    if vehicles:
        if len(vehicles) == 1:
            detail_parts.append(f"the vehicle or plate signal {vehicles[0]}")
        else:
            detail_parts.append(f"vehicle or plate signals {vehicles[0]} and {vehicles[1]}")
    if phones:
        if len(phones) == 1:
            detail_parts.append(f"the number {phones[0]}")
        else:
            detail_parts.append(f"numbers {phones[0]} and {phones[1]}")
    if detail_parts:
        if len(detail_parts) == 1:
            middle = f" Along the way, {detail_parts[0]} stood out in the text."
        else:
            middle = f" Along the way, {detail_parts[0]} and {detail_parts[1]} stood out in the text."
    else:
        middle = " No strong vehicle or phone anchors showed up in this slice."

    closing = f" Suggested next move: {_rule_based_next_action(summary, timeline, ranked)}"
    return lead + middle + closing


def main() -> None:
    st.set_page_config(page_title="Intel Search", layout="wide", initial_sidebar_state="expanded")
    st.title("AI-powered intelligence search")
    st.caption("Semantic search with FAISS, OpenAI embeddings, and entity-aware analytics.")

    default_data_path = Path(__file__).resolve().parent / "data"
    data_root = Path(st.sidebar.text_input("Data folder", value=str(default_data_path)))
    use_cache = st.sidebar.toggle("Use disk cache for index", value=True)
    top_k = st.sidebar.slider("Results (FAISS top-K)", min_value=5, max_value=50, value=15)
    keyword_boost = st.sidebar.slider("Keyword boost for hybrid ranking", 0.0, 0.25, 0.12)

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Set `OPENAI_API_KEY` in a `.env` file or your environment.")
        st.stop()

    try:
        _ = get_spacy_ready()
    except Exception:
        pass
    st.sidebar.info("Demo mode: fast entity extraction enabled.")

    if not data_root.is_dir():
        st.warning(f"Data folder does not exist yet: `{data_root}`. Creating sample tree is recommended.")
        st.stop()

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

    st.sidebar.success(f"Indexed **{len(store.records)}** chunks.")
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

    qv = embed_texts([query])
    raw_hits = store.search(qv[0], k=top_k)
    ranked = hybrid_rank(raw_hits, query, keyword_boost=keyword_boost)
    summary, edges, timeline = aggregate_dashboard(ranked, query)
    st.subheader("AI Summary")
    st.write(build_ai_summary(ranked, summary, timeline, query))
    q_tokens = [t for t in query.strip().lower().split() if len(t) > 1]
    if len(q_tokens) >= 2 and not has_exact_full_name_hit(ranked, query):
        st.info(
            "No evidence snippet contains an exact full-name match for your query; "
            "ranked results may still include partial mentions or similarly named people."
        )

    res_tab, ent_tab, rel_tab, time_tab = st.tabs(
        ["Evidence", "Entity Profile", "Link Analysis", "Activity Timeline"]
    )

    with res_tab:
        st.subheader("Ranked evidence")
        for r, score in ranked:
            with st.expander(f"[{r.source_type}] {r.doc_title} — score {score:.3f}"):
                st.markdown(highlight_query(r.text[:6000], query))
                st.caption(f"`{r.chunk_id}`")

    with ent_tab:
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
        if not edges:
            st.write("No edges found for this result set.")
        else:
            gfig, gnote = build_entity_link_graph_figure(ranked, edges, query)
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
