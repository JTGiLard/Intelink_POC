from __future__ import annotations

import os
import re
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
from intelligence.loaders import iter_documents
from intelligence.timeline import TimelineEvent, extract_timeline_events, timeline_sort_key


load_dotenv()


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


def hybrid_rank(
    hits: list[tuple[ChunkRecord, float]],
    query: str,
    keyword_boost: float = 0.12,
) -> list[tuple[ChunkRecord, float]]:
    q = query.strip().lower()
    if not q:
        return hits
    rescored: list[tuple[ChunkRecord, float]] = []
    for r, s in hits:
        text_l = r.text.lower()
        bonus = keyword_boost if q in text_l else 0.0
        if len(q) > 3:
            parts = [p for p in q.split() if len(p) > 2]
            if parts and all(p in text_l for p in parts):
                bonus = max(bonus, keyword_boost * 0.85)
        rescored.append((r, s + bonus))
    rescored.sort(key=lambda x: -x[1])
    return rescored


def aggregate_dashboard(
    ranked: list[tuple[ChunkRecord, float]],
    query: str,
) -> tuple[EntitySummary, list[tuple[str, str, int]], list[TimelineEvent]]:
    texts_entities: list[tuple[str, list]] = []
    all_hits = []
    timeline: list[TimelineEvent] = []
    for r, _ in ranked:
        ents = extract_all_entities(r.text)
        texts_entities.append((r.text, ents))
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


def _format_list(values: list[str], empty: str, limit: int = 3) -> str:
    if not values:
        return empty
    shown = values[:limit]
    suffix = " and others" if len(values) > limit else ""
    if len(shown) == 1:
        return f"{shown[0]}{suffix}"
    if len(shown) == 2:
        return f"{shown[0]} and {shown[1]}{suffix}"
    return f"{', '.join(shown[:-1])}, and {shown[-1]}{suffix}"


def _source_label(source_type: str, count: int) -> str:
    if source_type == "whatsapp":
        return "WhatsApp"
    if source_type == "report":
        return "a report" if count == 1 else "reports"
    if source_type == "email":
        return "an email" if count == 1 else "emails"
    return source_type


def _format_sources_plain(source_types: list[str]) -> str:
    if not source_types:
        return ""
    counts: dict[str, int] = {}
    for s in source_types:
        counts[s] = counts.get(s, 0) + 1
    ordered = list(dict.fromkeys(source_types))
    labels = [_source_label(s, counts.get(s, 0)) for s in ordered]
    return _format_list(labels, "", limit=3)


def build_ai_summary(
    ranked: list[tuple[ChunkRecord, float]], summary: EntitySummary, timeline: list[TimelineEvent], query: str
) -> str:
    _ = timeline
    if not ranked:
        return "No records were returned for this query."

    main_person = summary.persons.most_common(1)
    subject = ""
    if main_person and main_person[0][1] >= 2:
        subject = main_person[0][0]
    elif query.strip():
        subject = query.strip()

    source_types = [r.source_type for r, _ in ranked]
    vehicles = [name for name, _ in summary.vehicles.most_common(2)]
    phones = [name for name, _ in summary.phones.most_common(2)]

    sentences: list[str] = []
    sources_text = _format_sources_plain(source_types)
    if subject:
        if sources_text:
            sentences.append(f"{subject} appears across {len(ranked)} records from {sources_text}.")
        else:
            sentences.append(f"{subject} appears across {len(ranked)} records.")
    elif sources_text:
        sentences.append(f"{len(ranked)} records were returned from {sources_text}.")
    else:
        sentences.append(f"{len(ranked)} records were returned.")

    if vehicles:
        sentences.append(f"Linked vehicles include {_format_list(vehicles, '', limit=2)}.")
    if phones:
        sentences.append(f"Associated phone numbers include {_format_list(phones, '', limit=2)}.")

    sentences.append("Recommended next step: review linked entities and gather corroborating records.")
    return " ".join(sentences)


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

    spacy_status = "fallback"
    try:
        spacy_status = get_spacy_ready()
    except Exception:
        spacy_status = "fallback"
    _ = spacy_status
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

    res_tab, ent_tab, rel_tab, time_tab = st.tabs(
        ["Search results", "Entity summary", "Relationship links", "Timeline"]
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
        st.caption("Pairs that appear together in the same text segment; stronger links mean more joint mentions.")
        if not edges:
            st.write("No edges found for this result set.")
        else:
            df_e = pd.DataFrame(edges, columns=["Entity A", "Entity B", "Weight"])
            st.dataframe(df_e, use_container_width=True, height=420)
            st.subheader("Strongest links")
            for a, b, w in edges[:12]:
                st.write(f"- **{a}** ↔ **{b}** (weight {w})")

    with time_tab:
        st.subheader("Chronology from retrieved content")
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
