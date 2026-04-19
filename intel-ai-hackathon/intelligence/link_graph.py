"""Small Plotly network graph for link analysis (retrieved chunks only)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import plotly.graph_objects as go

from intelligence.entities import extract_all_entities
from intelligence.index_store import ChunkRecord


MAX_GRAPH_NODES = 26
MAX_CASE_NODES = 6


def _case_node(doc_id: str) -> str:
    return f"case:{doc_id}"


def _source_node(source_type: str) -> str:
    return f"source:{source_type}"


def _entity_key(h: Any) -> str | None:
    t = h.text.strip()
    if len(t) <= 1:
        return None
    return f"{h.label}:{t}"


def _node_kind(node_id: str) -> str:
    if ":" not in node_id:
        return "other"
    prefix = node_id.split(":", 1)[0].lower()
    if prefix in ("person", "vehicle", "phone", "case", "source"):
        return prefix
    return "other"


def _short_label(node_id: str) -> str:
    kind, _, rest = node_id.partition(":")
    if kind == "case":
        return rest[:18] + ("…" if len(rest) > 18 else "")
    if kind == "source":
        return rest[:20]
    if len(rest) > 22:
        return rest[:20] + "…"
    return rest


def _marker_symbol(kind: str) -> str:
    return {
        "person": "circle",
        "vehicle": "square",
        "phone": "diamond",
        "case": "star",
        "source": "triangle-up",
        "other": "circle-open",
    }.get(kind, "circle-open")


def _marker_color(kind: str) -> str:
    return {
        "person": "#2563eb",
        "vehicle": "#059669",
        "phone": "#d97706",
        "case": "#64748b",
        "source": "#7c3aed",
        "other": "#94a3b8",
    }.get(kind, "#94a3b8")


def _fr_layout(
    node_list: list[str],
    weighted_pairs: list[tuple[str, str, float]],
    seed: int = 42,
    iterations: int = 56,
) -> dict[str, tuple[float, float]]:
    n = len(node_list)
    if n == 0:
        return {}
    if n == 1:
        return {node_list[0]: (0.0, 0.0)}
    rng = np.random.default_rng(seed)
    idx = {nid: i for i, nid in enumerate(node_list)}
    pos = rng.uniform(-0.55, 0.55, (n, 2))
    k = 1.0 / math.sqrt(float(n))
    t = 0.75
    for _ in range(iterations):
        disp = np.zeros((n, 2))
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[i] - pos[j]
                dist2 = float(np.dot(delta, delta)) + 1e-6
                dist = math.sqrt(dist2)
                force = (k * k / dist2) * delta
                disp[i] += force
                disp[j] -= force
        for u, v, w in weighted_pairs:
            if u not in idx or v not in idx:
                continue
            i, j = idx[u], idx[v]
            delta = pos[j] - pos[i]
            dist = math.sqrt(float(np.dot(delta, delta)) + 1e-9)
            attractive = (dist * dist / k) * (0.07 + 0.11 * min(float(w), 6.0)) * (delta / dist)
            disp[i] += attractive
            disp[j] -= attractive
        pos += t * disp
        t *= 0.965
    pos -= pos.mean(axis=0)
    mx = float(np.max(np.abs(pos)) + 1e-9)
    pos /= mx
    return {node_list[i]: (float(pos[i, 0]), float(pos[i, 1])) for i in range(n)}


def _merge_edge_weights(pairs: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    merged: dict[tuple[str, str], float] = {}
    for u, v, w in pairs:
        a, b = (u, v) if u < v else (v, u)
        merged[(a, b)] = max(merged.get((a, b), 0.0), float(w))
    return [(a, b, w) for (a, b), w in merged.items()]


def build_entity_link_graph_figure(
    ranked: list[tuple[ChunkRecord, float]],
    edges: list[tuple[str, str, float, str, str]],
) -> tuple[go.Figure | None, str]:
    """
    Build a spider-web graph from weighted co-occurrence edges plus light case/source context.

    Returns (figure, skip_reason). skip_reason is empty when a figure is returned.
    """
    if not edges or not ranked:
        return None, ""

    edge_endpoints: set[str] = set()
    for a, b, *_ in edges:
        edge_endpoints.add(a)
        edge_endpoints.add(b)

    def _chunk_entity_keys(record: ChunkRecord) -> set[str]:
        out: set[str] = set()
        for h in extract_all_entities(record.text):
            ek = _entity_key(h)
            if ek:
                out.add(ek)
        return out

    def _touches_endpoints(record: ChunkRecord) -> bool:
        return bool(_chunk_entity_keys(record) & edge_endpoints)

    case_doc_ids: list[str] = []
    seen_doc: set[str] = set()
    for r, _ in ranked:
        if r.doc_id in seen_doc or not _touches_endpoints(r):
            continue
        seen_doc.add(r.doc_id)
        case_doc_ids.append(r.doc_id)
        if len(case_doc_ids) >= MAX_CASE_NODES:
            break

    source_types = sorted(
        {
            r.source_type
            for r, _ in ranked
            if r.source_type and _touches_endpoints(r)
        }
    )

    nodes: set[str] = set(edge_endpoints)
    case_hover: dict[str, str] = {}
    for did in case_doc_ids:
        nid = _case_node(did)
        nodes.add(nid)
        title = next((r.doc_title for r, _ in ranked if r.doc_id == did), did)
        case_hover[nid] = f"{title} ({did})"
    for st in source_types:
        nodes.add(_source_node(st))

    if len(nodes) > MAX_GRAPH_NODES:
        return (
            None,
            f"Graph hidden: too many nodes for a readable view ({len(nodes)} > {MAX_GRAPH_NODES}). "
            "Try a narrower search or fewer results.",
        )

    layout_weights: list[tuple[str, str, float]] = []
    for a, b, s, _, _ in edges:
        if a in nodes and b in nodes:
            layout_weights.append((a, b, float(s)))
    for r, _ in ranked:
        if r.doc_id not in case_doc_ids:
            continue
        cid = _case_node(r.doc_id)
        sid = _source_node(r.source_type)
        for h in extract_all_entities(r.text):
            ek = _entity_key(h)
            if not ek or ek not in nodes:
                continue
            if cid in nodes:
                layout_weights.append((ek, cid, 0.22))
            if sid in nodes:
                layout_weights.append((ek, sid, 0.16))

    node_list = sorted(nodes)
    pos = _fr_layout(node_list, _merge_edge_weights(layout_weights))

    smax = max((s for _, _, s, _, _ in edges), default=1.0)
    if smax <= 0:
        smax = 1.0

    fig = go.Figure()

    # Context edges (case / source): thin, neutral (deduped)
    ax_c, ay_c = [], []
    drawn_ctx: set[tuple[str, str]] = set()
    for r, _ in ranked:
        if r.doc_id not in case_doc_ids:
            continue
        cid = _case_node(r.doc_id)
        sid = _source_node(r.source_type)
        for h in extract_all_entities(r.text):
            ek = _entity_key(h)
            if not ek or ek not in pos:
                continue
            x0, y0 = pos[ek]
            if cid in pos:
                key = tuple(sorted((ek, cid)))
                if key not in drawn_ctx:
                    drawn_ctx.add(key)
                    x1, y1 = pos[cid]
                    ax_c.extend([x0, x1, None])
                    ay_c.extend([y0, y1, None])
            if sid in pos:
                key2 = tuple(sorted((ek, sid)))
                if key2 not in drawn_ctx:
                    drawn_ctx.add(key2)
                    x2, y2 = pos[sid]
                    ax_c.extend([x0, x2, None])
                    ay_c.extend([y0, y2, None])
    if ax_c:
        fig.add_trace(
            go.Scatter(
                x=ax_c,
                y=ay_c,
                mode="lines",
                line=dict(color="rgba(148,163,184,0.55)", width=1),
                hoverinfo="skip",
                showlegend=True,
                name="Context (doc / source)",
            )
        )

    # Co-occurrence edges: width from strength, dash from link type
    buckets: dict[tuple[str, int], tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for a, b, s, _lbl, lt in edges:
        if a not in pos or b not in pos:
            continue
        dash = "solid" if lt == "direct" else "dash"
        r = s / smax
        if r >= 0.66:
            wb = 2
        elif r >= 0.33:
            wb = 1
        else:
            wb = 0
        key = (dash, wb)
        xl, yl = buckets[key]
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        xl.extend([x0, x1, None])
        yl.extend([y0, y1, None])

    for (dash, wb), (xl, yl) in buckets.items():
        if not xl:
            continue
        w = 2.0 + wb * 1.35
        kind = "direct" if dash == "solid" else "indirect"
        fig.add_trace(
            go.Scatter(
                x=xl,
                y=yl,
                mode="lines",
                line=dict(color="rgba(30,64,175,0.78)", width=w, dash=dash),
                hoverinfo="skip",
                showlegend=True,
                name=f"Co-occurrence ({kind}, weight {wb + 1})",
            )
        )

    kinds = ("person", "vehicle", "phone", "case", "source", "other")
    for kind in kinds:
        ns = [n for n in node_list if _node_kind(n) == kind]
        if not ns:
            continue
        hover = []
        for n in ns:
            if n.startswith("case:"):
                hover.append(case_hover.get(n, n))
            elif n.startswith("source:"):
                hover.append(f"Source channel: {n.split(':', 1)[1]}")
            else:
                hover.append(n)
        fig.add_trace(
            go.Scatter(
                x=[pos[n][0] for n in ns],
                y=[pos[n][1] for n in ns],
                mode="markers+text",
                name=kind,
                text=[_short_label(n) for n in ns],
                textposition="top center",
                textfont=dict(size=10, color="#1e293b"),
                marker=dict(
                    symbol=_marker_symbol(kind),
                    size=16 if kind != "case" else 18,
                    color=_marker_color(kind),
                    line=dict(width=1, color="#0f172a"),
                ),
                hovertext=hover,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=dict(text="Relationship graph (current retrieval only)", font=dict(size=14)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, x=0),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        plot_bgcolor="#f8fafc",
        margin=dict(l=8, r=8, t=40, b=80),
        height=520,
    )
    return fig, ""
