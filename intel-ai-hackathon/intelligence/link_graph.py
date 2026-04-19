"""Plotly relationship graph for link analysis (current retrieval only)."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import numpy as np
import plotly.graph_objects as go

from intelligence.entities import extract_all_entities
from intelligence.index_store import ChunkRecord


GRAPH_TOP_EDGES = 12
MAX_TOTAL_NODES = 16
MAX_CASE_NODES = 4

SIMPLIFIED_CAPTION = "Showing simplified relationship graph for readability."
GRAPH_FAIL_MSG = "Graph could not be rendered from current extracted entities."


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
    if prefix in ("person", "vehicle", "phone", "case", "source", "pseudo"):
        return prefix
    return "other"


def _short_label(node_id: str) -> str:
    kind, _, rest = node_id.partition(":")
    if kind == "case":
        return rest[:12] + ("…" if len(rest) > 12 else "")
    if kind == "source":
        return rest[:12]
    if kind == "pseudo":
        t = rest.strip()
        disp = " ".join(w[:1].upper() + w[1:].lower() if w else w for w in t.split())
        return (disp[:14] + "…") if len(disp) > 14 else disp
    if len(rest) > 14:
        return rest[:12] + "…"
    return rest


def _marker_symbol(kind: str) -> str:
    return {
        "person": "circle",
        "pseudo": "circle",
        "vehicle": "square",
        "phone": "diamond",
        "case": "star",
        "source": "triangle-up",
        "other": "circle-open",
    }.get(kind, "circle-open")


def _marker_color(kind: str) -> str:
    return {
        "person": "#2563eb",
        "pseudo": "#1d4ed8",
        "vehicle": "#059669",
        "phone": "#d97706",
        "case": "#64748b",
        "source": "#7c3aed",
        "other": "#94a3b8",
    }.get(kind, "#94a3b8")


def _first_query_segment(query: str) -> str:
    raw = query.strip()
    if not raw:
        return ""
    parts = [p.strip() for p in re.split(r"[+,]", raw) if p.strip()]
    return parts[0] if parts else raw


def _looks_like_person_name(s: str) -> bool:
    s = s.strip()
    if len(s) < 3:
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z\s.'-]*[A-Za-z]$", s)) and len(s.split()) >= 2


def _dedupe_edges_keep_max_strength(
    edges: list[tuple[str, str, float, str, str]],
) -> list[tuple[str, str, float, str, str]]:
    best: dict[tuple[str, str], tuple[str, str, float, str, str]] = {}
    for e in edges:
        a, b, s, _lbl, _lt = e
        k = (a, b) if a < b else (b, a)
        if k not in best or s > best[k][2]:
            best[k] = e
    return list(best.values())


def _largest_connected_subgraph(
    edges: list[tuple[str, str, float, str, str]],
) -> list[tuple[str, str, float, str, str]]:
    if not edges:
        return []
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, *_ in edges:
        union(a, b)
    groups: dict[str, list[tuple[str, str, float, str, str]]] = defaultdict(list)
    for e in edges:
        groups[find(e[0])].append(e)
    best: list[tuple[str, str, float, str, str]] = []
    best_score = -1.0
    for comp in groups.values():
        sc = sum(c[2] for c in comp)
        if sc > best_score or (sc == best_score and len(comp) > len(best)):
            best_score = sc
            best = comp
    return best


def _entity_nodes_from_edges(edges: list[tuple[str, str, float, str, str]]) -> set[str]:
    out: set[str] = set()
    for a, b, *_ in edges:
        out.add(a)
        out.add(b)
    return out


def _degree_in_edges(node: str, edges: list[tuple[str, str, float, str, str]]) -> int:
    return sum(1 for a, b, *_ in edges if a == node or b == node)


def _find_person_node_case_insensitive(entity_nodes: set[str], person_candidate: str) -> str | None:
    pl = person_candidate.strip().lower()
    if not pl:
        return None
    for n in entity_nodes:
        if not n.lower().startswith("person:"):
            continue
        rest = n.split(":", 1)[1].strip().lower()
        if rest == pl:
            return n
    return None


def _pick_anchor_node(
    entity_nodes: set[str],
    reduced_edges: list[tuple[str, str, float, str, str]],
    pseudo_id: str | None,
) -> str | None:
    if pseudo_id and pseudo_id in entity_nodes:
        return pseudo_id
    persons = [n for n in entity_nodes if _node_kind(n) == "person"]
    if persons:
        return max(persons, key=lambda n: _degree_in_edges(n, reduced_edges))
    if not reduced_edges:
        return next(iter(entity_nodes), None)
    top = max(reduced_edges, key=lambda e: e[2])
    a, b = top[0], top[1]
    da, db = _degree_in_edges(a, reduced_edges), _degree_in_edges(b, reduced_edges)
    return a if da >= db else b


def _recenter_positions(
    pos: dict[str, tuple[float, float]],
    anchor_id: str | None,
    anchor_b: str | None = None,
) -> dict[str, tuple[float, float]]:
    if anchor_id is None or anchor_id not in pos:
        return pos
    if anchor_b and anchor_b in pos:
        ox = (pos[anchor_id][0] + pos[anchor_b][0]) / 2.0
        oy = (pos[anchor_id][1] + pos[anchor_b][1]) / 2.0
    else:
        ox, oy = pos[anchor_id]
    return {nid: (float(x - ox), float(y - oy)) for nid, (x, y) in pos.items()}


def _sanitize_and_scale_positions(
    pos: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    if not pos:
        return {}
    out: dict[str, tuple[float, float]] = {}
    xs: list[float] = []
    ys: list[float] = []
    for nid, (x, y) in pos.items():
        xf = float(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        yf = float(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0))
        out[nid] = (xf, yf)
        xs.append(xf)
        ys.append(yf)
    if not xs:
        return out
    mx = max(abs(v) for v in xs) or 1e-6
    my = max(abs(v) for v in ys) or 1e-6
    m = max(mx, my, 1e-6)
    scaled = {k: (v[0] / m, v[1] / m) for k, v in out.items()}
    return scaled


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
    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
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
    query: str = "",
) -> tuple[go.Figure | None, str]:
    if not edges:
        return None, ""

    try:
        reduced_edges = sorted(edges, key=lambda e: -float(e[2]))[:GRAPH_TOP_EDGES]
        nodes: set[str] = {a for a, _, _, _, _ in reduced_edges} | {b for _, b, _, _, _ in reduced_edges}
        if not nodes:
            return None, GRAPH_FAIL_MSG

        node_list = sorted(nodes)
        n = len(node_list)
        if n <= 12:
            angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            pos_arr = np.column_stack((np.cos(angles), np.sin(angles)))
        else:
            rng = np.random.default_rng(42)
            pos_arr = rng.uniform(-1.2, 1.2, size=(n, 2))
        pos_arr = np.nan_to_num(pos_arr, nan=0.0, posinf=0.0, neginf=0.0)
        pos = {node_list[i]: (float(pos_arr[i, 0]), float(pos_arr[i, 1])) for i in range(n)}

        fig = go.Figure()
        edge_trace_count = 0
        for a, b, strength, _lbl, link_type in reduced_edges:
            if a not in pos or b not in pos:
                continue
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            dash = "solid" if link_type == "direct" else "dash"
            width = max(2.0, float(strength) * 2.0)
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color="#0f172a", width=width, dash=dash),
                    hoverinfo="skip",
                )
            )
            edge_trace_count += 1

        fig.add_trace(
            go.Scatter(
                x=[pos[nid][0] for nid in node_list],
                y=[pos[nid][1] for nid in node_list],
                mode="markers+text",
                marker=dict(size=20, color="#2563eb", line=dict(width=1, color="#0f172a")),
                text=node_list,
                textposition="top center",
                textfont=dict(size=11, color="#0f172a"),
                hoverinfo="text",
                hovertext=node_list,
            )
        )

        # Fallback: if edge traces failed to draw, keep a nodes-only graph visible.
        if edge_trace_count == 0 and reduced_edges:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[pos[nid][0] for nid in node_list],
                    y=[pos[nid][1] for nid in node_list],
                    mode="markers+text",
                    marker=dict(size=20, color="#2563eb", line=dict(width=1, color="#0f172a")),
                    text=node_list,
                    textposition="top center",
                    textfont=dict(size=11, color="#0f172a"),
                    hoverinfo="text",
                    hovertext=node_list,
                )
            )

        fig.update_layout(
            title=dict(text="Relationship graph (current retrieval only)", font=dict(size=14, color="#0f172a")),
            xaxis=dict(range=[-2, 2], visible=False),
            yaxis=dict(range=[-2, 2], visible=False),
            showlegend=False,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#f8fafc",
            margin=dict(l=20, r=20, t=48, b=20),
            height=520,
        )
        return fig, SIMPLIFIED_CAPTION
    except Exception:
        # Last-resort fallback: render nodes from raw edges.
        fallback_nodes: set[str] = {a for a, _, _, _, _ in edges} | {b for _, b, _, _, _ in edges}
        if not fallback_nodes:
            return None, GRAPH_FAIL_MSG
        node_list = sorted(fallback_nodes)
        n = len(node_list)
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) if n > 1 else np.array([0.0])
        pos_arr = np.column_stack((np.cos(angles), np.sin(angles))) if n > 1 else np.array([[0.0, 0.0]])
        pos_arr = np.nan_to_num(pos_arr, nan=0.0, posinf=0.0, neginf=0.0)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[float(pos_arr[i, 0]) for i in range(n)],
                y=[float(pos_arr[i, 1]) for i in range(n)],
                mode="markers+text",
                marker=dict(size=20, color="#2563eb", line=dict(width=1, color="#0f172a")),
                text=node_list,
                textposition="top center",
                textfont=dict(size=11, color="#0f172a"),
                hoverinfo="text",
                hovertext=node_list,
            )
        )
        fig.update_layout(
            title=dict(text="Relationship graph (current retrieval only)", font=dict(size=14, color="#0f172a")),
            xaxis=dict(range=[-2, 2], visible=False),
            yaxis=dict(range=[-2, 2], visible=False),
            showlegend=False,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#f8fafc",
            margin=dict(l=20, r=20, t=48, b=20),
            height=520,
        )
        return fig, SIMPLIFIED_CAPTION
