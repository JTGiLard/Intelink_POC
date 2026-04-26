"""Plotly relationship graph for link analysis (current retrieval only)."""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from typing import Any

import numpy as np
import plotly.graph_objects as go

from intelligence.index_store import ChunkRecord


GRAPH_TOP_EDGES = 14
MAX_TOTAL_NODES = 16
MAX_CASE_NODES = 4
EDGE_LABEL_MAX = 6
INNER_RING_R = 1.05
OUTER_RING_R = 1.78

SIMPLIFIED_CAPTION = (
    "Radial layout (subject center, grouped by type). "
    f"Relationship phrases on up to {EDGE_LABEL_MAX} strongest direct links only."
)
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


def _vehicle_display_label(node_id: str, plate_to_model: dict[str, str]) -> str:
    """Render vehicle nodes as plate-first labels with optional model on second line."""
    if not node_id.lower().startswith("vehicle:"):
        return _short_label(node_id)
    value = node_id.split(":", 1)[1].strip()
    model = plate_to_model.get(value, "").strip()
    if model:
        return f"{value}\n{model}"
    return value


def _looks_like_plate_token(raw: str) -> bool:
    t = raw.strip().upper().replace(" ", "")
    if len(t) < 5:
        return False
    return bool(re.search(r"[A-Z]+\d+[A-Z]*", t))


def _looks_like_vehicle_model(raw: str) -> bool:
    t = raw.strip()
    if len(t) < 4:
        return False
    words = [w for w in t.split() if w]
    if len(words) < 2:
        return False
    if any(any(ch.isdigit() for ch in w) for w in words):
        return False
    return all(w[:1].isalpha() for w in words)


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


def _normalize_entity_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _canonicalize_entity_node(node_id: str, vehicle_texts: set[str]) -> str:
    kind, sep, rest = node_id.partition(":")
    if not sep:
        return node_id
    if _normalize_entity_text(rest) in vehicle_texts:
        return f"vehicle:{rest.strip()}"
    return f"{kind}:{rest.strip()}"


def _relationship_edge_label(node_a: str, node_b: str) -> str:
    kinds = {_node_kind(node_a), _node_kind(node_b)}
    if kinds == {"person", "phone"}:
        return "uses phone"
    if kinds == {"person", "vehicle"}:
        return "drives vehicle"
    if kinds == {"person"}:
        return "associated with"
    if "person" in kinds and kinds <= {"person", "pseudo", "vehicle", "phone"}:
        return "linked to"
    return "linked to"


def _collapse_vehicle_model_nodes(
    edges: list[tuple[str, str, float, str, str]],
) -> tuple[list[tuple[str, str, float, str, str]], dict[str, str], set[str]]:
    """Attach models to plate nodes and remove duplicate model nodes where possible."""
    model_to_plate_best: dict[str, tuple[str, float]] = {}
    for a, b, strength, _lbl, link_type in edges:
        if not (a.startswith("vehicle:") and b.startswith("vehicle:")):
            continue
        av = a.split(":", 1)[1].strip()
        bv = b.split(":", 1)[1].strip()
        a_plate, b_plate = _looks_like_plate_token(av), _looks_like_plate_token(bv)
        a_model, b_model = _looks_like_vehicle_model(av), _looks_like_vehicle_model(bv)
        if a_plate and b_model and (link_type == "direct" or strength >= 0.6):
            prev = model_to_plate_best.get(bv)
            if prev is None or strength > prev[1]:
                model_to_plate_best[bv] = (av, strength)
        elif b_plate and a_model and (link_type == "direct" or strength >= 0.6):
            prev = model_to_plate_best.get(av)
            if prev is None or strength > prev[1]:
                model_to_plate_best[av] = (bv, strength)

    plate_to_model: dict[str, str] = {}
    for model, (plate, _s) in sorted(model_to_plate_best.items(), key=lambda x: -x[1][1]):
        if plate not in plate_to_model:
            plate_to_model[plate] = model

    suppressed_models = set(model_to_plate_best.keys())
    remapped: list[tuple[str, str, float, str, str]] = []
    for a, b, strength, lbl, link_type in edges:
        ra = f"vehicle:{model_to_plate_best[a.split(':', 1)[1].strip()][0]}" if a.startswith("vehicle:") and a.split(":", 1)[1].strip() in suppressed_models else a
        rb = f"vehicle:{model_to_plate_best[b.split(':', 1)[1].strip()][0]}" if b.startswith("vehicle:") and b.split(":", 1)[1].strip() in suppressed_models else b
        if ra == rb:
            continue
        remapped.append((ra, rb, strength, lbl, link_type))
    return remapped, plate_to_model, suppressed_models


def _type_group_rank(kind: str) -> int:
    order = ("person", "pseudo", "vehicle", "phone", "case", "source", "other")
    try:
        return order.index(kind)
    except ValueError:
        return len(order)


def _bfs_distance_from(anchor: str, adj: dict[str, set[str]], nodes: set[str]) -> dict[str, int]:
    if anchor not in nodes:
        return {n: 99 for n in nodes}
    dist: dict[str, int] = {anchor: 0}
    q: deque[str] = deque([anchor])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in nodes or v in dist:
                continue
            dist[v] = dist[u] + 1
            q.append(v)
    return {n: dist.get(n, 99) for n in nodes}


def _circular_mean_angle(radians_list: list[float]) -> float:
    if not radians_list:
        return 0.0
    s = sum(math.sin(t) for t in radians_list)
    c = sum(math.cos(t) for t in radians_list)
    return math.atan2(s, c) if (s != 0.0 or c != 0.0) else 0.0


def _angle_of(xy: tuple[float, float]) -> float:
    return math.atan2(xy[1], xy[0])


def _radial_grouped_ring_layout(
    node_list: list[str],
    reduced_edges: list[tuple[str, str, float, str, str]],
    anchor: str,
) -> dict[str, tuple[float, float]]:
    """Deterministic radial layout: anchor center, grouped inner ring, farther hops on outer rings."""
    nodes = set(node_list)
    if anchor not in nodes:
        anchor = sorted(nodes)[0]
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b, *_rest in reduced_edges:
        if a in nodes and b in nodes:
            adj[a].add(b)
            adj[b].add(a)

    pos: dict[str, tuple[float, float]] = {anchor: (0.0, 0.0)}
    direct = sorted(n for n in adj[anchor] if n in nodes and n != anchor)
    groups: dict[str, list[str]] = defaultdict(list)
    for nid in direct:
        groups[_node_kind(nid)].append(nid)
    for k in groups:
        groups[k].sort()

    nonempty_kinds = sorted([k for k, v in groups.items() if v], key=_type_group_rank)
    n_sectors = max(1, len(nonempty_kinds))
    sector = (2.0 * math.pi) / float(n_sectors)
    for si, kind in enumerate(nonempty_kinds):
        members = groups[kind]
        m = len(members)
        base = si * sector
        members_ranked = sorted(
            members,
            key=lambda nid: (
                -sum(e[2] for e in reduced_edges if (e[0] == anchor and e[1] == nid) or (e[1] == anchor and e[0] == nid)),
                nid,
            ),
        )
        for j, nid in enumerate(members_ranked):
            t = base + (j + 1.0) * (sector / float(m + 1))
            pos[nid] = (INNER_RING_R * math.cos(t), INNER_RING_R * math.sin(t))

    dist = _bfs_distance_from(anchor, adj, nodes)
    farther = sorted([n for n in nodes if n not in pos], key=lambda x: (dist.get(x, 99), x))
    ring_dr = 0.58

    for idx, nid in enumerate(farther):
        d_raw = dist.get(nid, 99)
        nbr_angles = [_angle_of(pos[nb]) for nb in adj[nid] if nb in pos]
        if nbr_angles:
            theta = _circular_mean_angle(nbr_angles)
        else:
            theta = 2.0 * math.pi * ((hash(nid) % 10007) / 10007.0)
        jitter = ((idx % 11) - 5) * 0.085
        theta2 = theta + jitter
        if d_raw >= 99:
            r = OUTER_RING_R + ring_dr * 2.4
        else:
            d_use = min(d_raw, 6)
            r = OUTER_RING_R + ring_dr * max(0, d_use - 2)
        pos[nid] = (r * math.cos(theta2), r * math.sin(theta2))

    return pos


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


def build_entity_link_graph_figure(
    ranked: list[tuple[ChunkRecord, float]],
    edges: list[tuple[str, str, float, str, str]],
    query: str = "",
    person_centric: bool = False,
    anchor_person: str = "",
) -> tuple[go.Figure | None, str]:
    if not edges:
        return None, ""

    try:
        vehicle_texts = {
            _normalize_entity_text(n.split(":", 1)[1])
            for e in edges
            for n in (e[0], e[1])
            if n.lower().startswith("vehicle:")
        }
        canonical_edges: list[tuple[str, str, float, str, str]] = []
        for a, b, strength, lbl, link_type in edges:
            ca = _canonicalize_entity_node(a, vehicle_texts)
            cb = _canonicalize_entity_node(b, vehicle_texts)
            if ca == cb:
                continue
            canonical_edges.append((ca, cb, strength, lbl, link_type))
        canonical_edges, plate_to_model, _suppressed_models = _collapse_vehicle_model_nodes(canonical_edges)
        canonical_edges = _dedupe_edges_keep_max_strength(canonical_edges)
        reduced_edges = sorted(canonical_edges, key=lambda e: -float(e[2]))[:GRAPH_TOP_EDGES]
        nodes: set[str] = {a for a, _, _, _, _ in reduced_edges} | {b for _, b, _, _, _ in reduced_edges}
        if not nodes:
            return None, GRAPH_FAIL_MSG

        node_list = sorted(nodes)
        anchor_id: str | None = None
        if person_centric:
            preferred_name = anchor_person.strip() or _first_query_segment(query)
            if preferred_name:
                anchor_id = _find_person_node_case_insensitive(set(node_list), preferred_name)
        if anchor_id is None:
            anchor_id = _pick_anchor_node(set(node_list), reduced_edges, None)
        if anchor_id is None or anchor_id not in node_list:
            anchor_id = node_list[0]

        pos = _radial_grouped_ring_layout(node_list, reduced_edges, anchor_id)
        pos = _recenter_positions(pos, anchor_id)
        pos = _sanitize_and_scale_positions(pos)
        label_map = {nid: _vehicle_display_label(nid, plate_to_model) for nid in node_list}
        color_map = {nid: _marker_color(_node_kind(nid)) for nid in node_list}

        max_strength = max((float(e[2]) for e in reduced_edges), default=1.0)

        fig = go.Figure()
        edge_trace_count = 0
        for a, b, strength, _lbl, link_type in reduced_edges:
            if a not in pos or b not in pos:
                continue
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            dash = "solid" if link_type == "direct" else "dash"
            norm = float(strength) / max_strength if max_strength > 0 else 0.0
            width = 2.0 + norm * 6.5
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

        node_x = [pos[nid][0] for nid in node_list]
        node_y = [pos[nid][1] for nid in node_list]
        text_positions: list[str] = []
        for nid in node_list:
            if nid == anchor_id:
                text_positions.append("top center")
                continue
            x, y = pos[nid]
            if abs(x) > abs(y):
                text_positions.append("middle right" if x >= 0 else "middle left")
            else:
                text_positions.append("top center" if y >= 0 else "bottom center")
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(
                    size=20,
                    color=[color_map[nid] for nid in node_list],
                    line=dict(width=1, color="#0f172a"),
                    symbol=[_marker_symbol(_node_kind(nid)) for nid in node_list],
                ),
                text=[label_map[nid] for nid in node_list],
                textposition=text_positions,
                textfont=dict(size=10, color="#0f172a"),
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
                    marker=dict(
                        size=20,
                        color=[color_map[nid] for nid in node_list],
                        line=dict(width=1, color="#0f172a"),
                        symbol=[_marker_symbol(_node_kind(nid)) for nid in node_list],
                    ),
                    text=[label_map[nid] for nid in node_list],
                    textposition="top center",
                    textfont=dict(size=11, color="#0f172a"),
                    hoverinfo="text",
                    hovertext=node_list,
                )
            )

        label_edges = [e for e in reduced_edges if e[4] == "direct"]
        label_edges.sort(key=lambda e: -float(e[2]))
        ann: list[dict[str, Any]] = []
        for a, b, _strength, _lbl, _lt in label_edges[:EDGE_LABEL_MAX]:
            if a not in pos or b not in pos:
                continue
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            dx, dy = x1 - x0, y1 - y0
            ln = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / ln, dx / ln
            off = 0.09
            ox, oy = mx + nx * off, my + ny * off
            textangle = math.degrees(math.atan2(dy, dx))
            if textangle > 90:
                textangle -= 180.0
            if textangle < -90:
                textangle += 180.0
            ann.append(
                {
                    "xref": "x",
                    "yref": "y",
                    "x": ox,
                    "y": oy,
                    "text": _relationship_edge_label(a, b),
                    "showarrow": False,
                    "font": {"size": 10, "color": "#334155"},
                    "textangle": textangle,
                }
            )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=11, color="#2563eb", symbol="circle", line=dict(width=1, color="#0f172a")),
                name="Person",
                showlegend=True,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=11, color="#059669", symbol="square", line=dict(width=1, color="#0f172a")),
                name="Vehicle (plate)",
                showlegend=True,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=11, color="#d97706", symbol="diamond", line=dict(width=1, color="#0f172a")),
                name="Phone",
                showlegend=True,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None, None],
                y=[None, None],
                mode="lines",
                line=dict(color="#0f172a", width=5),
                name="Thicker line = stronger relationship",
                showlegend=True,
                hoverinfo="skip",
            )
        )

        fig.update_layout(
            title=dict(text="Relationship graph (current retrieval only)", font=dict(size=14, color="#0f172a")),
            xaxis=dict(range=[-2, 2], visible=False),
            yaxis=dict(range=[-2, 2], visible=False),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.02,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(248,250,252,0.92)",
            ),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#f8fafc",
            margin=dict(l=20, r=20, t=48, b=20),
            height=520,
            annotations=ann,
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
