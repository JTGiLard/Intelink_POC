"""Structured Case #2 / Case #3 linkage scaffold when Johnnie Walker is the selected subject.

Analyst-facing: verify all facts in primary source material. This module separates
direct Case #2 involvement from indirect Case #3 company/gang context.
"""

from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher

from intelligence.timeline import TimelineEvent, timeline_sort_key

try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz
except Exception:  # pragma: no cover - optional dependency
    _rapidfuzz_fuzz = None

WALKER_FUZZY_ACTIVATION_THRESHOLD = 88.0

LINK_ANALYSIS_SCAFFOLD_NOTE = (
    "This Johnnie Walker case scaffold is derived from known Case #2 and Case #3 report patterns "
    "and should be validated against source evidence."
)

# Relationship types (edge metadata / UI)
DIRECT_CASE_INVOLVEMENT = "DIRECT_CASE_INVOLVEMENT"
CO_ARRESTED_WITH = "CO_ARRESTED_WITH"
VEHICLE_USED_IN_CASE = "VEHICLE_USED_IN_CASE"
COMPANY_OWNERSHIP = "COMPANY_OWNERSHIP"
COMPANY_EMPLOYMENT = "COMPANY_EMPLOYMENT"
INDIRECT_CASE_CONTEXT = "INDIRECT_CASE_CONTEXT"

_WALKER_KEY = "johnnie walker"


def is_johnnie_walker_selected(name: str | None) -> bool:
    if not name or not str(name).strip():
        return False
    return " ".join(str(name).strip().lower().split()) == _WALKER_KEY


def _person_phrase_from_query(query: str) -> str:
    raw = query.strip()
    if not raw:
        return ""
    parts = [p.strip() for p in re.split(r"[+,]", raw) if p.strip()]
    return parts[0] if parts else raw


def _fuzzy_ratio_to_reference(query_fragment: str, reference: str = _WALKER_KEY) -> float:
    q = " ".join(query_fragment.strip().lower().split())
    ref = " ".join(reference.strip().lower().split())
    if not q or not ref:
        return 0.0
    if _rapidfuzz_fuzz is not None:
        return float(_rapidfuzz_fuzz.ratio(q, ref))
    return SequenceMatcher(None, q, ref).ratio() * 100.0


def query_matches_walker_fuzzy(query: str, threshold: float = WALKER_FUZZY_ACTIVATION_THRESHOLD) -> bool:
    phrase = _person_phrase_from_query(query)
    return _fuzzy_ratio_to_reference(phrase) >= threshold


def should_activate_walker_scaffold(
    query: str,
    selected_analysis_name: str | None,
    *,
    threshold: float = WALKER_FUZZY_ACTIVATION_THRESHOLD,
) -> bool:
    """Enable analyst scaffold only for confirmed selected name or high-confidence query spelling."""
    if is_johnnie_walker_selected(selected_analysis_name):
        return True
    return query_matches_walker_fuzzy(query, threshold)


def walker_graph_anchor_person(selected_analysis_name: str | None) -> str:
    """Person-centric graph center when the Walker scaffold is active."""
    if is_johnnie_walker_selected(selected_analysis_name):
        return (selected_analysis_name or "").strip()
    return "Johnnie Walker"


def _edge_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _short_graph_label(rel_type: str) -> str:
    return {
        DIRECT_CASE_INVOLVEMENT: "direct case (Case #2)",
        CO_ARRESTED_WITH: "Case #2 associate link",
        VEHICLE_USED_IN_CASE: "vehicle in case",
        COMPANY_OWNERSHIP: "company ownership (context)",
        COMPANY_EMPLOYMENT: "employment / works under company",
        INDIRECT_CASE_CONTEXT: "indirect / contextual link",
    }.get(rel_type, rel_type.replace("_", " ").lower())


def walker_structured_edges_and_labels() -> tuple[
    list[tuple[str, str, float, str, str]],
    dict[tuple[str, str], str],
]:
    """Return cooccurrence-style edges plus graph annotation labels.

    Tuple: (entity_a, entity_b, strength, relationship_type, plotly_link_type)
    plotly_link_type: ``direct`` = solid line, anything else = dashed.
    """
    jw = "person:Johnnie Walker"
    c2 = "case:Case #2"
    c3 = "case:Case #3"
    v_ll = "vehicle:LL010"
    v_ff = "vehicle:FF777"
    v_fg = "vehicle:FG123"
    p_arj = "person:Alamak Roti John, 31"
    p_arp = "person:Alamak Roti Prata, 28"
    co = "pseudo:Clean & Innocent"
    p_rah = "person:Rahman"
    grp = "pseudo:Rahman's group (Case #3)"
    p_i = "person:Ishak, 49"
    p_aa = "person:Ah Ali, 45"
    p_s = "person:Saberi, 48"

    edges: list[tuple[str, str, float, str, str]] = [
        (jw, c2, 12.0, DIRECT_CASE_INVOLVEMENT, "direct"),
        (jw, v_ll, 11.0, VEHICLE_USED_IN_CASE, "direct"),
        (jw, p_arj, 10.5, CO_ARRESTED_WITH, "direct"),
        (jw, p_arp, 10.5, CO_ARRESTED_WITH, "direct"),
        (c2, v_ll, 9.0, VEHICLE_USED_IN_CASE, "direct"),
        (jw, co, 8.0, INDIRECT_CASE_CONTEXT, "indirect"),
        (co, p_rah, 7.5, COMPANY_EMPLOYMENT, "indirect"),
        (p_rah, c3, 7.5, INDIRECT_CASE_CONTEXT, "indirect"),
        (p_rah, grp, 7.2, INDIRECT_CASE_CONTEXT, "indirect"),
        (grp, p_i, 6.8, INDIRECT_CASE_CONTEXT, "indirect"),
        (grp, p_aa, 6.8, INDIRECT_CASE_CONTEXT, "indirect"),
        (grp, p_s, 6.8, INDIRECT_CASE_CONTEXT, "indirect"),
        (c3, v_ff, 6.5, VEHICLE_USED_IN_CASE, "indirect"),
        (c3, v_fg, 6.5, VEHICLE_USED_IN_CASE, "indirect"),
        (co, c3, 6.2, COMPANY_OWNERSHIP, "indirect"),
    ]

    labels: dict[tuple[str, str], str] = {}
    for a, b, _s, rel_t, _lt in edges:
        labels[_edge_key(a, b)] = _short_graph_label(rel_t)

    return edges, labels


def merge_walker_case_edges(
    base_edges: list[tuple[str, str, float, str, str]],
    selected_analysis_name: str | None,
    query: str = "",
) -> tuple[list[tuple[str, str, float, str, str]], dict[tuple[str, str], str]]:
    if not should_activate_walker_scaffold(query, selected_analysis_name):
        return list(base_edges), {}
    structured, labels = walker_structured_edges_and_labels()
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, float, str, str]] = []
    for e in structured:
        seen.add(_edge_key(e[0], e[1]))
        out.append(e)
    for e in base_edges:
        if _edge_key(e[0], e[1]) in seen:
            continue
        seen.add(_edge_key(e[0], e[1]))
        out.append(e)
    out.sort(key=lambda x: -float(x[2]))
    return out, labels


def johnnie_walker_case_summary_block() -> str:
    return (
        "Case linkage (analyst scaffold — verify in primary source):\n"
        f"- **Case #2 — direct case involvement:** Johnnie Walker is treated as **directly tied to Case #2** "
        "in this scaffold (operational / subject-role context per Case #2 report patterns — validate in source). "
        f"Structured link types: `{DIRECT_CASE_INVOLVEMENT}`, `{CO_ARRESTED_WITH}`, `{VEHICLE_USED_IN_CASE}`. "
        "Illustrative direct links include vehicle **LL010**; named operational context includes "
        "**Alamak Roti John, 31** and **Alamak Roti Prata, 28**.\n"
        f"- **Case #3 — indirect / company-linked context only:** Frame Case #3 as **context**, not as Johnnie Walker "
        "being **directly involved** in Case #3. **Johnnie Walker is indirectly linked to Case #3 through "
        "Clean & Innocent and Rahman** (Rahman associated with the company; Rahman described as leader of the Case #3 gang). "
        f"Structured link types: `{INDIRECT_CASE_CONTEXT}`, `{COMPANY_EMPLOYMENT}`. "
        "`COMPANY_OWNERSHIP` may apply in source where corporate control is asserted — not assumed here.\n"
        "**Indirect / contextual (Case #3):** Company **Clean & Innocent**; **Rahman** (leader); "
        "Rahman's group: **Ishak, 49**; **Ah Ali, 45**; **Saberi, 48**; vehicles **FF777**, **FG123**. "
        f"Use `{COMPANY_OWNERSHIP}` where the source ties the company to Case #3 corporate context."
    )


def indirect_context_markdown() -> str:
    return (
        "### Indirect / contextual links (Case #3)\n"
        "**Indirect / company-linked context** — not direct Johnnie Walker involvement in Case #3.\n\n"
        "- **Company:** Clean & Innocent\n"
        "- **Rahman:** leader (Case #3 gang); employment / works under **Clean & Innocent**\n"
        "- **Rahman's group:** Ishak, 49 · Ah Ali, 45 · Saberi, 48\n"
        "- **Vehicles:** FF777 · FG123\n\n"
        f"Relationship types: `{INDIRECT_CASE_CONTEXT}`, `{COMPANY_EMPLOYMENT}` "
        f"(and `{COMPANY_OWNERSHIP}` in source where ownership is explicit)."
    )


def direct_context_markdown() -> str:
    return (
        "### Direct case involvement (Case #2)\n"
        "Johnnie Walker is shown here as **direct case involvement** for Case #2 in this scaffold (validate in source).\n\n"
        "- **Vehicle:** LL010\n"
        "- **Operational / named context (Case #2):** Alamak Roti John, 31 · Alamak Roti Prata, 28\n\n"
        f"Relationship types: `{DIRECT_CASE_INVOLVEMENT}`, `{CO_ARRESTED_WITH}`, `{VEHICLE_USED_IN_CASE}`."
    )


def supplement_walker_timeline(
    events: list[TimelineEvent],
    selected_analysis_name: str | None,
    query: str = "",
) -> list[TimelineEvent]:
    if not should_activate_walker_scaffold(query, selected_analysis_name):
        return list(events)
    extra = [
        TimelineEvent(
            datetime(2017, 5, 5, 0, 0),
            "5 May 2017 — Case #2 direct involvement",
            "Scaffold marker: direct case involvement for Case #2 — validate against source evidence.",
            "scaffold-case-2",
        ),
        TimelineEvent(
            datetime(2021, 11, 30, 0, 0),
            "30 Nov 2021 — Case #3 indirect/company-linked context",
            "Scaffold marker: indirect / company-linked context for Case #3 — not direct Johnnie Walker involvement in Case #3.",
            "scaffold-case-3-indirect",
        ),
    ]
    merged = list(events) + extra
    merged.sort(key=lambda e: timeline_sort_key(e.when))
    dedup: list[TimelineEvent] = []
    seen: set[tuple[str, str]] = set()
    for e in merged:
        key = (timeline_sort_key(e.when), e.label, e.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(e)
    return dedup
