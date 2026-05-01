"""Person-name identity clustering from retrieved evidence (disambiguation, not merge-by-name)."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from intelligence.entities import extract_all_entities, normalize_phone_number


def _vehicle_token_norm(value: str) -> str:
    return re.sub(r"\s+", "", value.strip().upper())


def extract_linking_identifiers(text: str) -> set[str]:
    """
    Strong keys used to link name mentions (excludes person entities — never merge on name alone).
    Covers phone, vehicle/plate, NRIC/FIN-style IDs, passport-like tokens, case ID, company, address.
    """
    ids: set[str] = set()
    for h in extract_all_entities(text):
        value = h.text.strip()
        if not value:
            continue
        if h.label == "phone":
            n = normalize_phone_number(value)
            if n:
                ids.add(f"phone:{n}")
        elif h.label == "vehicle":
            vn = _vehicle_token_norm(value)
            if vn:
                ids.add(f"vehicle:{vn}")
    for m in re.finditer(r"\b[STFG]\d{7}[A-Z]\b", text, flags=re.IGNORECASE):
        ids.add(f"nric:{m.group(0).upper()}")
    for m in re.finditer(r"\bG\d{7}[A-Z]\b", text, flags=re.IGNORECASE):
        ids.add(f"fin:{m.group(0).upper()}")
    for m in re.finditer(r"\b[A-Z]\d{8}\b", text):
        ids.add(f"passport:{m.group(0).upper()}")
    for m in re.finditer(r"\b(?:CASE|CID|CR|INV)[-\s]?\d{2,}\b", text, flags=re.IGNORECASE):
        ids.add(f"case:{re.sub(r'\\s+', '', m.group(0).upper())}")
    company_rx = re.compile(
        r"\b([A-Z][A-Za-z0-9&'()\-]*(?:\s+[A-Z][A-Za-z0-9&'()\-]*){0,5}\s+(?:Pte\.?\s+Ltd|Ltd|LLP|Inc\.?|Corp\.?))\b"
    )
    for m in company_rx.finditer(text):
        ids.add(f"company:{' '.join(m.group(1).lower().split())}")
    street_rx = re.compile(
        r"(?i)\b\d{1,4}\s+[A-Za-z0-9'.,\s]{2,45}?(?:Road|Rd\.?|Street|St\.?|Avenue|Ave\.?|Drive|Dr\.?|"
        r"Lane|Ln\.?|Close|Cl\.?|Crescent|Cres\.?|Way|Boulevard|Blvd\.?)\b"
    )
    for m in street_rx.finditer(text):
        ids.add(f"address:{' '.join(m.group(0).lower().split())}")
    for m in re.finditer(r"(?i)\bblk\.?\s*\d+[A-Za-z]?\b", text):
        ids.add(f"address:{' '.join(m.group(0).lower().split())}")
    for m in re.finditer(r"#\d{2}-\d{2,5}\b", text):
        ids.add(f"address:{m.group(0).lower()}")
    for m in re.finditer(r"\b(?:Singapore\s*)?\d{6}\b", text):
        ids.add(f"address:postal {m.group(0).strip()[-6:]}")
    return ids


def _normalize_person_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _chunk_mentions_target_name(person_norm: str, text: str) -> bool:
    if not person_norm:
        return False
    tl = text.lower()
    if person_norm in tl:
        return True
    for h in extract_all_entities(text):
        if h.label != "person":
            continue
        if _normalize_person_name(h.text) == person_norm:
            return True
    return False


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _cluster_confidence(chunks: list[tuple[Any, float]], union_ids: set[str]) -> str:
    if not union_ids:
        return "Low"
    id_to_sources: dict[str, set[str]] = defaultdict(set)
    for r, _ in chunks:
        for k in extract_linking_identifiers(r.text):
            id_to_sources[k].add(r.source_file)
    if any(len(srcs) >= 2 for srcs in id_to_sources.values()):
        return "High"
    return "Medium"


def format_linking_id_key(k: str) -> str:
    prefix, _, rest = k.partition(":")
    return f"{prefix.replace('_', ' ').title()}: {rest}"


@dataclass
class IdentityCluster:
    """A connected component of name mentions linked by shared non-person identifiers."""

    display_index: int
    chunks: list[tuple[Any, float]]
    linking_ids: set[str]
    confidence: str
    is_unlinked: bool


@dataclass
class SpellingMatchHint:
    surface_name: str
    similarity: float
    example_chunk_id: str
    source_file: str
    snippet: str


@dataclass
class IdentityClusterResult:
    query_phrase: str
    clusters: list[IdentityCluster] = field(default_factory=list)
    spelling_matches: list[SpellingMatchHint] = field(default_factory=list)


def _spelling_similarity(a: str, b: str) -> float:
    al, bl = a.lower(), b.lower()
    if al == bl:
        return 1.0
    return SequenceMatcher(None, al, bl).ratio()


def _is_spelling_candidate(query_norm: str, candidate_norm: str) -> bool:
    if not query_norm or not candidate_norm or query_norm == candidate_norm:
        return False
    sim = _spelling_similarity(query_norm, candidate_norm)
    if sim >= 0.92:
        return True
    q_parts = query_norm.split()
    c_parts = candidate_norm.split()
    if len(q_parts) >= 2 and len(c_parts) >= 2:
        f = SequenceMatcher(None, q_parts[0], c_parts[0]).ratio()
        l = SequenceMatcher(None, q_parts[-1], c_parts[-1]).ratio()
        if f >= 0.78 and l >= 0.78 and sim >= 0.72:
            return True
    return False


def build_person_identity_clusters(
    person_phrase: str,
    ranked: list[tuple[Any, float]],
) -> IdentityClusterResult:
    """
    Partition retrieved chunks that mention ``person_phrase`` by shared linking identifiers.
    Does not merge different spellings into one identity; those appear under spelling hints only.
    """
    phrase = person_phrase.strip()
    pn = _normalize_person_name(phrase)
    out = IdentityClusterResult(query_phrase=phrase)
    if not pn:
        return out

    mention_rows: list[tuple[Any, float]] = []
    for r, s in ranked:
        if _chunk_mentions_target_name(pn, r.text):
            mention_rows.append((r, s))
    if not mention_rows:
        return out

    n = len(mention_rows)
    per_node_ids: list[set[str]] = [extract_linking_identifiers(r.text) for r, _ in mention_rows]
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if per_node_ids[i] & per_node_ids[j]:
                uf.union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    raw_components: list[list[tuple[ChunkRecord, float]]] = []
    for idxs in groups.values():
        ch = [mention_rows[i] for i in idxs]
        raw_components.append(ch)

    linked_clusters: list[IdentityCluster] = []
    unlinked_rows: list[list[tuple[ChunkRecord, float]]] = []
    for comp in raw_components:
        uids: set[str] = set()
        for r, _ in comp:
            uids |= extract_linking_identifiers(r.text)
        if not uids:
            unlinked_rows.append(comp)
        else:
            conf = _cluster_confidence(comp, uids)
            linked_clusters.append(
                IdentityCluster(
                    display_index=0,
                    chunks=comp,
                    linking_ids=uids,
                    confidence=conf,
                    is_unlinked=False,
                )
            )

    def _sort_key(ic: IdentityCluster) -> tuple[int, int]:
        rank = {"High": 3, "Medium": 2, "Low": 1}.get(ic.confidence, 0)
        return (rank, len(ic.chunks))

    linked_clusters.sort(key=_sort_key, reverse=True)
    display_n = 1
    for ic in linked_clusters:
        ic.display_index = display_n
        display_n += 1
    unlinked_ics: list[IdentityCluster] = [
        IdentityCluster(
            display_index=0,
            chunks=comp,
            linking_ids=set(),
            confidence="Low",
            is_unlinked=True,
        )
        for comp in unlinked_rows
    ]
    out.clusters = linked_clusters + unlinked_ics

    seen_names: set[str] = {pn}
    hints: list[SpellingMatchHint] = []
    for r, _ in ranked:
        for h in extract_all_entities(r.text):
            if h.label != "person":
                continue
            raw = h.text.strip()
            if not raw:
                continue
            cn = _normalize_person_name(raw)
            if cn in seen_names:
                continue
            if not _is_spelling_candidate(pn, cn):
                continue
            seen_names.add(cn)
            snip = " ".join(r.text.split())[:220]
            hints.append(
                SpellingMatchHint(
                    surface_name=raw,
                    similarity=round(_spelling_similarity(pn, cn), 3),
                    example_chunk_id=r.chunk_id,
                    source_file=r.source_file,
                    snippet=snip + ("…" if len(r.text) > 220 else ""),
                )
            )
    out.spelling_matches = sorted(hints, key=lambda h: -h.similarity)[:12]
    return out


def pick_default_cluster_index(clusters: list[IdentityCluster]) -> int:
    """Index into ``clusters`` for highest-confidence / largest linked component; else first row."""
    if not clusters:
        return 0
    linked = [(i, c) for i, c in enumerate(clusters) if not c.is_unlinked]
    if linked:
        i, _best = max(
            linked,
            key=lambda it: ({"High": 3, "Medium": 2, "Low": 1}.get(it[1].confidence, 0), len(it[1].chunks)),
        )
        return i
    return 0


def cluster_summary_label(cluster: IdentityCluster) -> str:
    if cluster.is_unlinked:
        return "Unlinked mention"
    return f"Identity Cluster {cluster.display_index}"
