"""Lexical + semantic diversity metrics for persona texts.

Uses the same layered metric as parallel_gepa:
  - lexical: 1 - word-ngram Jaccard over shingles
  - semantic: embedding cosine similarity (optional, via litellm)
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    toks = normalize_tokens(text)
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard_dissim(a: str, b: str, *, n: int = 3) -> float:
    sa, sb = shingles(a, n=n), shingles(b, n=n)
    if not sa and not sb:
        return 0.0
    union = len(sa | sb)
    return 1.0 - (len(sa & sb) / union if union else 0.0)


def min_lexical_dissim(text: str, others: list[str], *, n: int = 3) -> float:
    return 1.0 if not others else min(jaccard_dissim(text, o, n=n) for o in others)


def cosine(u: list[float], v: list[float]) -> float:
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    return sum(x * y for x, y in zip(u, v)) / (nu * nv) if nu and nv else 0.0


@dataclass
class DiversityConfig:
    shingle_n: int = 3
    embedding_model: str | None = "text-embedding-3-small"
    min_dissimilarity: float = 0.3
    max_semantic_similarity: float = 0.9


class Embedder:
    """litellm embedding with process cache; disables on first failure."""

    def __init__(self, model: str | None):
        self.model = model
        self._cache: dict[str, list[float]] = {}
        self._disabled = False
        self.disable_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.model) and not self._disabled

    def embed(self, text: str) -> list[float] | None:
        if not self.enabled:
            return None
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        try:
            import litellm

            resp = litellm.embedding(model=self.model, input=[text])
            vec = list(resp.data[0]["embedding"])
            self._cache[key] = vec
            return vec
        except Exception as e:  # noqa: BLE001 — best-effort semantic layer
            self._disabled = True
            self.disable_reason = str(e)
            return None

    def max_cosine(self, text: str, others: list[str]) -> float | None:
        vec = self.embed(text)
        if vec is None or not others:
            return None
        best = 0.0
        for o in others:
            ov = self.embed(o)
            if ov is not None:
                best = max(best, cosine(vec, ov))
        return best


def _most_similar_snippet(text: str, seen: list[str], n: int = 3) -> str:
    """Return a short snippet of the most lexically similar seen persona."""
    if not seen:
        return ""
    best_idx, best_sim = 0, 0.0
    for i, s in enumerate(seen):
        sim = 1.0 - min_lexical_dissim(text, [s], n=n)
        if sim > best_sim:
            best_sim, best_idx = sim, i
    lines = seen[best_idx].strip().split("\n")
    snippet = "\n".join(lines[:8])
    if len(snippet) > 400:
        snippet = snippet[:400] + "..."
    return snippet


def diversity_check(
    cfg: DiversityConfig,
    text: str,
    seen: list[str],
    embedder: Embedder | None = None,
) -> tuple[bool, float, float | None, str, str]:
    """Same gate as parallel_gepa mutation diversity check.
    Returns (ok, lex_dissim, sem_sim, reason, nearest_snippet)."""
    lex = min_lexical_dissim(text, seen, n=cfg.shingle_n)
    if lex < cfg.min_dissimilarity:
        snippet = _most_similar_snippet(text, seen, n=cfg.shingle_n)
        return False, lex, None, "lexical_near_duplicate", snippet
    emb = embedder or Embedder(cfg.embedding_model)
    sem = emb.max_cosine(text, seen) if emb.enabled else None
    if sem is not None and sem > cfg.max_semantic_similarity:
        snippet = _most_similar_snippet(text, seen, n=cfg.shingle_n)
        return False, lex, sem, "semantic_near_duplicate", snippet
    return True, lex, sem, "ok", ""


@dataclass
class PersonaItem:
    id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PairViolation:
    a: str
    b: str
    lexical_dissim: float
    semantic_sim: float | None
    reason: str


@dataclass
class DiversityReport:
    items: list[PersonaItem]
    config: DiversityConfig
    lexical_matrix: list[list[float]]
    semantic_matrix: list[list[float | None]]
    per_item: list[dict[str, Any]]
    summary: dict[str, Any]
    violations: list[PairViolation]
    embedding_enabled: bool
    embedding_disable_reason: str | None = None


def _pairwise_lexical(texts: list[str], *, n: int) -> list[list[float]]:
    n_items = len(texts)
    mat = [[0.0] * n_items for _ in range(n_items)]
    for i in range(n_items):
        for j in range(i + 1, n_items):
            d = jaccard_dissim(texts[i], texts[j], n=n)
            mat[i][j] = mat[j][i] = d
    return mat


def _pairwise_semantic(
    texts: list[str], embedder: Embedder
) -> list[list[float | None]]:
    n_items = len(texts)
    mat: list[list[float | None]] = [[None] * n_items for _ in range(n_items)]
    if not embedder.enabled:
        return mat
    vecs = [embedder.embed(t) for t in texts]
    for i in range(n_items):
        for j in range(i + 1, n_items):
            if vecs[i] is None or vecs[j] is None:
                continue
            s = cosine(vecs[i], vecs[j])  # type: ignore[arg-type]
            mat[i][j] = mat[j][i] = s
    return mat


def _off_diag(values: list[list[float | None]]) -> list[float]:
    out: list[float] = []
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            v = values[i][j]
            if v is not None:
                out.append(float(v))
    return out


def analyze_diversity(
    items: list[PersonaItem], cfg: DiversityConfig | None = None
) -> DiversityReport:
    if len(items) < 2:
        raise ValueError("need at least 2 personas to measure diversity")
    cfg = cfg or DiversityConfig()
    texts = [it.text for it in items]
    embedder = Embedder(cfg.embedding_model)

    lex_mat = _pairwise_lexical(texts, n=cfg.shingle_n)
    sem_mat = _pairwise_semantic(texts, embedder)

    per_item: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        min_lex = min((lex_mat[i][j] for j in range(len(texts)) if j != i), default=1.0)
        sem_sims = [
            sem_mat[i][j]
            for j in range(len(texts))
            if j != i and sem_mat[i][j] is not None
        ]
        max_sem = max(sem_sims) if sem_sims else None
        per_item.append(
            {
                "id": it.id,
                "min_lexical_dissim": round(min_lex, 4),
                "max_semantic_sim": round(max_sem, 4) if max_sem is not None else None,
                "passes_lexical_gate": min_lex >= cfg.min_dissimilarity,
                "passes_semantic_gate": max_sem is None
                or max_sem <= cfg.max_semantic_similarity,
                **it.meta,
            }
        )

    lex_pairs = _off_diag(lex_mat)
    sem_pairs = _off_diag(sem_mat)

    violations: list[PairViolation] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            lex = lex_mat[i][j]
            sem = sem_mat[i][j]
            reasons: list[str] = []
            if lex < cfg.min_dissimilarity:
                reasons.append("lexical_near_duplicate")
            if sem is not None and sem > cfg.max_semantic_similarity:
                reasons.append("semantic_near_duplicate")
            if reasons:
                violations.append(
                    PairViolation(
                        a=items[i].id,
                        b=items[j].id,
                        lexical_dissim=lex,
                        semantic_sim=sem,
                        reason="; ".join(reasons),
                    )
                )

    summary = {
        "n": len(items),
        "lexical_pairwise_mean": round(sum(lex_pairs) / len(lex_pairs), 4)
        if lex_pairs
        else None,
        "lexical_pairwise_min": round(min(lex_pairs), 4) if lex_pairs else None,
        "lexical_pairwise_max": round(max(lex_pairs), 4) if lex_pairs else None,
        "min_lexical_dissim_mean": round(
            sum(p["min_lexical_dissim"] for p in per_item) / len(per_item), 4
        ),
        "min_lexical_dissim_min": round(
            min(p["min_lexical_dissim"] for p in per_item), 4
        ),
        "semantic_pairwise_mean": round(sum(sem_pairs) / len(sem_pairs), 4)
        if sem_pairs
        else None,
        "semantic_pairwise_max": round(max(sem_pairs), 4) if sem_pairs else None,
        "max_semantic_sim_mean": round(
            sum(
                p["max_semantic_sim"]
                for p in per_item
                if p["max_semantic_sim"] is not None
            )
            / max(1, sum(1 for p in per_item if p["max_semantic_sim"] is not None)),
            4,
        )
        if any(p["max_semantic_sim"] is not None for p in per_item)
        else None,
        "gate_violations": len(violations),
        "all_pass_lexical_gate": all(p["passes_lexical_gate"] for p in per_item),
        "all_pass_semantic_gate": all(p["passes_semantic_gate"] for p in per_item),
    }

    return DiversityReport(
        items=items,
        config=cfg,
        lexical_matrix=lex_mat,
        semantic_matrix=sem_mat,
        per_item=per_item,
        summary=summary,
        violations=violations,
        embedding_enabled=embedder.enabled,
        embedding_disable_reason=embedder.disable_reason,
    )


def report_to_dict(report: DiversityReport) -> dict[str, Any]:
    return {
        "config": {
            "shingle_n": report.config.shingle_n,
            "embedding_model": report.config.embedding_model,
            "min_dissimilarity": report.config.min_dissimilarity,
            "max_semantic_similarity": report.config.max_semantic_similarity,
        },
        "embedding_enabled": report.embedding_enabled,
        "embedding_disable_reason": report.embedding_disable_reason,
        "summary": report.summary,
        "items": report.per_item,
        "violations": [
            {
                "a": v.a,
                "b": v.b,
                "lexical_dissim": round(v.lexical_dissim, 4),
                "semantic_sim": round(v.semantic_sim, 4)
                if v.semantic_sim is not None
                else None,
                "reason": v.reason,
            }
            for v in report.violations
        ],
        "lexical_matrix": [[round(x, 4) for x in row] for row in report.lexical_matrix],
        "semantic_matrix": [
            [round(x, 4) if x is not None else None for x in row]
            for row in report.semantic_matrix
        ],
        "ids": [it.id for it in report.items],
    }


def _fmt_matrix(
    ids: list[str], mat: list[list[float | None]], *, upper_triangle: bool = True
) -> str:
    w = max(len(i) for i in ids)
    header = " " * (w + 2) + "  ".join(f"{i:>{w}}" for i in ids)
    lines = [header]
    for i, row_id in enumerate(ids):
        cells: list[str] = []
        for j, val in enumerate(mat[i]):
            if upper_triangle and j <= i:
                cells.append(" " * w)
            elif val is None:
                cells.append(f"{'—':>{w}}")
            else:
                cells.append(f"{val:>{w}.3f}")
        lines.append(f"{row_id:>{w}}  " + "  ".join(cells))
    return "\n".join(lines)


def render_markdown(report: DiversityReport) -> str:
    cfg = report.config
    s = report.summary
    ids = [it.id for it in report.items]

    def _sem_cell(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "—"

    sem_note = (
        f"embedding `{cfg.embedding_model}`"
        if report.embedding_enabled
        else f"semantic layer off{': ' + report.embedding_disable_reason if report.embedding_disable_reason else ''}"
    )

    viol_block = ""
    if report.violations:
        rows = "\n".join(
            f"| {v.a} | {v.b} | {v.lexical_dissim:.3f} | "
            f"{_sem_cell(v.semantic_sim)} | {v.reason} |"
            for v in report.violations
        )
        viol_block = f"""
## Gate violations (lex>={cfg.min_dissimilarity}, sem<={cfg.max_semantic_similarity})

| a | b | lexical_dissim | semantic_sim | reason |
|---|---|----------------|--------------|--------|
{rows}
"""
    else:
        viol_block = "\n## Gate violations\n\nNone — all pairs pass configured gates.\n"

    per_rows = "\n".join(
        f"| {p['id']} | {p['min_lexical_dissim']:.3f} | "
        f"{_sem_cell(p['max_semantic_sim'])} | "
        f"{'✓' if p['passes_lexical_gate'] else '✗'} | "
        f"{'✓' if p['passes_semantic_gate'] else '✗'} |"
        for p in report.per_item
    )

    return f"""# Persona diversity report

## Config
- lexical: 1 − word-{cfg.shingle_n}gram Jaccard
- semantic: {sem_note}
- gates: min_lexical_dissim ≥ **{cfg.min_dissimilarity}**, max_semantic_sim ≤ **{cfg.max_semantic_similarity}**

## Summary ({s["n"]} personas)
- pairwise lexical: mean **{s["lexical_pairwise_mean"]}**, min **{s["lexical_pairwise_min"]}**, max **{s["lexical_pairwise_max"]}**
- per-persona min lexical (nearest neighbor): mean **{s["min_lexical_dissim_mean"]}**, min **{s["min_lexical_dissim_min"]}**
- pairwise semantic: mean **{s["semantic_pairwise_mean"] or "—"}**, max **{s["semantic_pairwise_max"] or "—"}**
- gate violations: **{s["gate_violations"]}**

## Per persona (nearest-neighbor view)

| id | min_lex_dissim | max_sem_sim | lex gate | sem gate |
|----|----------------|-------------|----------|----------|
{per_rows}
{viol_block}
## Pairwise lexical dissimilarity

```
{_fmt_matrix(ids, report.lexical_matrix)}
```

## Pairwise semantic similarity

```
{_fmt_matrix(ids, report.semantic_matrix)}
```
"""
