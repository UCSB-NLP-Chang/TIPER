#!/usr/bin/env python3
"""Build a parent→child lineage tree from a pgepa run state file.

Accepted candidates become branch nodes; each generation log (accepted or
rejected) is attached under its selected parent idx. Rejected attempts are
stored in ``rejected``; accepted ones promote to a ``children`` pool node.

Outputs under <run-dir>/:
  lineage_tree.json   — nested tree + flat index
  lineage_tree.md     — human summary + mermaid diagram

Usage:
    python experiments/persona_search/scripts/pgepa_lineage_tree.py \\
        --run-dir data/persona_opt/pgepa_runs/pgepa_brinks01_run1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import PERSONAS_DIR, REPO_ROOT, read_json, write_json  # noqa: E402

_TITLE_RE = re.compile(r"^#\s*Persona:\s*(.+?)(?:\s*\(effect|\s*$)", re.I)


def persona_title(text: str | None) -> str:
    if not text:
        return "?"
    for line in text.splitlines():
        m = _TITLE_RE.match(line.strip())
        if m:
            return m.group(1).strip()
    return "?"


def skill_path(base: str, gen: int) -> Path:
    return PERSONAS_DIR / f"opt_{base}_g{gen:02d}_k00.skill"


def resolve_persona_text(
    gl: dict,
    candidates: dict[int, dict],
    base: str,
) -> str | None:
    if gl.get("persona_text"):
        return gl["persona_text"]
    if gl.get("accepted") and gl.get("idx") is not None:
        c = candidates.get(gl["idx"])
        if c and c.get("persona_text"):
            return c["persona_text"]
    p = skill_path(base, gl["gen"])
    return p.read_text() if p.exists() else None


def _attempt_node(gl: dict, text: str | None, base: str) -> dict[str, Any]:
    g = gl["gen"]
    sp = skill_path(base, g)
    return {
        "kind": "attempt",
        "gen": g,
        "parent_idx": gl["parent"],
        "title": persona_title(text),
        "accepted": bool(gl.get("accepted")),
        "accept_reason": gl.get("accept_reason"),
        "idx": gl.get("idx"),
        "minibatch_mean": gl.get("minibatch_mean"),
        "parent_minibatch_mean": gl.get("parent_minibatch_mean"),
        "minibatch_passed": gl.get("minibatch_passed"),
        "difficulty_agg": gl.get("difficulty_agg"),
        "validity_agg": gl.get("validity_agg"),
        "feasible": gl.get("feasible"),
        "lexical_dissim": gl.get("lexical_dissim"),
        "semantic_sim": gl.get("semantic_sim"),
        "confirm_rerun": gl.get("confirm_rerun"),
        "skill_path": str(sp.relative_to(REPO_ROOT)) if sp.exists() else None,
        "persona_text": text,
    }


def _pool_node(c: dict, *, include_text: bool = True) -> dict[str, Any]:
    node: dict[str, Any] = {
        "kind": "pool",
        "idx": c["idx"],
        "gen": c.get("gen"),
        "parent_idx": c.get("parent"),
        "title": persona_title(c.get("persona_text")),
        "difficulty_agg": c.get("difficulty_agg"),
        "validity_agg": c.get("validity_agg"),
        "feasible": c.get("feasible"),
        "accepted": True,
        "children": [],
        "rejected": [],
        "via_gen": c.get("gen"),
    }
    if include_text:
        node["persona_text"] = c.get("persona_text")
    return node


def _slim_attempt(att: dict) -> dict:
    return {k: v for k, v in att.items() if k != "persona_text"}


def _slim_pool(node: dict) -> dict:
    out = {
        k: v
        for k, v in node.items()
        if k not in ("persona_text", "children", "rejected")
    }
    out["children"] = [
        {
            "via_gen": e["via_gen"],
            "attempt": _slim_attempt(e["attempt"]),
            "node": _slim_pool(e["node"]),
        }
        for e in node.get("children") or []
    ]
    out["rejected"] = [_slim_attempt(a) for a in node.get("rejected") or []]
    return out


def build_lineage_tree(state: dict) -> dict[str, Any]:
    """Return ``{root, index, stats}`` from a pgepa_state.json dict."""
    cfg = state.get("config") or {}
    base = cfg.get("persona_id_base") or "pgepa"
    candidates = {c["idx"]: c for c in (state.get("candidates") or [])}
    generations = sorted(state.get("generations") or [], key=lambda x: x.get("gen", 0))

    if 0 not in candidates:
        raise ValueError("state has no seed candidate (idx 0)")

    pool_by_idx = {
        idx: _pool_node(c, include_text=False) for idx, c in candidates.items()
    }
    for node in pool_by_idx.values():
        node["children"] = []
        node["rejected"] = []
    index: list[dict[str, Any]] = []

    for gl in generations:
        parent_idx = gl.get("parent")
        if parent_idx is None or parent_idx not in pool_by_idx:
            continue
        text = resolve_persona_text(gl, candidates, base)
        attempt = _attempt_node(gl, text, base)
        index.append(attempt)
        if gl.get("accepted") and gl.get("idx") is not None:
            child_idx = gl["idx"]
            pool_by_idx[parent_idx]["children"].append(
                {
                    "via_gen": gl["gen"],
                    "attempt": attempt,
                    "node": pool_by_idx[child_idx],
                }
            )
        else:
            pool_by_idx[parent_idx]["rejected"].append(attempt)

    accepted_gens = sum(1 for g in generations if g.get("accepted"))
    rejected_gens = len(generations) - accepted_gens

    return {
        "persona_id_base": base,
        "root": _slim_pool(pool_by_idx[0]),
        "pool": {
            str(k): {
                "idx": k,
                "gen": v.get("gen"),
                "title": v.get("title"),
                "difficulty_agg": v.get("difficulty_agg"),
                "parent_idx": v.get("parent_idx"),
            }
            for k, v in pool_by_idx.items()
        },
        "personas": {
            **{f"idx{k}": candidates[k]["persona_text"] for k in candidates},
            **{
                f"gen{a['gen']}": a["persona_text"]
                for a in index
                if a.get("persona_text") and not a.get("accepted")
            },
        },
        "index": [{k: v for k, v in a.items() if k != "persona_text"} for a in index],
        "stats": {
            "pool_size": len(candidates),
            "generations": len(generations),
            "accepted_generations": accepted_gens,
            "rejected_generations": rejected_gens,
            "best_idx": state.get("best_idx"),
            "best_agg": state.get("best_agg"),
        },
    }


def build_idx_tree(state: dict) -> dict[str, Any]:
    """Build a tree keyed by pool idx: parent→child edges follow candidate.parent."""
    cfg = state.get("config") or {}
    base = cfg.get("persona_id_base") or "pgepa"
    candidates = {c["idx"]: c for c in (state.get("candidates") or [])}
    generations = state.get("generations") or []

    if 0 not in candidates:
        raise ValueError("state has no seed candidate (idx 0)")

    rejected_by_parent: dict[int, list[dict]] = {}
    via_gen_by_idx: dict[int, int] = {}
    for gl in generations:
        parent_idx = gl.get("parent")
        if parent_idx is None:
            continue
        g = gl["gen"]
        if gl.get("accepted") and gl.get("idx") is not None:
            via_gen_by_idx[gl["idx"]] = g
        else:
            text = resolve_persona_text(gl, candidates, base)
            rejected_by_parent.setdefault(parent_idx, []).append(
                _attempt_node(gl, text, base)
            )

    def _node(idx: int) -> dict[str, Any]:
        c = candidates[idx]
        child_idxs = sorted(
            (i for i, cc in candidates.items() if cc.get("parent") == idx),
            key=lambda i: (via_gen_by_idx.get(i, candidates[i].get("gen") or 0), i),
        )
        return {
            "idx": idx,
            "gen": c.get("gen"),
            "via_gen": via_gen_by_idx.get(idx, c.get("gen") if idx else None),
            "parent_idx": c.get("parent"),
            "title": persona_title(c.get("persona_text")),
            "difficulty_agg": c.get("difficulty_agg"),
            "validity_agg": c.get("validity_agg"),
            "feasible": c.get("feasible"),
            "rejected": sorted(rejected_by_parent.get(idx, []), key=lambda x: x["gen"]),
            "children": [_node(i) for i in child_idxs],
        }

    best_idx = state.get("best_idx")
    return {
        "persona_id_base": base,
        "best_idx": best_idx,
        "root": _node(0),
        "pool": {
            str(k): {
                "idx": k,
                "gen": v.get("gen"),
                "via_gen": via_gen_by_idx.get(k, v.get("gen")),
                "parent_idx": v.get("parent"),
                "title": persona_title(v.get("persona_text")),
                "difficulty_agg": v.get("difficulty_agg"),
            }
            for k, v in candidates.items()
        },
    }


def _idx_label(node: dict[str, Any], *, best_idx: int | None = None) -> str:
    idx = node["idx"]
    diff = node.get("difficulty_agg")
    diff_s = f" {diff:.2f}" if isinstance(diff, (int, float)) else ""
    title = (node.get("title") or "?")[:26]
    via = node.get("via_gen")
    via_s = f"g{via}" if via is not None and idx != 0 else "seed"
    star = " ★" if best_idx is not None and idx == best_idx else ""
    return f"idx{idx} {via_s}{diff_s}{star}<br/>{title}"


def _reject_label(att: dict) -> str:
    mb = att.get("minibatch_mean")
    mb_s = f" mb={mb:.2f}" if isinstance(mb, (int, float)) else ""
    title = (att.get("title") or "?")[:22]
    return f"g{att['gen']} ✗{mb_s}<br/>{title}"


def _render_idx_md(
    node: dict[str, Any],
    depth: int = 0,
    *,
    best_idx: int | None = None,
    include_rejected: bool = False,
) -> list[str]:
    indent = "  " * depth
    idx = node["idx"]
    diff = node.get("difficulty_agg")
    diff_s = f" diff={diff:.2f}" if isinstance(diff, (int, float)) else ""
    via = node.get("via_gen")
    via_s = f"via g{via}" if via is not None and idx != 0 else "seed"
    best_s = " **★ best**" if best_idx is not None and idx == best_idx else ""
    lines = [f"{indent}- **idx{idx}** ({via_s}) {node.get('title')!r}{diff_s}{best_s}"]

    if include_rejected:
        for att in node.get("rejected") or []:
            mb = att.get("minibatch_mean")
            mb_s = f" mb={mb:.2f}" if isinstance(mb, (int, float)) else ""
            lines.append(
                f"{indent}  - g{att['gen']} ✗ ({att.get('accept_reason', 'reject')})"
                f"{mb_s} — {att.get('title')!r}"
            )
    for child in node.get("children") or []:
        lines.extend(
            _render_idx_md(
                child, depth + 1, best_idx=best_idx, include_rejected=include_rejected
            )
        )
    return lines


def _render_idx_mermaid(
    node: dict[str, Any],
    lines: list[str],
    *,
    best_idx: int | None = None,
    include_rejected: bool = False,
) -> None:
    nid = f"idx{node['idx']}"
    if node["idx"] == 0:
        lines.append(f'  {nid}["{_idx_label(node, best_idx=best_idx)}"]')
        lines.append(f"  class {nid} pool")
    if include_rejected:
        for att in node.get("rejected") or []:
            aid = f"g{att['gen']}_x"
            lines.append(f"  {nid} --> {aid}")
            lines.append(f'  {aid}["{_reject_label(att)}"]')
            lines.append(f"  class {aid} rejected")
    for child in node.get("children") or []:
        cid = f"idx{child['idx']}"
        via = child.get("via_gen")
        edge = f"|g{via}|" if via is not None else ""
        lines.append(f"  {nid} -->{edge} {cid}")
        lines.append(f'  {cid}["{_idx_label(child, best_idx=best_idx)}"]')
        if best_idx is not None and child["idx"] == best_idx:
            lines.append(f"  class {cid} best")
        else:
            lines.append(f"  class {cid} pool")
        _render_idx_mermaid(
            child, lines, best_idx=best_idx, include_rejected=include_rejected
        )


def render_idx_mermaid(
    idx_tree: dict[str, Any], *, include_rejected: bool = False
) -> str:
    best_idx = idx_tree.get("best_idx")
    lines = ["flowchart TD"]
    if include_rejected:
        lines.append("  classDef rejected fill:#fee,stroke:#c99,color:#633")
    lines.extend(
        [
            "  classDef pool fill:#eef6ff,stroke:#69c,color:#036",
            "  classDef best fill:#efe,stroke:#6a6,color:#363",
        ]
    )
    _render_idx_mermaid(
        idx_tree["root"], lines, best_idx=best_idx, include_rejected=include_rejected
    )
    return "\n".join(lines)


def render_idx_markdown(
    idx_tree: dict[str, Any], stats: dict, *, include_rejected: bool = False
) -> str:
    body = "\n".join(
        _render_idx_md(
            idx_tree["root"],
            best_idx=idx_tree.get("best_idx"),
            include_rejected=include_rejected,
        )
    )
    mermaid = render_idx_mermaid(idx_tree, include_rejected=include_rejected)
    rejected_note = (
        " (rejected attempts shown)" if include_rejected else " (pool idx only)"
    )
    return f"""# PGEPA idx tree — `{idx_tree["persona_id_base"]}`

## Stats
- pool: **{stats["pool_size"]}** candidates (idx0–idx{stats["pool_size"] - 1})
- generations: **{stats["generations"]}** ({stats["accepted_generations"]} accepted, {stats["rejected_generations"]} rejected)
- best: **idx{stats["best_idx"]}** (difficulty {stats["best_agg"]})

## Tree by idx{rejected_note}

{body}

## Mermaid (idx nodes)

```mermaid
{mermaid}
```
"""


def _mermaid_id(kind: str, idx: int | None = None, gen: int | None = None) -> str:
    if kind == "pool" and idx is not None:
        return f"idx{idx}"
    if gen is not None:
        return f"gen{gen}"
    return "root"


def _mermaid_label(node: dict[str, Any], *, attempt: dict | None = None) -> str:
    if attempt:
        status = "ok" if attempt.get("accepted") else "x"
        diff = attempt.get("difficulty_agg")
        diff_s = f" {diff:.2f}" if isinstance(diff, (int, float)) else ""
        mb = attempt.get("minibatch_mean")
        mb_s = f" mb={mb:.2f}" if isinstance(mb, (int, float)) else ""
        reason = attempt.get("accept_reason") or ""
        if not attempt.get("accepted") and reason:
            short = reason.replace("below_parent_on_minibatch", "mb<parent")
            short = short.replace("failed_feasibility_gate", "infeasible")
            return f"g{attempt['gen']} [{status}]{mb_s}<br/>{short}"
        return f"g{attempt['gen']} [{status}]{diff_s}{mb_s}"
    title = (node.get("title") or "?")[:28]
    diff = node.get("difficulty_agg")
    diff_s = f" {diff:.2f}" if isinstance(diff, (int, float)) else ""
    idx = node.get("idx")
    gen = node.get("gen")
    if idx == 0:
        return f"seed{diff_s}<br/>{title}"
    return f"idx{idx} g{gen}{diff_s}<br/>{title}"


def _walk_mermaid(
    node: dict[str, Any], lines: list[str], parent_id: str | None
) -> None:
    nid = _mermaid_id("pool", idx=node.get("idx"))
    if parent_id:
        lines.append(f"  {parent_id} --> {nid}")
    lines.append(f'  {nid}["{_mermaid_label(node)}"]')

    for att in node.get("rejected") or []:
        aid = _mermaid_id("attempt", gen=att["gen"])
        lines.append(f"  {nid} --> {aid}")
        lines.append(f'  {aid}["{_mermaid_label({}, attempt=att)}"]')
        lines.append(f"  class {aid} rejected")

    for edge in node.get("children") or []:
        child = edge["node"]
        attempt = edge.get("attempt") or {}
        aid = _mermaid_id("attempt", gen=attempt.get("gen"))
        cid = _mermaid_id("pool", idx=child.get("idx"))
        lines.append(f"  {nid} --> {aid}")
        lines.append(f'  {aid}["{_mermaid_label({}, attempt=attempt)}"]')
        lines.append(f"  {aid} --> {cid}")
        _walk_mermaid(child, lines, parent_id=None)


def render_mermaid(tree: dict[str, Any]) -> str:
    lines = [
        "flowchart TD",
        "  classDef rejected fill:#fee,stroke:#c99,color:#633",
        "  classDef pool fill:#eef6ff,stroke:#69c,color:#036",
    ]
    root = tree["root"]
    rid = _mermaid_id("pool", idx=0)
    lines.append(f'  {rid}["{_mermaid_label(root)}"]')
    lines.append(f"  class {rid} pool")

    for att in root.get("rejected") or []:
        aid = _mermaid_id("attempt", gen=att["gen"])
        lines.append(f"  {rid} --> {aid}")
        lines.append(f'  {aid}["{_mermaid_label({}, attempt=att)}"]')
        lines.append(f"  class {aid} rejected")

    for edge in root.get("children") or []:
        child = edge["node"]
        attempt = edge.get("attempt") or {}
        aid = _mermaid_id("attempt", gen=attempt.get("gen"))
        cid = _mermaid_id("pool", idx=child.get("idx"))
        lines.append(f"  {rid} --> {aid}")
        lines.append(f'  {aid}["{_mermaid_label({}, attempt=attempt)}"]')
        lines.append(f"  {aid} --> {cid}")
        lines.append(f"  class {cid} pool")
        _walk_mermaid(child, lines, parent_id=None)

    return "\n".join(lines)


def _render_md_tree(node: dict[str, Any], depth: int = 0) -> list[str]:
    indent = "  " * depth
    idx = node.get("idx")
    diff = node.get("difficulty_agg")
    diff_s = f" diff={diff:.2f}" if isinstance(diff, (int, float)) else ""
    lines = [
        f"{indent}- **idx{idx}** (gen {node.get('gen')}) {node.get('title')!r}{diff_s}"
    ]

    for att in sorted(node.get("rejected") or [], key=lambda x: x["gen"]):
        mb = att.get("minibatch_mean")
        mb_s = f" mb={mb:.2f}" if isinstance(mb, (int, float)) else ""
        lines.append(
            f"{indent}  - gen{att['gen']} **reject** ({att.get('accept_reason')})"
            f"{mb_s} — {att.get('title')!r}"
        )

    for edge in sorted(node.get("children") or [], key=lambda x: x["attempt"]["gen"]):
        attempt = edge["attempt"]
        child = edge["node"]
        mb = attempt.get("minibatch_mean")
        mb_s = f" mb={mb:.2f}" if isinstance(mb, (int, float)) else ""
        lines.append(
            f"{indent}  - gen{attempt['gen']} **accept → idx{child.get('idx')}**"
            f"{mb_s} — {attempt.get('title')!r}"
        )
        lines.extend(_render_md_tree(child, depth + 2))
    return lines


def render_markdown(tree: dict[str, Any]) -> str:
    stats = tree["stats"]
    body = "\n".join(_render_md_tree(tree["root"]))
    mermaid = render_mermaid(tree)
    return f"""# PGEPA lineage tree — `{tree["persona_id_base"]}`

## Stats
- pool: **{stats["pool_size"]}** candidates
- generations: **{stats["generations"]}** ({stats["accepted_generations"]} accepted, {stats["rejected_generations"]} rejected)
- best: **idx{stats["best_idx"]}** (difficulty {stats["best_agg"]})

## Tree (accepted branches + rejected leaves)

{body}

## Mermaid

```mermaid
{mermaid}
```
"""


def write_lineage_tree(
    run_dir: Path, state: dict | None = None, *, include_rejected: bool = False
) -> Path:
    """Write lineage_tree.json, idx_tree.md; return json path."""
    run_dir = Path(run_dir)
    if state is None:
        state = read_json(run_dir / "pgepa_state.json")
    tree = build_lineage_tree(state)
    idx_tree = build_idx_tree(state)
    tree["idx_tree"] = idx_tree["root"]
    json_path = run_dir / "lineage_tree.json"
    idx_md_path = run_dir / "idx_tree.md"
    write_json(json_path, tree)
    idx_md_path.write_text(
        render_idx_markdown(idx_tree, tree["stats"], include_rejected=include_rejected)
    )
    return json_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export pgepa lineage as a nested tree")
    ap.add_argument("--run-dir", required=True, help="pgepa run directory")
    ap.add_argument(
        "--include-rejected",
        action="store_true",
        help="include rejected gen attempts in idx tree output",
    )
    ap.add_argument(
        "--print-mermaid", action="store_true", help="print idx-tree mermaid to stdout"
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    state_path = run_dir / "pgepa_state.json"
    if not state_path.exists():
        raise SystemExit(f"no state file: {state_path}")

    state = read_json(state_path)
    out = write_lineage_tree(run_dir, state, include_rejected=args.include_rejected)
    tree = read_json(out)
    print(f"Wrote {out}")
    print(f"Wrote {run_dir / 'idx_tree.md'}")
    print(
        f"  pool={tree['stats']['pool_size']}  gens={tree['stats']['generations']}  "
        f"best=idx{tree['stats']['best_idx']}"
    )
    if args.print_mermaid:
        print()
        print(
            render_idx_mermaid(
                build_idx_tree(state), include_rejected=args.include_rejected
            )
        )


if __name__ == "__main__":
    main()
