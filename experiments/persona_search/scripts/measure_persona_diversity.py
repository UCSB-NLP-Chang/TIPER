#!/usr/bin/env python3
"""Measure lexical + semantic diversity among persona texts.

Uses the same metrics as parallel_gepa mutation gate. Sources:
  --run-dir       pool candidates from pgepa_state.json
  --include-seen  also add rejected gen persona_text from generations[]
  --skill         one or more .skill / .md files
  --glob          glob pattern (e.g. data/personas/opt_pgepa_*.skill)

Examples:
    python experiments/persona_search/scripts/measure_persona_diversity.py \\
        --run-dir data/persona_opt/pgepa_runs/pgepa_brinks01_run1

    python experiments/persona_search/scripts/measure_persona_diversity.py \\
        --glob 'data/personas/opt_pgepa_brinks01_run1_*.skill' \\
        --output /tmp/brinks01_diversity
"""

from __future__ import annotations

import argparse
import glob as globmod
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, read_json, write_json  # noqa: E402
from persona_diversity import (  # noqa: E402
    DiversityConfig,
    PersonaItem,
    analyze_diversity,
    render_markdown,
    report_to_dict,
)
from pgepa_lineage_tree import persona_title  # noqa: E402


def _load_run_pool(run_dir: Path, *, include_seen: bool) -> list[PersonaItem]:
    state = read_json(run_dir / "pgepa_state.json")
    items: list[PersonaItem] = []
    seen_texts: set[str] = set()

    for c in state.get("candidates") or []:
        text = c.get("persona_text")
        if not text:
            continue
        pid = f"idx{c['idx']}"
        items.append(
            PersonaItem(
                id=pid,
                text=text,
                meta={
                    "title": persona_title(text),
                    "gen": c.get("gen"),
                    "difficulty_agg": c.get("difficulty_agg"),
                },
            )
        )
        seen_texts.add(text)

    if include_seen:
        for gl in state.get("generations") or []:
            text = gl.get("persona_text")
            if not text or text in seen_texts:
                continue
            gid = f"g{gl['gen']}{'_ok' if gl.get('accepted') else '_rej'}"
            items.append(
                PersonaItem(
                    id=gid,
                    text=text,
                    meta={
                        "title": persona_title(text),
                        "accepted": gl.get("accepted"),
                        "parent": gl.get("parent"),
                    },
                )
            )
            seen_texts.add(text)

    return items


def _load_paths(paths: list[str]) -> list[PersonaItem]:
    items: list[PersonaItem] = []
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            path = REPO_ROOT / path
        text = path.read_text()
        items.append(
            PersonaItem(
                id=path.stem,
                text=text,
                meta={"title": persona_title(text), "path": str(path)},
            )
        )
    return items


def _load_glob(pattern: str) -> list[PersonaItem]:
    if not Path(pattern).is_absolute():
        pattern = str(REPO_ROOT / pattern)
    paths = sorted(globmod.glob(pattern))
    if not paths:
        raise SystemExit(f"no files matched glob: {pattern}")
    return _load_paths(paths)


def _config_from_state(run_dir: Path | None, args) -> DiversityConfig:
    cfg = DiversityConfig(
        embedding_model=None if args.no_embedding else args.embedding_model,
        min_dissimilarity=args.min_dissimilarity,
        max_semantic_similarity=args.max_semantic_similarity,
    )
    if run_dir is None:
        return cfg
    state_path = run_dir / "pgepa_state.json"
    if not state_path.exists():
        return cfg
    sc = read_json(state_path).get("config") or {}
    if (
        not args.no_embedding
        and args.embedding_model == DiversityConfig.embedding_model
    ):
        cfg.embedding_model = sc.get("embedding_model", cfg.embedding_model)
    if args.min_dissimilarity == DiversityConfig.min_dissimilarity:
        cfg.min_dissimilarity = float(
            sc.get("min_dissimilarity", cfg.min_dissimilarity)
        )
    if args.max_semantic_similarity == DiversityConfig.max_semantic_similarity:
        cfg.max_semantic_similarity = float(
            sc.get("max_semantic_similarity", cfg.max_semantic_similarity)
        )
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="pgepa run dir with pgepa_state.json")
    src.add_argument("--skill", nargs="+", help="persona .skill or .md files")
    src.add_argument("--glob", dest="glob_pattern", help="glob of persona files")

    ap.add_argument(
        "--include-seen",
        action="store_true",
        help="with --run-dir, also include rejected gen persona_text",
    )
    ap.add_argument(
        "--output", type=Path, help="write diversity_report.md + .json (prefix or dir)"
    )
    ap.add_argument("--no-embedding", action="store_true", help="lexical only")
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--min-dissimilarity", type=float, default=0.3)
    ap.add_argument("--max-semantic-similarity", type=float, default=0.9)
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir and not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    if args.run_dir:
        items = _load_run_pool(run_dir, include_seen=args.include_seen)  # type: ignore[arg-type]
    elif args.skill:
        items = _load_paths(args.skill)
    else:
        items = _load_glob(args.glob_pattern)

    if len(items) < 2:
        raise SystemExit(f"need ≥2 personas, got {len(items)}")

    cfg = _config_from_state(run_dir, args)
    report = analyze_diversity(items, cfg)
    md = render_markdown(report)
    print(md)

    if args.output:
        out = args.output
        if out.suffix in (".md", ".json"):
            base = out.with_suffix("")
            md_path, json_path = base.with_suffix(".md"), base.with_suffix(".json")
        else:
            out.mkdir(parents=True, exist_ok=True)
            md_path, json_path = (
                out / "diversity_report.md",
                out / "diversity_report.json",
            )
        md_path.write_text(md)
        write_json(json_path, report_to_dict(report))
        print(f"\nWrote {md_path}\nWrote {json_path}")


if __name__ == "__main__":
    main()
