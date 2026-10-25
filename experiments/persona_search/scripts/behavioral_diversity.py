#!/usr/bin/env python3
"""Behavioral diversity of a persona pool via naturalness-manifold overlap.

Companion to measure_persona_diversity.py (which measures TEXT diversity: lexical
word-3gram + embedding cosine of the persona prompts). This measures BEHAVIORAL
diversity: do the personas' *user trajectories* occupy different regions of the
sim2real behavioral-feature space, or do they all talk the same way?

Pipeline (reuses the validated sim2real_pr.py rule pipeline, no LLM, no new sims):

    user trajectory  -> 19 rule behavioral metrics (sim2real_metrics.METRIC_ORDER)
    persona          -> a point cloud (one point per saved user trajectory)
    reference frame  -> standardize + PCA fit on the HUMAN manifold, so overlap
                        and naturalness live in the same "naturalness" coordinates

For each candidate idx we read its saved GEPA full-valset sims
(data/simulations/gepa_opt/<base>_g<NN>_k00_full_diff/.../results.json; the seed is
<base>_g00_seed), extract user turns, and project into human-PCA space. Then:

  * per persona:  P_geom = frac of its user turns inside the HUMAN manifold
                         = NATURALNESS (comparable across all candidates + seed);
                  R_geom = coverage (mode-collapse detector).
  * pairwise:     F1( cloud_i, cloud_j ) = geometric k-NN manifold overlap of two
                  persona clouds = the DISTRIBUTION OVERLAP. Low off-diagonal =
                  behaviorally distinct; high = textually varied but behaves the
                  same (redundant adversary).

NB: the human reference is cross-domain, so absolute naturalness is relative-
ranking-only -- which is all we need to compare the pool against itself.

    .venv/bin/python experiments/persona_search/scripts/behavioral_diversity.py \
        --run-dir data/persona_opt/pgepa_runs/pgepa_brinks01_run1 --k 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
sys.path.insert(0, str(REPO))  # import the repo-root sim2real_*.py

import sys

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
import os
import sys

import sim2real_metrics as M  # noqa: E402

_r2 = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r2, "pyproject.toml")) and _r2 != "/":
    _r2 = os.path.dirname(_r2)
sys.path.insert(0, os.path.join(_r2, "experiments", "baseline", "scripts"))
from sim2real_pr import (  # noqa: E402
    dims4,
    feature_matrix,
    load_human_features,
    pca_fit,
    precision_recall,
    sim_turn_lists,
    standardizer,
)

SIMS = REPO / "data" / "simulations" / "gepa_opt"


def _sim_dir(base: str, gen: int) -> Path:
    """Map a candidate (base run id, generation) -> its saved full-valset sim dir."""
    if gen == 0:
        return SIMS / f"{base}_g00_seed"
    return SIMS / f"{base}_g{gen:02d}_k00_full_diff"


def _results_path(d: Path) -> Path | None:
    """Find the richest results.json under a sim dir.

    A persona may have several eval_XXXX runs (some empty/aborted); pick the one
    with the most simulations rather than the first.
    """
    if not d.exists():
        return None
    best, best_n = None, -1
    for rp in sorted(d.glob("**/results.json")):
        try:
            n = len(json.load(open(rp)).get("simulations", []))
        except Exception:
            n = 0
        if n > best_n:
            best, best_n = rp, n
    return best if best_n > 0 else None


def _load_pool(run_dir: Path) -> list[dict]:
    """idx, gen, title, persona_text per candidate from pgepa_state.json."""
    state = json.load(open(run_dir / "pgepa_state.json"))
    base = state.get("config", {}).get("persona_id_base", run_dir.name)
    out = []
    for c in sorted(state["candidates"], key=lambda z: z["idx"]):
        title = c["persona_text"].splitlines()[0].lstrip("# ").strip()
        title = title.replace("Persona:", "").strip()
        out.append({"idx": c["idx"], "gen": c["gen"], "title": title, "base": base})
    return out


def _projector(mode: str, space: str):
    """Fit standardizer + PCA on the human manifold; return (project_fn, Hp)."""
    Hx_raw = load_human_features(mode)
    mu, sd = standardizer(Hx_raw)
    Hz = (Hx_raw - mu) / sd
    c, W, dpca = pca_fit(Hz)

    def project(x_raw: np.ndarray) -> np.ndarray:
        z = (x_raw - mu) / sd
        if space == "raw19":
            return z
        if space == "dims4":
            return dims4(z)
        return (z - c) @ W  # pca (default)

    Hp = project(Hx_raw)
    return project, Hp, dpca


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", default="strict", choices=["strict", "faithful"])
    ap.add_argument("--space", default="pca", choices=["pca", "raw19", "dims4"])
    ap.add_argument("--k", type=int, default=3, help="k-NN manifold radius rank")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-plots", action="store_true", help="skip PNG scatter/violins")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "diversity"
    out_dir.mkdir(parents=True, exist_ok=True)

    M.set_mode(args.mode)
    project, Hp, dpca = _projector(args.mode, args.space)

    pool = _load_pool(run_dir)

    # Build each persona's projected point cloud + per-persona naturalness/coverage.
    clouds: list[np.ndarray | None] = []
    raw_by_idx: dict[int, np.ndarray] = {}  # idx -> raw 19-metric trajectory matrix
    rows = []
    for c in pool:
        rp = _results_path(_sim_dir(c["base"], c["gen"]))
        if rp is None:
            print(f"  WARN idx{c['idx']} (gen {c['gen']}): no results.json, skipping")
            clouds.append(None)
            rows.append({**c, "n": 0, "naturalness": None, "coverage": None})
            continue
        Sx_raw = feature_matrix(sim_turn_lists(rp))
        if len(Sx_raw) == 0:
            print(f"  WARN idx{c['idx']} (gen {c['gen']}): 0 feature rows, skipping")
            clouds.append(None)
            rows.append({**c, "n": 0, "naturalness": None, "coverage": None})
            continue
        raw_by_idx[c["idx"]] = Sx_raw
        Sp = project(Sx_raw)
        clouds.append(Sp)
        P, Rc, _ = precision_recall(Hp, Sp, args.k)  # P=sim in human=naturalness
        rows.append(
            {
                **c,
                "n": int(len(Sp)),
                "naturalness": P,
                "coverage": Rc,
                "results_path": str(rp),
            }
        )

    n = len(pool)
    # Pairwise behavioral overlap matrix (F1 of cloud_i vs cloud_j).
    F1 = [[None] * n for _ in range(n)]
    for i in range(n):
        if clouds[i] is None:
            continue
        F1[i][i] = 1.0
        for j in range(i + 1, n):
            if clouds[j] is None:
                continue
            _, _, f1 = precision_recall(clouds[i], clouds[j], args.k)
            F1[i][j] = F1[j][i] = round(float(f1), 4)

    # Nearest behavioral neighbor (max off-diagonal F1) per persona.
    for i, r in enumerate(rows):
        offs = [(F1[i][j], j) for j in range(n) if j != i and F1[i][j] is not None]
        if offs:
            f, j = max(offs)
            r["nn_idx"], r["nn_overlap"], r["nn_title"] = (
                j,
                round(f, 4),
                rows[j]["title"],
            )
        else:
            r["nn_idx"] = r["nn_overlap"] = r["nn_title"] = None

    valid_overlaps = [
        F1[i][j] for i in range(n) for j in range(i + 1, n) if F1[i][j] is not None
    ]
    summary = {
        "space": f"{args.space}{dpca if args.space == 'pca' else ''}",
        "mode": args.mode,
        "k": args.k,
        "n_personas": sum(1 for c in clouds if c is not None),
        "pairwise_overlap_mean": round(float(np.mean(valid_overlaps)), 4)
        if valid_overlaps
        else None,
        "pairwise_overlap_min": round(float(np.min(valid_overlaps)), 4)
        if valid_overlaps
        else None,
        "pairwise_overlap_max": round(float(np.max(valid_overlaps)), 4)
        if valid_overlaps
        else None,
    }

    # Separability: does persona TEXT actually drive distinct behavior distributions?
    sep = _separability(
        raw_by_idx,
        {r["idx"]: r["title"] for r in rows},
        out_dir,
        mode=args.mode,
        make_plots=not args.no_plots,
    )
    report = {
        "summary": summary,
        "personas": rows,
        "overlap_matrix": F1,
        "separability": sep,
    }
    (out_dir / "behavioral_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "behavioral_report.md").write_text(_render_md(report))
    print(_render_md(report))
    print(f"\n-> {out_dir / 'behavioral_report.md'}")


def _separability(raw_by_idx, titles, out_dir, mode="strict", make_plots=True):
    """Quantify how much persona TEXT drives behavior, and visualize the clouds.

    - eta2 per metric: fraction of behavioral variance explained by which persona
      (the axes the persona prompt actually controls).
    - silhouette + leave-out logreg accuracy: can a trajectory be attributed to its
      persona? (>0 / >>chance => distinct distributions; ~0 / ~chance => collapsed).
    - per-persona z-scored centroid on the top driver metrics (the visible signature).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import cross_val_score

    idxs = sorted(raw_by_idx)
    X = np.vstack([raw_by_idx[i] for i in idxs])
    y = np.concatenate([[i] * len(raw_by_idx[i]) for i in idxs])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    MO = M.METRIC_ORDER

    # eta^2 per metric (between-persona SS / total SS) on z-scored features
    grand = Z.mean(0)
    eta = {}
    for k, name in enumerate(MO):
        ss_tot = float(((Z[:, k] - grand[k]) ** 2).sum())
        ss_bet = float(
            sum(
                len(raw_by_idx[i]) * (Z[y == i, k].mean() - grand[k]) ** 2 for i in idxs
            )
        )
        eta[name] = round(ss_bet / ss_tot, 4) if ss_tot > 0 else 0.0
    eta_sorted = dict(sorted(eta.items(), key=lambda kv: -kv[1]))

    sil = float(silhouette_score(Z, y))
    n_personas = len(idxs)
    cv = min(5, min(len(raw_by_idx[i]) for i in idxs))
    acc = float(cross_val_score(LogisticRegression(max_iter=2000), Z, y, cv=cv).mean())

    # per-persona z-scored centroid on the top-5 driver metrics
    drivers = list(eta_sorted)[:5]
    di = [MO.index(d) for d in drivers]
    profile = {
        int(i): {
            drivers[t]: round(float(Z[y == i][:, di[t]].mean()), 3)
            for t in range(len(drivers))
        }
        for i in idxs
    }

    sep = {
        "eta2_by_metric": eta_sorted,
        "silhouette": round(sil, 4),
        "logreg_cv_accuracy": round(acc, 4),
        "chance_accuracy": round(1.0 / n_personas, 4),
        "driver_metrics": drivers,
        "zprofile_on_drivers": profile,
    }

    if make_plots:
        try:
            _plots(Z, y, idxs, titles, drivers, di, out_dir)
            sep["plots"] = ["behavioral_pca.png", "behavioral_drivers.png"]
        except Exception as e:  # pragma: no cover
            print(f"  plot warning: {e}")
    return sep


def _plots(Z, y, idxs, titles, drivers, di, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    cmap = plt.get_cmap("tab10")
    # (1) PCA scatter of all trajectories, colored by persona, centroids labeled
    P2 = PCA(n_components=2).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c, i in enumerate(idxs):
        pts = P2[y == i]
        ax.scatter(
            pts[:, 0], pts[:, 1], s=14, alpha=0.45, color=cmap(c % 10), label=f"idx{i}"
        )
        cen = pts.mean(0)
        ax.scatter(
            *cen, s=180, color=cmap(c % 10), edgecolor="black", marker="o", zorder=5
        )
        ax.annotate(f"idx{i}", cen, fontsize=9, fontweight="bold", zorder=6)
    ax.set_title("Per-persona user-behavior distributions (19-metric z-space, PCA-2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "behavioral_pca.png", dpi=130)
    plt.close(fig)

    # (2) violins of the top driver metrics, per persona (where they DO diverge)
    nd = len(drivers)
    fig, axes = plt.subplots(1, nd, figsize=(3.2 * nd, 4.5), sharex=False)
    if nd == 1:
        axes = [axes]
    for t, ax in enumerate(axes):
        data = [Z[y == i][:, di[t]] for i in idxs]
        ax.violinplot(data, showmeans=True)
        ax.set_xticks(range(1, len(idxs) + 1))
        ax.set_xticklabels([f"{i}" for i in idxs], fontsize=8)
        ax.set_title(drivers[t], fontsize=10)
        ax.set_xlabel("persona idx", fontsize=8)
        if t == 0:
            ax.set_ylabel("z-score (vs pool)", fontsize=9)
    fig.suptitle("Driver-metric distributions per persona (z-scored)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "behavioral_drivers.png", dpi=130)
    plt.close(fig)


def _render_md(rep: dict) -> str:
    s = rep["summary"]
    rows = rep["personas"]
    F1 = rep["overlap_matrix"]
    L = []
    L.append("# Persona behavioral diversity (naturalness-manifold overlap)\n")
    L.append("## Config")
    L.append(f"- space: **{s['space']}** (standardize + PCA fit on human manifold)")
    L.append(
        f"- metrics: 19 sim2real rule features; mode **{s['mode']}**; k-NN k=**{s['k']}**"
    )
    L.append("- naturalness = frac of a persona's user turns inside the HUMAN manifold")
    L.append("- overlap = F1 of geometric k-NN manifold of two persona clouds\n")
    L.append("## Summary")
    L.append(f"- personas scored: **{s['n_personas']}**")
    L.append(
        f"- pairwise behavioral overlap: mean **{s['pairwise_overlap_mean']}**, "
        f"min **{s['pairwise_overlap_min']}**, max **{s['pairwise_overlap_max']}**"
    )
    L.append(
        "  (low = behaviorally distinct personas; high = textually varied but "
        "behave the same)\n"
    )
    L.append("## Per persona")
    L.append(
        "| idx | title | n | naturalness | coverage | nearest beh. neighbor (overlap) |"
    )
    L.append(
        "|----|-------|---|-------------|----------|--------------------------------|"
    )
    for r in rows:
        nat = f"{r['naturalness']:.3f}" if r["naturalness"] is not None else "—"
        cov = f"{r['coverage']:.3f}" if r["coverage"] is not None else "—"
        if r["nn_idx"] is not None:
            nn = f"idx{r['nn_idx']} {r['nn_title']} ({r['nn_overlap']:.3f})"
        else:
            nn = "—"
        L.append(f"| idx{r['idx']} | {r['title']} | {r['n']} | {nat} | {cov} | {nn} |")

    sep = rep.get("separability")
    if sep:
        L.append("\n## Does persona text drive distinct behavior distributions?\n")
        L.append(
            f"- **silhouette** (persona labels, 19-D z-space): **{sep['silhouette']}** "
            "(>0 distinct clouds; ≤0 distributions overlap / collapsed)"
        )
        L.append(
            f"- **logreg attribution accuracy**: **{sep['logreg_cv_accuracy']}** "
            f"(chance **{sep['chance_accuracy']}**) — how often a trajectory is "
            "correctly traced to its persona"
        )
        L.append(
            "\n**Which behaviors the persona text controls** (η² = variance explained "
            "by which persona):\n"
        )
        L.append("| metric | η² |")
        L.append("|--------|----|")
        for k, v in list(sep["eta2_by_metric"].items())[:8]:
            L.append(f"| {k} | {v} |")
        L.append(
            "\n**Per-persona signature on the driver metrics** (z vs pool mean):\n"
        )
        drivers = sep["driver_metrics"]
        L.append("| idx | " + " | ".join(drivers) + " |")
        L.append("|----|" + "|".join("----" for _ in drivers) + "|")
        prof = sep["zprofile_on_drivers"]
        for r in rows:
            p = prof.get(str(r["idx"])) or prof.get(r["idx"])
            if not p:
                continue
            L.append(
                f"| idx{r['idx']} | "
                + " | ".join(f"{p[d]:+.2f}" for d in drivers)
                + " |"
            )
        if sep.get("plots"):
            L.append("\nPlots: " + ", ".join(f"`{p}`" for p in sep["plots"]))

    L.append("\n## Pairwise behavioral overlap (F1)\n")
    idxs = [r["idx"] for r in rows]
    head = "      " + "".join(f"idx{i:<3d}" for i in idxs)
    L.append("```")
    L.append(head)
    for i, ri in enumerate(rows):
        cells = ""
        for j in range(len(rows)):
            v = F1[i][j]
            cells += f"  {v:.2f}" if isinstance(v, (int, float)) else "    — "
        L.append(f"idx{ri['idx']:<3d}{cells}")
    L.append("```")
    return "\n".join(L)


if __name__ == "__main__":
    main()
