"""Behavior-diversity of persona sets via Jensen-Shannon divergence on the
Sim2Real 19 behavioral metrics (D1-D4).

Goal (option "b"): show that persona OPTIMIZATION widens behavioral coverage --
i.e. the mean PAIRWISE JS distance among the optimized personas is larger than
among the seed personas. Behavior is measured, not prompt wording.

Method
------
1. For each persona we collect its rollouts (tau2 results.json sims). For every
   conversation we compute the 19 atomic metrics (reused verbatim from
   sim2real_metrics.metrics_for_interaction) on the USER turns -> one 19-dim
   sample per conversation. A persona is thus a SAMPLE of behavior vectors.
2. For each metric we bin its values into shared quantile bins (edges computed
   over the pooled values of ALL personas, so histograms are comparable), then
   build a Laplace-smoothed per-persona histogram = that persona's marginal
   distribution for the metric.
3. Pairwise behavioral distance between persona i and j = mean over metrics of
   the per-metric Jensen-Shannon DISTANCE (scipy, base 2, range [0,1]). We also
   break it down per dimension (D1-D4). Per-metric marginals (not a 19-D joint)
   are used on purpose: with ~25-50 rollouts a joint density is unestimable, the
   marginal-averaged JSD is the robust, standard choice.
4. Report: full pairwise matrix (overall + per-dim), a heatmap, and the headline
   within-group means -- mean pairwise JS among seeds vs among optimized.

Task confound: every persona here is evaluated on the SAME val task set with the
SAME #trials, so the task mix is held fixed at the set level; differences in the
distributions reflect persona behavior, not task mix.

Usage
-----
  # explicit personas: --persona LABEL GROUP "GLOB_OR_DIR"  (repeatable)
  python persona_behavior_jsd.py \
      --persona brinks_01 seed   "data/simulations/gepa_opt/xdom6_g00_seed1_val" \
      --persona brinks_02 seed   "data/simulations/gepa_opt/xdom6_g00_seed2_val" \
      --persona opt_g04   optim  "data/simulations/gepa_opt/xdom7_g04_k00_full_diff" \
      --out-prefix data/persona_opt/jsd_v7

  # auto mode: read a pgepa run + its seed-sim source, build specs automatically
  python persona_behavior_jsd.py \
      --pgepa-run   data/persona_opt/pgepa_runs/xdom_v7_3domain \
      --seed-sims   data/simulations/gepa_opt/xdom6_g00_seed{i}_val \
      --opt-sims    data/simulations/gepa_opt/xdom7_g{gen:02d}_k00_full_diff \
      --out-prefix  data/persona_opt/jsd_v7
"""

import argparse
import glob
import json

# Reuse the paper-faithful metric implementation verbatim.
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
from sim2real_metrics import METRIC_ORDER, metrics_for_interaction

# 19 metrics grouped by the four Sim2Real dimensions.
DIMENSIONS = {
    "D1_style": [
        "Wds/trn",
        "Short%",
        "Polite%",
        "Formal%",
        "Ack%",
        "VerbCV",
        "Repeat%",
        "IDConf%",
    ],
    "D2_struct": ["FrntLoad%", "IDs/trn", "Open.Wds"],
    "D3_stance": ["Uncert%", "Certn%", "Pushbk%", "ClarfyQ%", "InfoQ%"],
    "D4_affect": ["Emot%", "Accuse%", "Pivot%"],
}
METRIC_TO_DIM = {m: d for d, ms in DIMENSIONS.items() for m in ms}

_CTRL = re.compile(r"#{2,}[A-Z_]+#{2,}")  # ###STOP### / ###TRANSFER### markers


def user_turns_tau2(messages):
    """USER turns from a tau2 sim, control tokens stripped, empties dropped."""
    out = []
    for m in messages:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if not isinstance(c, str):
            continue
        c = _CTRL.sub(" ", c).strip()
        if c:
            out.append(c)
    return out


def persona_samples(path_glob, min_turns=1):
    """All per-conversation 19-dim metric vectors for one persona."""
    base = Path(path_glob)
    if base.is_dir():
        files = glob.glob(str(base / "**" / "results.json"), recursive=True)
    else:
        files = glob.glob(path_glob, recursive=True)
        files = [f for f in files if f.endswith("results.json")] or files
    samples = []
    n_sims = 0
    for f in files:
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for s in data.get("simulations", []) or []:
            n_sims += 1
            turns = user_turns_tau2(s.get("messages", []) or [])
            if len(turns) < min_turns:
                continue
            m = metrics_for_interaction(turns)
            if m is not None:
                samples.append(m)
    return samples, n_sims


def build_histograms(persona_vecs, bins, alpha):
    """Per (metric, persona) smoothed, normalized histogram on shared edges.

    Returns {metric: {label: prob_vector}} and the set of degenerate metrics
    (no variance across the pool) which are excluded from JS averaging.
    """
    labels = list(persona_vecs)
    hists = {}
    degenerate = []
    for metric in METRIC_ORDER:
        pooled = np.array(
            [v[metric] for lab in labels for v in persona_vecs[lab]], dtype=float
        )
        edges = np.unique(np.quantile(pooled, np.linspace(0, 1, bins + 1)))
        if edges.size < 2:  # metric constant across everyone -> no signal
            degenerate.append(metric)
            continue
        per_label = {}
        for lab in labels:
            vals = np.array([v[metric] for v in persona_vecs[lab]], dtype=float)
            counts, _ = np.histogram(vals, bins=edges)
            p = counts.astype(float) + alpha  # Laplace smoothing
            p /= p.sum()
            per_label[lab] = p
        hists[metric] = per_label
    return hists, degenerate


def pairwise(labels, hists, metrics):
    """Mean JS distance (base 2, [0,1]) over `metrics` for every label pair."""
    n = len(labels)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ds = []
            for metric in metrics:
                if metric not in hists:
                    continue
                d = jensenshannon(
                    hists[metric][labels[i]], hists[metric][labels[j]], base=2
                )
                if not np.isnan(d):
                    ds.append(d)
            v = float(np.mean(ds)) if ds else 0.0
            mat[i, j] = mat[j, i] = v
    return mat


def group_mean(mat, labels, groups, ga, gb=None):
    """Mean of upper-triangle pairs within ga (gb=None) or across ga x gb."""
    idx_a = [i for i, lab in enumerate(labels) if groups[lab] == ga]
    if gb is None:
        vals = [mat[i, j] for a, i in enumerate(idx_a) for j in idx_a[a + 1 :]]
    else:
        idx_b = [i for i, lab in enumerate(labels) if groups[lab] == gb]
        vals = [mat[i, j] for i in idx_a for j in idx_b]
    return float(np.mean(vals)) if vals else float("nan")


def heatmap(mat, labels, groups, title, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(range(len(labels)), key=lambda i: (groups[labels[i]], labels[i]))
    m = mat[np.ix_(order, order)]
    labs = [labels[i] for i in order]
    grps = [groups[labs[k]] for k in range(len(labs))]

    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(labs), 0.8 + 0.6 * len(labs)))
    im = ax.imshow(m, cmap="viridis", vmin=0)
    ax.set_xticks(range(len(labs)))
    ax.set_yticks(range(len(labs)))
    ax.set_xticklabels(
        [f"{l}\n[{g}]" for l, g in zip(labs, grps)], rotation=45, ha="right", fontsize=7
    )
    ax.set_yticklabels([f"{l} [{g}]" for l, g in zip(labs, grps)], fontsize=7)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(
                j,
                i,
                f"{m[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if m[i, j] < m.max() * 0.6 else "black",
                fontsize=6,
            )
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, label="mean JS distance")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def auto_specs(pgepa_run, seed_sims, opt_sims):
    """Build [(label, group, glob)] from a pgepa run + sim-path templates.

    seeds  -> gen0 candidates, sim dir from seed_sims.format(i=<seed pos+1>)
    optim  -> accepted candidates (gen>0, in pool, feasible),
              sim dir from opt_sims.format(gen=<gen>)
    The seed index mapping assumes v7 seeds == v6 seed1..4 (drop seed0).
    """
    state = json.load(open(Path(pgepa_run) / "pgepa_state.json"))
    specs = []
    seed_pos = 0
    for c in state["candidates"]:
        name = c["persona_text"].splitlines()[0].replace("# Persona:", "").strip()
        short = re.sub(r"\s*\(effect.*$", "", name).strip()[:20].replace(" ", "_")
        if c["gen"] == 0:
            seed_pos += 1
            g = seed_sims.format(i=seed_pos)
            specs.append((f"seed{seed_pos}_{short}", "seed", g))
        elif c.get("feasible"):
            g = opt_sims.format(gen=c["gen"])
            specs.append((f"g{c['gen']:02d}_{short}", "optim", g))
    return specs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--persona",
        nargs=3,
        action="append",
        metavar=("LABEL", "GROUP", "GLOB"),
        default=[],
        help="explicit persona spec; repeatable",
    )
    ap.add_argument("--pgepa-run", default=None, help="auto mode: pgepa run dir")
    ap.add_argument(
        "--seed-sims",
        default=str(DATA_ROOT / "data/simulations/gepa_opt/xdom6_g00_seed{i}_val"),
    )
    ap.add_argument(
        "--opt-sims",
        default=str(
            DATA_ROOT / "data/simulations/gepa_opt/xdom7_g{gen:02d}_k00_full_diff"
        ),
    )
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5, help="Laplace smoothing")
    ap.add_argument("--min-turns", type=int, default=1)
    ap.add_argument(
        "--out-prefix", default=str(DATA_ROOT / "data/persona_opt/persona_jsd")
    )
    args = ap.parse_args()

    specs = list(args.persona)
    if args.pgepa_run:
        specs += auto_specs(args.pgepa_run, args.seed_sims, args.opt_sims)
    if len(specs) < 2:
        ap.error("need >=2 personas (via --persona and/or --pgepa-run)")

    # collect samples
    persona_vecs, groups = {}, {}
    print(f"{'persona':28s} {'group':6s} {'#sims':>6s} {'#convs':>7s}")
    for label, group, g in specs:
        samples, n_sims = persona_samples(g, args.min_turns)
        if not samples:
            print(f"  !! {label}: 0 usable conversations from {g}")
            continue
        persona_vecs[label] = samples
        groups[label] = group
        print(f"{label:28s} {group:6s} {n_sims:6d} {len(samples):7d}")
    labels = list(persona_vecs)
    if len(labels) < 2:
        ap.error("fewer than 2 personas had usable data")

    hists, degenerate = build_histograms(persona_vecs, args.bins, args.alpha)
    if degenerate:
        print(f"\n[degenerate metrics excluded (no variance): {', '.join(degenerate)}]")

    # overall + per-dimension pairwise matrices
    overall = pairwise(labels, hists, METRIC_ORDER)
    per_dim = {d: pairwise(labels, hists, ms) for d, ms in DIMENSIONS.items()}

    # headline: within/between group means
    grp_set = sorted(set(groups.values()))
    print("\n=== mean pairwise JS distance (overall, 19 metrics) ===")
    for gr in grp_set:
        print(f"  within {gr:6s}: {group_mean(overall, labels, groups, gr):.4f}")
    if len(grp_set) == 2:
        a, b = grp_set
        print(f"  between {a}-{b}: {group_mean(overall, labels, groups, a, b):.4f}")
        wa = group_mean(overall, labels, groups, a)
        wb = group_mean(overall, labels, groups, b)
        if not (np.isnan(wa) or np.isnan(wb)):
            hi, lo = (b, a) if wb > wa else (a, b)
            print(
                f"  -> '{hi}' set is behaviorally MORE diverse than '{lo}' "
                f"(Δ={abs(wb - wa):.4f})"
            )

    print("\n=== within-group mean JS by dimension ===")
    hdr = "  " + "dim".ljust(12) + "".join(f"{gr:>10s}" for gr in grp_set)
    print(hdr)
    for d, mat in per_dim.items():
        row = (
            "  "
            + d.ljust(12)
            + "".join(f"{group_mean(mat, labels, groups, gr):>10.4f}" for gr in grp_set)
        )
        print(row)

    # dump matrix CSV + heatmaps
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv = [",".join([""] + labels)]
    for i, li in enumerate(labels):
        csv.append(
            ",".join([li] + [f"{overall[i, j]:.4f}" for j in range(len(labels))])
        )
    Path(f"{args.out_prefix}_matrix.csv").write_text("\n".join(csv))
    heatmap(
        overall,
        labels,
        groups,
        "Behavior JS distance (19 metrics)",
        f"{args.out_prefix}_heatmap.png",
    )
    for d, mat in per_dim.items():
        heatmap(
            mat,
            labels,
            groups,
            f"Behavior JS distance ({d})",
            f"{args.out_prefix}_heatmap_{d}.png",
        )

    print(f"\nwrote {args.out_prefix}_matrix.csv")
    print(f"wrote {args.out_prefix}_heatmap.png (+ per-dimension heatmaps)")


if __name__ == "__main__":
    main()
