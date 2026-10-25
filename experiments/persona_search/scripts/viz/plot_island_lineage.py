"""Plot island lineage trees (brinks_01 .. brinks_04).

Reads lineage_tree.json + old_run_backup for parent/child relationships,
val success rate from full_diff eval results.  One subplot per island.

Usage:
    cd experiments/persona_search/scripts/viz
    python plot_island_lineage.py
    # -> data/persona_opt/figs/island_lineage_trees.png
"""

from __future__ import annotations

import json
import re
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import os

import _style as st
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[4].parent / "tau2bench-data"),
    )
)
PGEPA = REPO / "data/persona_opt/pgepa_runs/islands_telecom_gpt54user"
SIMS = REPO / "data/simulations/gepa_opt"


def _pool_from_lineage(island: str) -> list[dict]:
    lt = json.loads((PGEPA / f"g54_{island}/lineage_tree.json").read_text())
    pool = lt.get("pool", {})
    entries = []
    for _k, v in pool.items():
        entries.append(
            {
                "idx": v["idx"],
                "gen": v["gen"],
                "parent_idx": v.get("parent_idx"),
                "title": v.get("title", ""),
            }
        )
    return sorted(entries, key=lambda e: e["idx"])


def _parent_map_from_backup(island: str) -> dict[int, int | None]:
    path = PGEPA / f"g54_{island}/old_run_backup/lineage_from_log.json"
    lineage = json.loads(path.read_text())
    pm: dict[int, int | None] = {0: None}
    for e in lineage:
        if e.get("accepted"):
            pm[e["child_idx"]] = e["parent_idx"]
    return pm


def _gen_map_from_backup(island: str) -> dict[int, str]:
    path = PGEPA / f"g54_{island}/old_run_backup/idx_to_gen_map.json"
    m = json.loads(path.read_text())
    gm: dict[int, str] = {0: "g00"}
    for idx_str, info in m.items():
        idx = int(idx_str)
        if isinstance(info, dict) and "gen_file" in info:
            gm[idx] = info["gen_file"]
    return gm


def _title_from_backup_md(island: str, gen_num: int) -> str:
    fs = glob(str(PGEPA / f"g54_{island}/old_run_backup/g{gen_num:02d}.md"))
    if not fs:
        fs = glob(str(PGEPA / f"g54_{island}/old_run_backup/g{gen_num}.md"))
    if fs:
        line = Path(fs[0]).read_text().split("\n")[0]
        return line.replace("# Persona: ", "").split("(")[0].strip()
    return ""


def _resume_parents(island: str) -> dict[int, int]:
    pm = {}
    for logname in [f"g54_{island}.log", f"g54_{island}_resume.log"]:
        path = PGEPA / f"logs/{logname}"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            m = re.search(r"parent=idx(\d+).*-> idx(\d+)", line)
            if m:
                pm[int(m.group(2))] = int(m.group(1))
    return pm


def _val_succ(island: str, idx: int, gen: str) -> float | None:
    if gen == "g00":
        fs = sorted(
            glob(str(SIMS / f"g54_{island}_g00_seed/eval_*/telecom/results.json"))
        )
    else:
        fs = sorted(
            glob(
                str(
                    SIMS
                    / f"g54_{island}_{gen}_k00_full_diff/eval_*/telecom/results.json"
                )
            )
        )
    if not fs:
        return None
    sims = json.loads(Path(fs[-1]).read_text()).get("simulations", [])
    r = [
        x
        for s in sims
        for x in [((s.get("reward_info") or {}).get("reward"))]
        if x is not None
    ]
    return round(sum(r) / len(r), 3) if r else None


def load_island(island: str) -> list[dict]:
    has_backup = (PGEPA / f"g54_{island}/old_run_backup").exists()

    if has_backup:
        gm = _gen_map_from_backup(island)
        pm = _parent_map_from_backup(island)
        lt = json.loads((PGEPA / f"g54_{island}/lineage_tree.json").read_text())
        pool_lt = lt.get("pool", {})
        # merge: pool_lt has titles for lower idx, backup has gen mapping
        entries = []
        for idx in sorted(gm.keys()):
            gen = gm[idx]
            parent = pm.get(idx)
            # title: try pool first, then backup md
            title = ""
            for _k, v in pool_lt.items():
                if v.get("idx") == idx:
                    title = v.get("title", "")
                    break
            if not title:
                gen_num = int(gen[1:]) if gen != "g00" else 0
                title = _title_from_backup_md(island, gen_num)
            if not title:
                personas = lt.get("personas", {})
                key = f"idx{idx}"
                if key in personas:
                    t = personas[key]
                    title = (
                        t.split("\n")[0]
                        .replace("# Persona: ", "")
                        .split("(")[0]
                        .strip()
                    )
            vd = _val_succ(island, idx, gen)
            entries.append(
                {
                    "idx": idx,
                    "gen": int(gen[1:]),
                    "parent_idx": parent,
                    "title": title,
                    "val_succ": vd,
                }
            )
        # check for resume-added entries not in backup
        rp = _resume_parents(island)
        for idx, parent in rp.items():
            if idx not in gm:
                lt_personas = lt.get("personas", {})
                key = f"idx{idx}"
                title = ""
                if key in lt_personas:
                    t = lt_personas[key]
                    title = (
                        t.split("\n")[0]
                        .replace("# Persona: ", "")
                        .split("(")[0]
                        .strip()
                    )
                # find gen from log
                gen_num = None
                for logname in [f"g54_{island}.log", f"g54_{island}_resume.log"]:
                    lp = PGEPA / f"logs/{logname}"
                    if not lp.exists():
                        continue
                    for line in lp.read_text().splitlines():
                        m = re.search(rf"-> idx{idx}\b", line)
                        if m:
                            m2 = re.search(r"\[iter (\d+)\]", line)
                            if m2:
                                gen_num = int(m2.group(1))
                gen_str = f"g{gen_num:02d}" if gen_num else "g??"
                vd = _val_succ(island, idx, gen_str) if gen_num else None
                entries.append(
                    {
                        "idx": idx,
                        "gen": gen_num or 0,
                        "parent_idx": parent,
                        "title": title,
                        "val_succ": vd,
                    }
                )
    else:
        pool = _pool_from_lineage(island)
        rp = _resume_parents(island)
        entries = []
        for e in pool:
            parent = e["parent_idx"]
            gen_str = f"g{e['gen']:02d}"
            vd = _val_succ(island, e["idx"], gen_str)
            entries.append(
                {
                    "idx": e["idx"],
                    "gen": e["gen"],
                    "parent_idx": parent,
                    "title": e["title"],
                    "val_succ": vd,
                }
            )
        # resume additions
        lt = json.loads((PGEPA / f"g54_{island}/lineage_tree.json").read_text())
        known_idx = {e["idx"] for e in entries}
        for idx, parent in rp.items():
            if idx not in known_idx:
                personas = lt.get("personas", {})
                key = f"idx{idx}"
                title = ""
                if key in personas:
                    t = personas[key]
                    title = (
                        t.split("\n")[0]
                        .replace("# Persona: ", "")
                        .split("(")[0]
                        .strip()
                    )
                gen_num = None
                for logname in [f"g54_{island}.log", f"g54_{island}_resume.log"]:
                    lp = PGEPA / f"logs/{logname}"
                    if not lp.exists():
                        continue
                    for line in lp.read_text().splitlines():
                        m = re.search(rf"-> idx{idx}\b", line)
                        if m:
                            m2 = re.search(r"\[iter (\d+)\]", line)
                            if m2:
                                gen_num = int(m2.group(1))
                gen_str = f"g{gen_num:02d}" if gen_num else "g??"
                vd = _val_succ(island, idx, gen_str) if gen_num else None
                entries.append(
                    {
                        "idx": idx,
                        "gen": gen_num or 0,
                        "parent_idx": parent,
                        "title": title,
                        "val_succ": vd,
                    }
                )

    return sorted(entries, key=lambda e: e["idx"])


# ----------- best / valtop markers -----------
ISLAND_TEST_BEST = {
    "brinks_01": 2,  # idx2
    "brinks_02": 10,  # idx10
    "brinks_03": 8,  # idx8
    "brinks_04": 4,  # idx4
}
ISLAND_VALTOP5 = {
    "brinks_01": 5,  # idx5
    "brinks_02": 8,  # idx8
}


def plot_island(ax, island: str, entries: list[dict]):
    # build tree
    children: dict[int, list[int]] = {}
    nodes = {e["idx"]: e for e in entries}
    for e in entries:
        pid = e["parent_idx"]
        if pid is not None:
            children.setdefault(pid, []).append(e["idx"])

    # layout: BFS, x = horizontal slot, y = -depth
    depth: dict[int, int] = {}
    x_slot: dict[int, int] = {}
    slot = [0]

    def assign(idx, d):
        depth[idx] = d
        kids = sorted(children.get(idx, []))
        if not kids:
            x_slot[idx] = slot[0]
            slot[0] += 1
        else:
            for k in kids:
                assign(k, d + 1)
            x_slot[idx] = (x_slot[kids[0]] + x_slot[kids[-1]]) / 2

    root = [e["idx"] for e in entries if e["parent_idx"] is None]
    for r in root:
        assign(r, 0)

    pos = {idx: (x_slot[idx], -depth[idx]) for idx in x_slot}

    # colormap
    norm = mcolors.Normalize(vmin=0.4, vmax=1.0)
    cmap = matplotlib.colormaps["RdYlGn"]

    # edges
    for e in entries:
        pid = e["parent_idx"]
        if pid is not None and pid in pos:
            x0, y0 = pos[pid]
            x1, y1 = pos[e["idx"]]
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#888",
                    lw=1.5,
                    connectionstyle="arc3,rad=0.1",
                ),
            )

    # nodes
    test_best_idx = ISLAND_TEST_BEST.get(island)
    valtop_idx = ISLAND_VALTOP5.get(island)

    for e in entries:
        idx = e["idx"]
        x, y = pos[idx]
        vd = e["val_succ"] or 0
        color = cmap(norm(vd))

        # marker for special nodes
        edgecolor = "black"
        linewidth = 1.2
        size = 700
        if idx == test_best_idx:
            edgecolor = "#d1495b"
            linewidth = 3
            size = 900
        elif idx == valtop_idx:
            edgecolor = "#1f77b4"
            linewidth = 3
            size = 900
        elif idx == 0:
            edgecolor = "#2e7d32"
            linewidth = 2.5
            size = 800

        ax.scatter(
            [x],
            [y],
            s=size,
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=3,
        )

        # node label
        ax.text(
            x,
            y + 0.02,
            f"idx{idx}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=8,
            zorder=4,
        )

        # difficulty below
        ax.text(
            x,
            y - 0.32,
            f"{vd:.2f}",
            ha="center",
            va="top",
            fontsize=8,
            color="#333",
            fontweight="bold",
        )

        # title below difficulty (only for seed and special nodes)
        if idx == 0 or idx == test_best_idx or idx == valtop_idx:
            title = e.get("title", "")
            if len(title) > 28:
                title = title[:26] + "…"
            ax.text(
                x,
                y - 0.50,
                title,
                ha="center",
                va="top",
                fontsize=6,
                color="#666",
                style="italic",
            )

    ax.set_yticks([])
    ax.set_xticks([])
    seed_title = nodes[0]["title"] if 0 in nodes else island
    n_accepted = len(entries)
    ax.set_title(
        f"{island}\nseed: {seed_title}  ({n_accepted} accepted)", fontsize=10, pad=8
    )
    ax.margins(0.2)


def main():
    st.apply_style()
    islands = ["brinks_01", "brinks_02", "brinks_03", "brinks_04"]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    for ax, island in zip(axes, islands):
        entries = load_island(island)
        plot_island(ax, island, entries)

    # legend
    legend_elements = [
        mpatches.Patch(
            edgecolor="#2e7d32", facecolor="none", linewidth=2.5, label="seed"
        ),
        mpatches.Patch(
            edgecolor="#d1495b",
            facecolor="none",
            linewidth=3,
            label="test best (Pareto)",
        ),
        mpatches.Patch(
            edgecolor="#1f77b4", facecolor="none", linewidth=3, label="val-top5"
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=True,
        edgecolor="#ccc",
        fancybox=True,
    )

    # colorbar
    norm = mcolors.Normalize(vmin=0.4, vmax=1.0)
    sm = cm.ScalarMappable(norm=norm, cmap=matplotlib.colormaps["RdYlGn"])
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("val success rate", fontsize=10)

    fig.suptitle(
        "GEPA Island Lineage Trees (brinks_01 – brinks_04)\nval success rate (30 tasks) · gpt-5.4 agent+user",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.15, right=0.90, top=0.90, bottom=0.06)

    out = REPO / "data/persona_opt/figs/island_lineage_trees.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=st.DPI)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
