#!/usr/bin/env python3
"""Task-level root-cause analysis: correlate per-task STRUCTURAL features (a priori,
defined from task spec + environment, not from agent success) with per-task OPT
EFFECT (how much the optimized persona reduced success vs seed).

Methodology:
  - opt_effect per task = seed_asr - opt_asr  (positive = opt hurt this task)
  - baseline_difficulty = 1 - no_persona_asr  (control variable: hard tasks have
    no headroom for opt to push down -> mechanical confound, MUST partial out)
  - features must vary across tasks WITHIN a domain (domain constants like
    "has user tools" can't explain within-domain opt_effect variance)

Dataset:
  telecom/retail/banking: fullset (seed/opt + shared_no_persona)
  airline: v5 cert (alaska_01_v5, alaska_03_v5) for seed/opt + fullset no_persona
           (fullset alaska_00/01 are stale pre-v5 personas, opt_effect ~1pp;
            v5 gives the ~10pp the certified latest run shows)

Usage:
  .venv/bin/python experiments/persona_search/scripts/task_level_opt_corr.py build   # dump dataset
  .venv/bin/python experiments/persona_search/scripts/task_level_opt_corr.py corr    # correlations
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
os.chdir(REPO)
FULL = REPO / "data/simulations/fullset"
SPEC = REPO / "datasets/tau2/domains"
OUT = REPO / "data/persona_opt/task_level_rootcause.json"

# ---- persona/cell config ----
PERSONAS = {
    "airline": [
        "alaska_00",
        "alaska_01",
    ],  # fullset (used only for no_persona baseline)
    "retail": ["gc_02", "gc_03", "gc_04"],
    "banking": ["cw_03", "cw_04"],
    "telecom": ["brinks_00", "brinks_02", "brinks_03", "brinks_04", "p0003"],
}
# airline seed/opt come from v5 cert instead of fullset
AIRLINE_V5 = {
    "seed": [
        str(DATA_ROOT / "data/simulations/cert_alaska01_v5/seed"),
        str(DATA_ROOT / "data/simulations/cert_alaska03_v5/seed"),
    ],
    "opt": [
        str(DATA_ROOT / "data/simulations/cert_alaska01_v5/optim"),
        str(DATA_ROOT / "data/simulations/cert_alaska03_v5/optim"),
    ],
}

TASK_SPEC = {
    "airline": SPEC / "airline/tasks.json",
    "retail": SPEC / "retail/tasks.json",
    "telecom": SPEC / "telecom/tasks.json",
    "banking": SPEC / "banking_knowledge/tasks.json",
}

# irreversible / state-changing action name patterns (domain-aware)
IRREV_PATTERNS = {
    "airline": ["cancel_", "book_", "update_reservation", "transfer_to_human"],
    "retail": ["return_", "exchange_", "cancel_", "modify_"],
    "banking": [
        "apply_for_",
        "submit_",
        "change_",
        "request_human",
        "transfer_to_human",
    ],
    "telecom": [
        "refuel_data"
    ],  # toggles are reversible; data purchase is the one irrev
}


def base_tid(task_id: str, domain: str) -> str:
    """Strip [PERSONA:...] suffix to get canonical base task id."""
    s = re.sub(r"\[PERSONA:[^\]]*\]", "", str(task_id)).strip()
    return s


def cell_path(domain, persona, cond):
    if persona == "p0003":
        return FULL / "telecom_p0003" / f"p0003_{cond}"
    return FULL / domain / f"{persona}_{cond}"


def load_sims(path):
    rp = Path(path) / "results.json"
    if not rp.exists():
        return []
    r = json.loads(rp.read_text())
    return r.get("simulations", [])


def rewards_by_basetask(sims, domain):
    """Return dict base_tid -> list of reward (0/1)."""
    out = defaultdict(list)
    for s in sims:
        bt = base_tid(s.get("task_id", ""), domain)
        if not bt:
            continue
        rw = s.get("reward_info", {}).get("reward", 0)
        out[bt].append(1 if rw and rw > 0 else 0)
    return out


def asr(rewards):
    return sum(rewards) / len(rewards) if rewards else None


# ---------------- feature extraction ----------------


def load_tasks(domain):
    d = json.loads(TASK_SPEC[domain].read_text())
    return d if isinstance(d, list) else d.get("tasks", d)


def count_faults_telecom(base_id):
    """base_id like '[mobile_data_issue]airplane_mode_on|bad_network_preference'."""
    m = re.match(r"\[([^\]]+)\](.*)", base_id)
    if not m:
        return 1, base_id
    cat, rest = m.group(1), m.group(2)
    faults = [f for f in rest.split("|") if f.strip()]
    return len(faults), cat


def extract_features(domain, task):
    ec = task.get("evaluation_criteria") or {}
    actions = ec.get("actions") or []
    action_names = [a.get("name", "") for a in actions if isinstance(a, dict)]
    f = {}
    f["num_required_actions"] = len(actions)
    f["num_nl_assertions"] = len(ec.get("nl_assertions") or [])
    f["num_communicate_info"] = len(ec.get("communicate_info") or [])
    # irreversible
    pats = IRREV_PATTERNS.get(domain, [])
    f["has_irreversible"] = int(any(any(p in n for p in pats) for n in action_names))
    f["num_irreversible_actions"] = sum(
        1 for n in action_names if any(p in n for p in pats)
    )
    # unique action count (procedural diversity)
    f["num_unique_actions"] = len(set(action_names))
    # instruction complexity
    instr = task.get("user_scenario", {}).get("instructions", {}) or {}
    if isinstance(instr, dict):
        rf = (
            (instr.get("reason_for_call") or "")
            + " "
            + (instr.get("task_instructions") or "")
        )
        f["instruction_chars"] = len(rf)
        unk = instr.get("unknown_info")
        f["num_unknown_info"] = len(unk) if isinstance(unk, list) else (1 if unk else 0)
    else:
        f["instruction_chars"] = len(str(instr))
        f["num_unknown_info"] = 0
    # reward basis
    rb = ec.get("reward_basis") or []
    f["reward_db"] = int("DB" in rb)
    f["reward_communicate"] = int("COMMUNICATE" in rb or "NL_ASSERTION" in rb)
    f["reward_action"] = int("ACTION" in rb)
    f["reward_env"] = int("ENV_ASSERTION" in rb)

    # domain-specific
    if domain == "telecom":
        nf, cat = count_faults_telecom(task.get("id", ""))
        f["num_faults"] = nf
        f["fault_category"] = cat
        # diagnostic-action count: toggles / set / reboot / reset / reseat (the fix actions)
        diag = {
            "toggle_",
            "set_network",
            "enable_roaming",
            "reboot_",
            "reset_",
            "reseat_",
            "refuel_",
            "grant_app",
        }
        f["num_diagnostic_actions"] = sum(
            1 for n in action_names if any(d in n for d in diag)
        )
    if domain == "banking":
        ut = task.get("user_tools")
        f["has_user_tools"] = int(bool(ut))
        f["num_required_documents"] = len(task.get("required_documents") or [])
        # discoverable / gated tool mechanism
        f["num_discoverable_calls"] = sum(
            1 for n in action_names if "discoverable" in n
        )
        f["num_unlock_calls"] = sum(1 for n in action_names if "unlock" in n)
    if domain == "airline":
        # task type from purpose
        pur = (task.get("description", {}) or {}).get("purpose") or ""
        pl = pur.lower()
        if "cancel" in pl:
            f["task_type"] = "cancel"
        elif "refund" in pl:
            f["task_type"] = "refund"
        elif "exchange" in pl or "modify" in pl or "update" in pl or "change" in pl:
            f["task_type"] = "modify"
        elif "book" in pl:
            f["task_type"] = "book"
        else:
            f["task_type"] = "other"
    if domain == "retail":
        rf = (
            (task.get("user_scenario", {}).get("instructions", {}) or {}).get(
                "reason_for_call"
            )
            or ""
        ).lower()
        f["has_exchange"] = int("exchange" in rf)
        f["has_return"] = int("return" in rf)
        f["has_cancel"] = int("cancel" in rf)
        # count items via action args get_product_details count
        f["num_products"] = sum(1 for n in action_names if n == "get_product_details")
    return f


# ---------------- build dataset ----------------


def build():
    rows = []
    for domain in ["telecom", "retail", "banking", "airline"]:
        # aggregate rewards per base task per cond
        seed_r = defaultdict(list)
        opt_r = defaultdict(list)
        np_r = defaultdict(list)
        if domain == "airline":
            for p in AIRLINE_V5["seed"]:
                for bt, rws in rewards_by_basetask(load_sims(p), domain).items():
                    seed_r[bt].extend(rws)
            for p in AIRLINE_V5["opt"]:
                for bt, rws in rewards_by_basetask(load_sims(p), domain).items():
                    opt_r[bt].extend(rws)
        else:
            for persona in PERSONAS[domain]:
                for bt, rws in rewards_by_basetask(
                    load_sims(cell_path(domain, persona, "seed")), domain
                ).items():
                    seed_r[bt].extend(rws)
                for bt, rws in rewards_by_basetask(
                    load_sims(cell_path(domain, persona, "opt")), domain
                ).items():
                    opt_r[bt].extend(rws)
        # no_persona baseline (fullset for all)
        npdir = FULL / domain / "shared_no_persona"
        for bt, rws in rewards_by_basetask(load_sims(npdir), domain).items():
            np_r[bt].extend(rws)

        # task specs -> features keyed by base id
        feats = {}
        for t in load_tasks(domain):
            bt = base_tid(t.get("id", ""), domain)
            feats[bt] = extract_features(domain, t)

        # union of base tasks seen in seed/opt
        base_tasks = sorted(set(seed_r) | set(opt_r))
        for bt in base_tasks:
            sr = asr(seed_r.get(bt, []))
            orb = asr(opt_r.get(bt, []))
            npr = asr(np_r.get(bt, []))
            if sr is None or orb is None:
                continue
            row = {
                "domain": domain,
                "task_id": bt,
                "seed_asr": sr,
                "opt_asr": orb,
                "np_asr": npr,
                "opt_effect": sr - orb,  # positive = opt hurt
                "opt_effect_rel": (sr - orb) / sr if sr > 0 else None,
                "baseline_difficulty": (1 - npr) if npr is not None else None,
                "headroom": sr,  # headroom = seed success (room to push down)
                "n_seed": len(seed_r.get(bt, [])),
                "n_opt": len(opt_r.get(bt, [])),
                "n_np": len(np_r.get(bt, [])),
            }
            row.update(feats.get(bt, {}))
            rows.append(row)
        print(
            f"{domain}: {len(base_tasks)} base tasks, "
            f"seed_asr={sum(sr for r in [asr(seed_r[b]) for b in base_tasks] if r for sr in [r]) / len(base_tasks):.3f} | "
            f"features matched {sum(1 for b in base_tasks if b in feats)}/{len(base_tasks)}"
        )
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\nwrote {len(rows)} rows -> {OUT}")
    return rows


# ---------------- correlation analysis ----------------

from scipy import stats as _scistats


def spearman(xs, ys):
    if len(xs) < 5:
        return None, None
    r, p = _scistats.spearmanr(xs, ys)
    if math.isnan(r):
        return None, None
    return float(r), float(p)


def pearson(xs, ys):
    if len(xs) < 5:
        return None, None
    r, p = _scistats.pearsonr(xs, ys)
    if math.isnan(r):
        return None, None
    return float(r), float(p)


def partial_corr(xs, ys, zs):
    """Partial Pearson correlation of xs,ys controlling for zs (single covariate)."""
    rx_z = _resid(xs, zs)
    ry_z = _resid(ys, zs)
    return pearson(rx_z, ry_z)


def _resid(a, b):
    n = len(a)
    mb = sum(b) / n
    ma = sum(a) / n
    sbb = sum((bi - mb) ** 2 for bi in b)
    if sbb == 0:
        return [ai - ma for ai in a]
    sab = sum((b[i] - mb) * (a[i] - ma) for i in range(n))
    beta = sab / sbb
    return [a[i] - (ma + beta * (b[i] - mb)) for i in range(n)]


# features to test (numeric, vary within domain)
NUMERIC_FEATS = [
    "num_required_actions",
    "num_unique_actions",
    "num_irreversible_actions",
    "has_irreversible",
    "num_nl_assertions",
    "num_communicate_info",
    "instruction_chars",
    "num_unknown_info",
    "num_faults",
    "num_diagnostic_actions",
    "has_user_tools",
    "num_required_documents",
    "num_discoverable_calls",
    "num_unlock_calls",
    "num_products",
    "has_exchange",
    "has_return",
    "has_cancel",
]


def corr():
    rows = json.loads(OUT.read_text())
    bydom = defaultdict(list)
    for r in rows:
        bydom[r["domain"]].append(r)

    print("=" * 80)
    print(
        "TASK-LEVEL: feature vs opt_effect (seed_asr - opt_asr), controlling baseline_difficulty"
    )
    print("opt_effect > 0 means the optimized persona HURT this task")
    print("=" * 80)

    # per-domain raw + partial (control baseline_difficulty)
    for dom in ["telecom", "retail", "banking", "airline"]:
        rs = bydom[dom]
        # filter rows with baseline_difficulty present
        rs = [r for r in rs if r.get("baseline_difficulty") is not None]
        n = len(rs)
        print(f"\n### {dom} (n={n}) ###")
        print(
            f"  mean opt_effect={sum(r['opt_effect'] for r in rs) / n:.3f}, "
            f"mean baseline_diff={sum(r['baseline_difficulty'] for r in rs) / n:.3f}"
        )
        print(
            f"  {'feature':28s} {'rho':>6s} {'p':>7s} {'r_partial*':>10s} {'p_part':>7s} {'mean':>6s}"
        )
        for feat in NUMERIC_FEATS:
            vals = [
                (r.get(feat), r["opt_effect"], r["baseline_difficulty"])
                for r in rs
                if r.get(feat) is not None and isinstance(r.get(feat), (int, float))
            ]
            if len(vals) < 8:
                continue
            xs = [v[0] for v in vals]
            ys = [v[1] for v in vals]
            zs = [v[2] for v in vals]
            rho, p = spearman(xs, ys)
            rp, pp = partial_corr(xs, ys, zs)
            mean = sum(xs) / len(xs)

            def f(v, w=6):
                return f"{v:{w}.3f}" if v is not None else f"{'--':{w}s}"

            print(f"  {feat:28s} {f(rho)} {f(p, 7)} {f(rp, 10)} {f(pp, 7)} {mean:6.2f}")

    # POOLED with domain fixed effects: de-mean opt_effect and features within domain,
    # then correlate (controls domain membership). Also partial baseline_difficulty.
    print(f"\n### POOLED (within-domain de-meaned + control baseline_difficulty) ###")
    print(
        f"  {'feature':28s} {'rho_within':>10s} {'p':>7s} {'r_partial':>10s} {'p_part':>7s} {'n':>4s}"
    )
    for feat in NUMERIC_FEATS:
        triples = []
        for dom, rs in bydom.items():
            rsf = [
                r
                for r in rs
                if r.get(feat) is not None
                and r.get("baseline_difficulty") is not None
                and isinstance(r.get(feat), (int, float))
            ]
            if len(rsf) < 5:
                continue
            xs = [r[feat] for r in rsf]
            ys = [r["opt_effect"] for r in rsf]
            zs = [r["baseline_difficulty"] for r in rsf]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            mz = sum(zs) / len(zs)
            for i in range(len(xs)):
                triples.append((xs[i] - mx, ys[i] - my, zs[i] - mz, dom))
        if len(triples) < 20:
            continue
        xs = [t[0] for t in triples]
        ys = [t[1] for t in triples]
        zs = [t[2] for t in triples]
        rho, p = spearman(xs, ys)
        rp, pp = partial_corr(xs, ys, zs)

        def f(v, w=10):
            return f"{v:{w}.3f}" if v is not None else f"{'--':{w}s}"

        print(f"  {feat:28s} {f(rho)} {f(p, 7)} {f(rp)} {f(pp, 7)} {len(triples):4d}")

    # also: opt_effect vs baseline_difficulty (the confound itself) and vs headroom
    print(
        f"\n### confound check: opt_effect vs baseline_difficulty / headroom (seed_asr) ###"
    )
    for dom in ["telecom", "retail", "banking", "airline"]:
        rs = [r for r in bydom[dom] if r.get("baseline_difficulty") is not None]
        if len(rs) < 5:
            continue
        xs = [r["baseline_difficulty"] for r in rs]
        ys = [r["opt_effect"] for r in rs]
        rho, p = spearman(xs, ys)
        hs = [r["headroom"] for r in rs]
        rh, ph = spearman(hs, ys)

        def f(v):
            return f"{v:.3f}" if v is not None else "--"

        print(
            f"  {dom:9s}: opt_effect~baseline_diff rho={f(rho)} (p={f(p)}) | "
            f"opt_effect~headroom(seed_asr) rho={f(rh)} (p={f(ph)})"
        )


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "corr"])
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    else:
        corr()


if __name__ == "__main__":
    main()
