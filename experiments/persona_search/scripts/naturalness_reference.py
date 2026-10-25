"""Naturalness reference: JSD vs human, z-score table, human conversation samples.

Used by the naturalness optimization loop to score candidate personas against
real human behavioral data (Brinks customer support, 2694 conversations).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

REPO = Path(
    os.environ.get(
        "TAU2BENCH_DATA_ROOT",
        str(Path(__file__).resolve().parents[3].parent / "tau2bench-data"),
    )
)
sys.path.insert(0, str(REPO))
import sys

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
from sim2real_metrics import METRIC_ORDER, metrics_for_interaction  # noqa: E402

BRINKS_PATH = (
    Path(os.environ.get("PERSONA_DATA_DIR", "/Users/yutao.xie/work/python-ai-services"))
    / "virtual_agent"
    / "user_simulator_dataset_v2.json"
)
HUMAN_REF_PATH = REPO / "data" / "persona_opt" / "human_reference_stats.json"
HUMAN_VECS_PATH = REPO / "data" / "persona_opt" / "human_reference_vectors.json"

JSD_BINS = 8
JSD_ALPHA = 0.5

METRIC_EXPLANATIONS = {
    "Wds/trn": "words per turn. Human ~15. Too high = verbose/essay-like. Too low = robotic one-word replies.",
    "Short%": "short turn rate (< 5 words). Human ~29%. Reflects natural mix of brief acks and longer responses.",
    "Polite%": "politeness rate (thanks, please, appreciate, sorry). Human ~17%. Real customers are naturally polite.",
    "Formal%": "formal register ('I am experiencing', 'please advise'). Human ~0%. Real customers speak casually.",
    "Ack%": "pure acknowledgments ('ok', 'got it', 'alright') with no other content. Human ~4%.",
    "VerbCV": "vocabulary diversity across turns. Human ~0.74. Too low = repetitive phrasing.",
    "Repeat%": "verbatim phrase repetition. Human ~0%. Repeating the same sentence is robotic.",
    "IDConf%": "identity confirmation rate. Human ~11%.",
    "FrntLoad%": "info given in first 30% of conversation. Human ~11%. Too high = dumping everything upfront.",
    "IDs/trn": "identifiers (phone #, account #) per turn. Human ~0.12. Too high = unnaturally repeating IDs.",
    "Open.Wds": "first message word count. Human ~1.8 (just 'hi' or account #). Long openers are unnatural.",
    "Uncert%": "uncertainty expressions ('i think', 'not sure'). Human ~12%.",
    "Certn%": "certainty expressions ('looks fine', 'definitely'). Human ~4%.",
    "Pushbk%": "pushback against agent instructions. Human ~0.1%.",
    "ClarfyQ%": "clarifying questions. Human ~0.7%.",
    "InfoQ%": "information questions ('how long?', 'will that fix it?'). Human ~12%.",
    "Emot%": "emotional language ('frustrated', 'annoyed'). Human ~0.7%. Real customers show frustration through brevity.",
    "Accuse%": "accusatory language. Human ~3%. Terse is not the same as hostile.",
    "Pivot%": "topic redirects ('what about voicemail?'). Human ~5%.",
}


def _extract_brinks_turns(conv: dict) -> list[str]:
    turns = []
    for m in conv.get("messages", []):
        if m.get("speaker") == "VISITOR":
            t = m.get("text", "").strip()
            if t:
                turns.append(t)
    return turns


def _render_brinks_conversation(conv: dict, max_turns: int = 20) -> str:
    lines = []
    for m in conv.get("messages", [])[:max_turns]:
        speaker = m.get("speaker", "")
        text = m.get("text", "").strip()
        if not text:
            continue
        role = "CUSTOMER" if speaker == "VISITOR" else "AGENT"
        if len(text) > 200:
            text = text[:200] + "..."
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def build_human_reference(brinks_path: str | Path | None = None) -> None:
    """Build human reference stats + vectors from brinks dataset. Run once."""
    path = Path(brinks_path) if brinks_path else BRINKS_PATH
    convs = json.loads(path.read_text())
    print(f"Loading {len(convs)} brinks conversations...")

    vecs = []
    for conv in convs:
        turns = _extract_brinks_turns(conv)
        if len(turns) >= 2:
            m = metrics_for_interaction(turns)
            if m is not None:
                vecs.append(m)

    print(f"Computed metrics for {len(vecs)} conversations")

    stats = {}
    for metric in METRIC_ORDER:
        vals = [v[metric] for v in vecs]
        stats[metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
        }

    ref = {"source": str(path.name), "n_conversations": len(vecs), "metrics": stats}
    HUMAN_REF_PATH.write_text(json.dumps(ref, indent=2))
    print(f"Wrote {HUMAN_REF_PATH}")

    HUMAN_VECS_PATH.write_text(json.dumps(vecs))
    print(f"Wrote {HUMAN_VECS_PATH} ({len(vecs)} vectors)")


def load_human_ref() -> dict:
    return json.loads(HUMAN_REF_PATH.read_text())


def load_human_vecs() -> list[dict]:
    if not HUMAN_VECS_PATH.exists():
        build_human_reference()
    return json.loads(HUMAN_VECS_PATH.read_text())


def extract_sim_vecs(results_json: dict) -> list[dict]:
    """Extract per-conversation metric vectors from tau2 results."""
    vecs = []
    for s in results_json.get("simulations", []):
        if s.get("termination_reason") == "infra_error" or not s.get("messages"):
            continue
        ut = [
            m["content"]
            for m in s["messages"]
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        if len(ut) >= 2:
            m = metrics_for_interaction(ut)
            if m is not None:
                vecs.append(m)
    return vecs


def compute_profile(vecs: list[dict]) -> dict[str, float]:
    """Mean profile from metric vectors."""
    if not vecs:
        return {m: 0.0 for m in METRIC_ORDER}
    return {m: float(np.mean([v[m] for v in vecs])) for m in METRIC_ORDER}


def compute_jsd_vs_human(
    sim_vecs: list[dict],
    human_vecs: list[dict],
    bins: int = JSD_BINS,
    alpha: float = JSD_ALPHA,
) -> float:
    """Mean per-metric Jensen-Shannon distance between sim and human distributions."""
    if not sim_vecs:
        return 1.0
    jsd_vals = []
    for metric in METRIC_ORDER:
        pooled = np.array(
            [v[metric] for v in sim_vecs] + [v[metric] for v in human_vecs], dtype=float
        )
        edges = np.unique(np.quantile(pooled, np.linspace(0, 1, bins + 1)))
        if edges.size < 2:
            continue
        vals_s = np.array([v[metric] for v in sim_vecs], dtype=float)
        vals_h = np.array([v[metric] for v in human_vecs], dtype=float)
        counts_s, _ = np.histogram(vals_s, bins=edges)
        counts_h, _ = np.histogram(vals_h, bins=edges)
        p_s = counts_s.astype(float) + alpha
        p_s /= p_s.sum()
        p_h = counts_h.astype(float) + alpha
        p_h /= p_h.sum()
        d = jensenshannon(p_s, p_h, base=2)
        if not np.isnan(d):
            jsd_vals.append(d)
    return float(np.mean(jsd_vals)) if jsd_vals else 1.0


def compute_zscore_table(
    profile: dict[str, float], human_ref: dict
) -> list[tuple[str, float, float, float, str, str]]:
    """Per-metric z-score table for mutation prompt.

    Returns list of (metric, persona_val, human_mean, z_score, direction, priority),
    sorted by z descending.
    """
    rows = []
    for metric in METRIC_ORDER:
        h = human_ref["metrics"][metric]
        p = profile.get(metric, 0.0)
        gap = p - h["mean"]
        if h["std"] > 0:
            z = abs(gap) / h["std"]
        else:
            z = min(2.0, 10.0 if abs(gap) > 0.01 else 0.0)
        if metric == "Formal%":
            z = min(z, 2.0)
        direction = "higher" if gap > 0 else "lower"
        if z > 1.5:
            priority = "FIX"
        elif z > 0.8:
            priority = "ADJUST"
        elif z > 0.4:
            priority = "minor"
        else:
            priority = "ok"
        rows.append((metric, p, h["mean"], z, direction, priority))
    rows.sort(key=lambda x: -x[3])
    return rows


def format_zscore_table(rows: list) -> str:
    lines = [
        f"| {'Metric':<12} | {'Persona':>8} | {'Human':>8} | {'z-score':>8} | {'Direction':>9} | Priority |"
    ]
    lines.append(f"|{'-' * 14}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 11}|----------|")
    for metric, p, hm, z, direction, priority in rows:
        lines.append(
            f"| {metric:<12} | {p:8.2f} | {hm:8.2f} | {z:8.2f} | {direction:>9} | {priority:<8} |"
        )
    return "\n".join(lines)


def format_metric_explanations() -> str:
    lines = []
    for metric in METRIC_ORDER:
        expl = METRIC_EXPLANATIONS.get(metric, "")
        lines.append(f"- **{metric}**: {expl}")
    return "\n".join(lines)


def sample_human_conversations(n: int = 3, seed: int = 42) -> list[str]:
    """Sample n rendered brinks conversations."""
    convs = json.loads(BRINKS_PATH.read_text())
    rng = np.random.default_rng(seed)
    good = [c for c in convs if len(_extract_brinks_turns(c)) >= 4]
    chosen = rng.choice(len(good), min(n, len(good)), replace=False)
    return [_render_brinks_conversation(good[i]) for i in chosen]


def get_full_traces(results_json: dict, n: int = 5) -> list[dict]:
    """Get full agent+user conversation traces from eval results."""
    traces = []
    for sim in results_json.get("simulations", []):
        msgs = sim.get("messages", [])
        if not msgs:
            continue
        reward = sim.get("reward_info", {}).get("reward", "?")
        lines = []
        for m in msgs:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user" and isinstance(content, str) and content.strip():
                lines.append(f"CUSTOMER: {content.strip()[:200]}")
            elif role == "assistant":
                if isinstance(content, str) and content.strip():
                    text = content.strip()
                    if text.startswith("{"):
                        try:
                            parsed = json.loads(text)
                            text = (
                                parsed.get("message")
                                or parsed.get("commentary_to_user")
                                or text
                            )
                        except (json.JSONDecodeError, AttributeError):
                            pass
                    if len(text) > 150:
                        text = text[:150] + "..."
                    lines.append(f"AGENT: {text}")
        if len(lines) >= 6:
            traces.append(
                {
                    "task": sim.get("task_id", "")[:60],
                    "reward": reward,
                    "transcript": "\n".join(lines[:20]),
                }
            )
        if len(traces) >= n:
            break
    return traces


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--build-ref", action="store_true")
    ap.add_argument("--brinks-path", default=str(BRINKS_PATH))
    ap.add_argument("--score", help="Path to results.json to score JSD")
    args = ap.parse_args()

    if args.build_ref:
        build_human_reference(args.brinks_path)
    elif args.score:
        human_vecs = load_human_vecs()
        human_ref = load_human_ref()
        results = json.loads(Path(args.score).read_text())
        sim_vecs = extract_sim_vecs(results)
        profile = compute_profile(sim_vecs)
        jsd = compute_jsd_vs_human(sim_vecs, human_vecs)
        zscore = compute_zscore_table(profile, human_ref)
        rewards = [
            s.get("reward_info", {}).get("reward", 0)
            for s in results["simulations"]
            if s.get("messages")
        ]
        diff = 1 - np.mean(rewards)
        print(
            f"N sims: {len(sim_vecs)}, Difficulty: {diff:.3f}, JSD vs human: {jsd:.4f}"
        )
        print(format_zscore_table(zscore))
