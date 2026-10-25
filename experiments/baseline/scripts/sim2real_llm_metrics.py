#!/usr/bin/env python3
"""LLM-rubric half of the *hybrid* Sim2Real behavioral metrics.

The rule-based metrics in ``sim2real_metrics.py`` reproduce 16/19 of the paper's
Figure-3 human values to <=1.6pp. The remaining 10 are semantic / context-
dependent and rely on a lexicon/regex the paper never published, so they are
unreliable and domain-fragile -- e.g. the faithful IDConf regex fires 87.6% on
telecom because customers legitimately say "let me check [my phone]".

This module computes those 10 metrics with an LLM-with-rubric judge (default
gpt-5.2): ONE call per interaction that multi-labels every user turn at once,
maj@N majority vote, disk-cached. ``hybrid_metrics_for_interaction()`` merges the
9 trustworthy rule metrics with these 10 LLM metrics into the SAME dict shape as
``sim2real_metrics.metrics_for_interaction`` so the runners are unchanged.

Routing (per-metric, static -- see the plan / audit):
    rule (9): Wds/trn Short% Polite% Formal% Ack% VerbCV Repeat% FrntLoad% Open.Wds
    LLM (10): IDConf% IDs/trn Uncert% Certn% Pushbk% ClarfyQ% InfoQ% Emot% Accuse% Pivot%

Reuses the gpt-5.2 sync-judge transport pattern from
``scripts/judge_user_intent_preservation.py`` (OpenAI client + reasoning_effort
+ ThreadPoolExecutor + maj@N voting).
"""

import hashlib
import json
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
import sim2real_metrics as SM
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --------------------------------------------------------------------------- #
# Config (env-overridable)
# --------------------------------------------------------------------------- #
MODEL = os.environ.get("SIM2REAL_LLM_MODEL", "gpt-5.2")
MAJ = int(os.environ.get("SIM2REAL_LLM_MAJ", "3"))
EFFORT = os.environ.get("SIM2REAL_LLM_EFFORT", "medium")
CONC = int(os.environ.get("SIM2REAL_LLM_CONC", "16"))
CACHE_DIR = Path(os.environ.get("SIM2REAL_LLM_CACHE", "data/sim2real_llm_cache"))

# The 10 metrics this module owns; everything else stays rule-based.
LLM_METRICS = [
    "IDConf%",
    "IDs/trn",
    "Uncert%",
    "Certn%",
    "Pushbk%",
    "ClarfyQ%",
    "InfoQ%",
    "Emot%",
    "Accuse%",
    "Pivot%",
]
RULE_METRICS = [k for k in SM.METRIC_ORDER if k not in LLM_METRICS]

# Per-turn boolean rubric fields (identity_confusion handled separately = per-interaction).
_BOOL_KEYS = ["emotion", "accusatory", "pivot", "uncertainty", "certainty"]
_QT = ("none", "pushback", "clarify", "infoseek")
_EMPTY = {
    "emotion": False,
    "accusatory": False,
    "pivot": False,
    "uncertainty": False,
    "certainty": False,
    "identity_confusion": False,
    "num_identifiers": 0,
    "question_type": "none",
}

# --------------------------------------------------------------------------- #
# Rubric (paper App. A.4 definitions + the domain-disambiguation the audit found)
# --------------------------------------------------------------------------- #
SYS = (
    "You are an expert conversation analyst. You will read the USER (customer) turns of a "
    "single customer-support conversation and label EACH turn on several behavioral "
    "dimensions. Judge ONLY what the user said; ignore the agent. Be conservative: when a "
    'signal is absent or ambiguous, choose false / "none" / 0.\n\n'
    "For each numbered user turn decide:\n\n"
    "- emotion (bool): the user expresses negative emotion/affect (frustration, anger, "
    "annoyance, anxiety, disappointment, impatience). Stating a problem factually ('my "
    "internet is down', 'this is complicated') is NOT emotion.\n\n"
    "- accusatory (bool): the user is accusatory or strongly negative TOWARD THE AGENT / "
    "company -- blaming, insulting, alleging incompetence or dishonesty ('you're useless', "
    "'this is a scam', 'you keep failing me', 'that's your fault'). Politely correcting a "
    "factual error, or calling the situation hard/wrong/frustrating WITHOUT blaming the "
    "agent, is NOT accusatory.\n\n"
    "- pivot (bool): the user genuinely CHANGES strategy or goal -- abandons the current "
    "request, switches to an alternative, or proposes a different plan ('forget the refund, "
    "just cancel it', 'let's try a different flight instead'). 'actually' used as filler, "
    "routine continuations, or merely supplying more detail are NOT pivots.\n\n"
    "- uncertainty (bool): the user hedges about their own knowledge/preference ('I'm not "
    "sure', 'maybe', 'I think', 'I don't remember', 'probably').\n\n"
    "- certainty (bool): the user signals strong certainty/emphasis ('definitely', 'for "
    "sure', 'absolutely', 'I'm certain', '100%').\n\n"
    "- identity_confusion (bool): the user talks AS IF THEY WERE THE SUPPORT AGENT -- using "
    "service-rep language directed at a customer ('How may I help you?', 'Thank you for "
    "calling', 'Let me pull up your account', 'Is there anything else I can help you "
    "with?'). CRITICAL: a customer operating THEIR OWN device/account ('let me check my "
    "phone', 'let me look at my settings', 'let me check my email') is NOT identity "
    "confusion -- that is normal customer behavior.\n\n"
    "- num_identifiers (int): count the distinct SPECIFIC / high-entropy values the user "
    "STATES in this turn -- proper names, account/order/reservation/customer IDs, phone "
    "numbers, emails, street addresses, ZIP, city / airport / place names, dates, times, "
    "flight numbers, product / plan names, confirmation / booking codes, card last-4, "
    "quantities and dollar amounts. Count each distinct value once; 0 if none. (E.g. 'book "
    "a one-way flight May 20 from New York to Seattle' = 3: May 20, New York, Seattle. "
    "'3 bags, economy' = 2: 3, economy.)\n\n"
    "- question_type (one of 'none','pushback','clarify','infoseek'): the user's PRIMARY "
    "question/challenge this turn, MUTUALLY EXCLUSIVE, in priority order:\n"
    "    pushback : the user explicitly CHALLENGES, CONTRADICTS, CORRECTS, or RESISTS the "
    "agent -- disputing what the agent said/did or objecting to a request ('are you sure?', "
    "'I already gave you that', 'that's not right', 'why do you need that again?'). A normal "
    "follow-up question, a new request, or simply providing more info is NOT pushback "
    "(pushback is rare -- only flag a genuine challenge/correction).\n"
    "    clarify  : asks the agent to clarify/explain something the user did not understand "
    "('what do you mean?', 'can you explain?', 'which one?').\n"
    "    infoseek : a genuine QUESTION seeking information / an answer the user wants to KNOW "
    "('how much does it cost?', 'what are my options?', 'is there a fee?', 'can you tell me "
    "my balance?'). A pure ACTION REQUEST or imperative ('book it', 'search all three and "
    "pick the cheapest', 'cancel my order', 'find me the lowest price') is NOT infoseek.\n"
    "    none     : no such question/challenge (statements, confirmations, action requests, "
    "providing info).\n"
    "  If several apply choose the highest priority (pushback > clarify > infoseek).\n\n"
    'Return STRICT JSON only: {"labels": [ {"turn": <int>, "emotion": <bool>, '
    '"accusatory": <bool>, "pivot": <bool>, "uncertainty": <bool>, "certainty": '
    '<bool>, "identity_confusion": <bool>, "num_identifiers": <int>, "question_type": '
    '"<one of the four>"}, ... ] } with EXACTLY one object per user turn, in order. '
    "No prose."
)


def _user_prompt(turns):
    block = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(turns))
    return (
        f"There are {len(turns)} USER turns. Label every one.\n\n"
        f"===== USER TURNS =====\n{block}\n\n"
        "Return the strict JSON described in the system message "
        f"(exactly {len(turns)} label objects)."
    )


def _extract_json(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"```\s*$", "", s).strip()
    return s


def _call(client, model, effort, turns):
    """One LLM labeling call -> list of per-turn dicts (length may differ; caller aligns)."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": _user_prompt(turns)},
        ],
        "response_format": {"type": "json_object"},
    }
    if model.startswith("gpt-5") or model.startswith("o"):
        body["reasoning_effort"] = effort
    else:
        body["temperature"] = 1.0
    resp = client.chat.completions.create(**body)
    obj = json.loads(_extract_json(resp.choices[0].message.content))
    labs = obj.get("labels") or []
    return labs if isinstance(labs, list) else []


# --------------------------------------------------------------------------- #
# maj@N voting (align votes by turn index)
# --------------------------------------------------------------------------- #
def _vote(votes, n):
    out = []
    for i in range(n):
        cell = [v[i] for v in votes if i < len(v) and isinstance(v[i], dict)]
        if not cell:
            out.append(dict(_EMPTY))
            continue
        d = {}
        for k in _BOOL_KEYS + ["identity_confusion"]:
            t = sum(1 for c in cell if c.get(k))
            d[k] = t * 2 > len(cell)  # strict majority; ties -> False (conservative)
        nums = sorted(int(c.get("num_identifiers") or 0) for c in cell)
        d["num_identifiers"] = nums[len(nums) // 2]  # median
        qts = [
            c.get("question_type") if c.get("question_type") in _QT else "none"
            for c in cell
        ]
        d["question_type"] = Counter(qts).most_common(1)[0][0]
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# Cache + batched labeling
# --------------------------------------------------------------------------- #
# Short fingerprint of the rubric so cache auto-invalidates when SYS changes.
_RUBRIC_SIG = hashlib.sha1(SYS.encode("utf-8")).hexdigest()[:8]


def _key(model, maj, effort, turns):
    blob = f"{model}|{maj}|{effort}|{_RUBRIC_SIG}\n" + "\n###\n".join(turns)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def label_interactions(
    turn_lists,
    model=MODEL,
    maj=MAJ,
    effort=EFFORT,
    concurrency=CONC,
    cache=True,
    verbose=True,
):
    """Label many interactions. Cache-misses fan out across ONE thread pool
    (interactions x maj jobs); returns voted per-turn labels aligned to turn_lists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = [None] * len(turn_lists)
    keys = [None] * len(turn_lists)
    jobs = []  # (idx, vote)
    hits = 0
    for i, turns in enumerate(turn_lists):
        if not turns:
            results[i] = []
            continue
        k = _key(model, maj, effort, turns)
        keys[i] = k
        cp = CACHE_DIR / f"{k}.json"
        if cache and cp.exists():
            try:
                results[i] = json.load(open(cp))
                hits += 1
                continue
            except Exception:
                pass
        for v in range(maj):
            jobs.append((i, v))

    if verbose:
        print(
            f"[llm-metrics] {len(turn_lists)} interactions: {hits} cached, "
            f"{len(jobs)} calls ({len(jobs) // max(maj, 1)} miss x maj={maj}) "
            f"| model={model} effort={effort} conc={concurrency}",
            flush=True,
        )

    if jobs:
        client = OpenAI()
        votes = defaultdict(list)
        errors = 0

        def _one(job):
            i, _v = job
            try:
                return i, _call(client, model, effort, turn_lists[i])
            except Exception:
                return i, None

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for i, labs in ex.map(_one, jobs):
                if labs is None:
                    errors += 1
                else:
                    votes[i].append(labs)
        if errors and verbose:
            print(
                f"[llm-metrics] {errors}/{len(jobs)} calls errored "
                "(treated as missing votes)",
                flush=True,
            )
        for i in range(len(turn_lists)):
            if results[i] is not None:
                continue
            voted = _vote(votes.get(i, []), len(turn_lists[i]))
            results[i] = voted
            if cache and keys[i]:
                try:
                    json.dump(voted, open(CACHE_DIR / f"{keys[i]}.json", "w"))
                except Exception:
                    pass
    return results


def llm_label_interaction(turns, model=MODEL, maj=MAJ, effort=EFFORT, cache=True):
    return label_interactions(
        [turns], model=model, maj=maj, effort=effort, cache=cache, verbose=False
    )[0]


# --------------------------------------------------------------------------- #
# Aggregation: labels -> the 10 LLM metrics (same semantics as the rule module)
# --------------------------------------------------------------------------- #
def metrics_from_labels(turns, labels):
    n = len(turns)
    if n == 0:
        return {k: float("nan") for k in LLM_METRICS}

    def rate(key):
        return 100.0 * sum(1 for x in labels if x.get(key)) / n

    qt = [x.get("question_type", "none") for x in labels]
    return {
        # IDConf% is a per-interaction binary in the rule module -> mirror it.
        "IDConf%": 100.0 if any(x.get("identity_confusion") for x in labels) else 0.0,
        "IDs/trn": st.mean([int(x.get("num_identifiers") or 0) for x in labels]),
        "Uncert%": rate("uncertainty"),
        "Certn%": rate("certainty"),
        "Pushbk%": 100.0 * qt.count("pushback") / n,
        "ClarfyQ%": 100.0 * qt.count("clarify") / n,
        "InfoQ%": 100.0 * qt.count("infoseek") / n,
        "Emot%": rate("emotion"),
        "Accuse%": rate("accusatory"),
        "Pivot%": rate("pivot"),
    }


def hybrid_metrics_for_interaction(
    turns, model=MODEL, maj=MAJ, effort=EFFORT, labels=None, cache=True
):
    """9 rule metrics + 10 LLM metrics, same dict shape as SM.metrics_for_interaction.

    The 9 rule metrics we keep are mode-independent (mode only affected the LLM-owned
    Accuse/Pivot/IDConf), so hybrid output does not depend on SM.set_mode()."""
    base = SM.metrics_for_interaction(turns)
    if base is None:
        return None
    if labels is None:
        labels = llm_label_interaction(
            turns, model=model, maj=maj, effort=effort, cache=cache
        )
    base.update(metrics_from_labels(turns, labels))
    return base


# --------------------------------------------------------------------------- #
# Smoke test: label the first few human interactions and dump the JSON labels.
# --------------------------------------------------------------------------- #
def _smoke(path, limit):
    data = json.load(open(path))
    items = list(data.values()) if isinstance(data, dict) else data
    turn_lists, srcs = [], []
    for entry in items:
        conv = entry.get("conversation") or entry.get("messages") or []
        turns = SM.user_turns(conv)
        if turns:
            turn_lists.append(turns)
            srcs.append(entry.get("instance_id", "?"))
        if len(turn_lists) >= limit:
            break
    labs = label_interactions(turn_lists, maj=1)
    for src, turns, L in zip(srcs, turn_lists, labs):
        print(f"\n===== {src} ({len(turns)} turns) =====")
        for i, (t, lab) in enumerate(zip(turns, L)):
            print(f"[{i + 1}] {t[:90]!r}")
            print(f"     {json.dumps(lab)}")
        print(
            "  metrics:",
            json.dumps(
                {k: round(v, 2) for k, v in metrics_from_labels(turns, L).items()}
            ),
        )


if __name__ == "__main__":
    p = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            os.environ.get("PERSONA_DATA_DIR", "/Users/yutao.xie/Downloads"),
            "tau_bench_tasks_unified.json",
        )
    )
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    _smoke(p, lim)
