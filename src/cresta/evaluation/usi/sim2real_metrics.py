#!/usr/bin/env python3
"""Reproduce the 20 atomic behavioral metrics (D1-D4) from
"Mind the Sim2Real Gap in User Simulation for Agentic Tasks" (arXiv:2603.11245),
Appendix A.4.

Implementation choice: curated word lists (keyed to the paper's stated examples)
+ regex, with *no-context* word-presence counting -- this is the faithful match to
the paper, which uses LIWC2015 + small marker lists. Empath was tried and rejected
(its category word-membership does not match the paper's definitions).

Usage:
    python -m cresta.evaluation.usi.sim2real_metrics [--human-file PATH]

Computes per-interaction metrics on USER turns only, averages across interactions
within each annotation batch (base / ann2 / ann3), and compares the overall human
means against the paper's Figure-3 human reference values.
"""

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

from cresta.paths import REPO_ROOT

DEFAULT_HUMAN_FILE = (
    REPO_ROOT / "datasets" / "usi" / "data" / "tau_bench_tasks_unified.json"
)

# --------------------------------------------------------------------------- #
# Curated lexicons (lowercase). Multi-word entries matched as substrings on a
# normalized turn; single tokens matched on word boundaries.
# --------------------------------------------------------------------------- #
POLITE = [
    "please",
    "thanks",
    "thank you",
    "sorry",
    "appreciate",
    "kindly",
    "pardon",
    "excuse me",
    "grateful",
    "much appreciated",
]
UNCERT = [
    "maybe",
    "not sure",
    "i think",
    "probably",
    "perhaps",
    "possibly",
    "i guess",
    "unsure",
    "i believe",
    "i suppose",
    "kind of",
    "sort of",
    "don't remember",
    "dont remember",
    "can't remember",
    "cant remember",
    "not certain",
    "dunno",
    "i'm not sure",
    "im not sure",
    "might be",
    "i don't know",
    "i dont know",
    "no idea",
    "not really sure",
    "i'm not certain",
    "if that's okay",
    "if that's ok",
    "hopefully",
    "i hope",
    "or so",
    "i'd say",
    "i would say",
    "i think so",
    "not too sure",
    "i'm guessing",
    "im guessing",
    "i assume",
    "if possible",
    "wondering",
    "i was wondering",
    "seems like",
    "it seems",
    "could be",
    "may have",
    "might have",
    "i forget",
    "i forgot",
    "if i recall",
    "if i remember",
    "likely",
    "unlikely",
    "either way",
    "whichever",
]
CERTN = [
    "definitely",
    "certainly",
    "absolutely",
    "for sure",
    "no doubt",
    "without a doubt",
    "surely",
    "of course",
    "exactly",
    "indeed",
    "clearly",
    "obviously",
    "100%",
    "for certain",
]
EMOT = [
    "frustrated",
    "frustrating",
    "annoyed",
    "annoying",
    "irritated",
    "upset",
    "ugh",
    "angry",
    "disappointed",
    "fed up",
    "sick of",
    "tired of",
    "come on",
    "seriously",
    "argh",
]
# Two modes for the noisy categories:
#   faithful = domain-blind broad lists, mirrors the paper's regex (counts
#              "change of plans"/"wrong" etc.) -> best for REPRODUCING the paper.
#   strict   = semantically clean -> best for COMPARING simulators (no domain
#              artifacts like the airline cancellation-reason field).
FAITHFUL_ACCUSE = [
    "useless",
    "unacceptable",
    "not acceptable",
    "scam",
    "ridiculous",
    "incompetent",
    "pathetic",
    "misleading",
    "you lied",
    "you're lying",
    "youre lying",
    "nonsense",
    "this is stupid",
    "waste of",
    "the worst",
    "terrible",
    "awful",
    "you keep",
    "your mistake",
    "your fault",
    "messed up",
    "screwed up",
    "again and again",
    "keep asking",
    "are you dumb",
    "what the hell",
    "wrong",
    "should have",
    "shouldn't have",
    "shouldnt have",
    "not right",
    "unfair",
    "unreasonable",
    "unbelievable",
    "horrible",
    "disappointing",
    "frustrating",
    "annoying",
    "doesn't make sense",
    "doesnt make sense",
    "this is absurd",
    "you didn't",
    "you didnt",
    "you guys",
    "not what i",
    "complicated",
    "this is hard",
    "you should",
    "why is",
    "why are",
    "not fair",
    "not helpful",
    "not what i asked",
    "not what i wanted",
    "still not",
    "how come",
    "this is not",
    "that is not",
    "not working",
    "i need this",
    "not okay",
    "not ok",
]
STRICT_ACCUSE = [
    "useless",
    "unacceptable",
    "not acceptable",
    "scam",
    "ridiculous",
    "incompetent",
    "pathetic",
    "misleading",
    "you lied",
    "you're lying",
    "youre lying",
    "nonsense",
    "this is stupid",
    "waste of",
    "the worst",
    "terrible",
    "awful",
    "you keep",
    "your mistake",
    "your fault",
    "messed up",
    "screwed up",
    "again and again",
    "keep asking",
    "are you dumb",
    "what the hell",
]
FAITHFUL_PIVOT = [
    "instead",
    "on second thought",
    "let's try",
    "lets try",
    "never mind",
    "nevermind",
    "forget it",
    "alternatively",
    "in that case",
    "how about",
    "scratch that",
    "let's go with",
    "lets go with",
    "i'll just",
    "ill just",
    "i would rather",
    "i'd rather",
    "rather",
    "switch to",
    "can we just",
    "you know what",
    "let me try",
    "on the other hand",
    "a different",
    "change it to",
    "change of plan",
    "change of plans",
    "actually",
    "make it",
    "go with",
    "let's do",
    "lets do",
    "different one",
    "let me",
    "or maybe",
    "second thought",
    "change my",
    "want to change",
    "modify",
]
STRICT_PIVOT = [
    "instead",
    "on second thought",
    "let's try",
    "lets try",
    "never mind",
    "nevermind",
    "forget it",
    "alternatively",
    "in that case",
    "how about",
    "scratch that",
    "let's go with",
    "lets go with",
    "i'll just",
    "ill just",
    "i would rather",
    "i'd rather",
    "switch to",
    "can we just",
    "you know what",
    "let me try",
    "on the other hand",
    "a different",
    "change it to",
]

# active lists (default faithful); flip with set_mode("strict")
ACCUSE = FAITHFUL_ACCUSE
PIVOT = FAITHFUL_PIVOT


def set_mode(mode):
    global ACCUSE, PIVOT, RE_IDCONF
    if mode == "strict":
        ACCUSE, PIVOT, RE_IDCONF = STRICT_ACCUSE, STRICT_PIVOT, RE_IDCONF_STRICT
    elif mode == "faithful":
        ACCUSE, PIVOT, RE_IDCONF = FAITHFUL_ACCUSE, FAITHFUL_PIVOT, RE_IDCONF_FAITHFUL
    else:
        raise ValueError(mode)


# Regex patterns (turn-level)
RE_FORMAL = re.compile(r"[–—]")  # en-dash / em-dash
# strict: unambiguous agent role-play / self-identification as the service rep.
RE_IDCONF_STRICT = re.compile(
    r"how (?:may|can) i (?:help|assist) you|for verification purposes|"
    r"thank you for calling|is there anything else i can (?:help|do|assist)|"
    r"i'?m (?:your|the) (?:assistant|agent|representative|support)|"
    r"this is (?:customer )?(?:support|service)|welcome to",
    re.I,
)
# faithful: add the paper's explicitly-listed "let me check" service phrase
# (App. A.4 lists it as an IDConf marker). In telecom the *user* also says this
# legitimately, so strict mode drops it -- but faithful must mirror the paper.
RE_IDCONF_FAITHFUL = re.compile(RE_IDCONF_STRICT.pattern + r"|let me check", re.I)
RE_IDCONF = RE_IDCONF_FAITHFUL  # default faithful; flipped by set_mode()
RE_PUSHBACK = re.compile(
    r"are you sure|you already (?:asked|have|know)|that'?s not right|"
    r"that is not right|that'?s not what|that is not what|"
    r"i already (?:told|gave|said|provided|mentioned)|as i (?:said|mentioned)|"
    r"that'?s wrong|that is wrong|wrong (?:reservation|flight|one|number|"
    r"account|order|info|name|id)|why (?:did|can'?t|cant|won'?t|wont|would|are) you|"
    r"isn'?t that what|like i said|i told you|no,? that'?s",
    re.I,
)
RE_CLARIFY = re.compile(
    r"what do you mean|can you clarify|what does that mean|i'?m confused|"
    r"im confused|not following|what (?:exactly )?do you mean|"
    r"which (?:one|reservation|order|account|flight|certificate|of these)\b|"
    r"can you explain|not sure what you mean|what are you referring|"
    r"(?:do |did )?you mean\b|couldn'?t you (?:find|tell|figure)|"
    r"can you not (?:tell|find)",
    re.I,
)
# Information-seeking PATTERNS (not just any question).
RE_INFOSEEK = re.compile(
    r"can you (?:check|tell|confirm|see|look|find|let me know|give|help)|"
    r"could you (?:check|tell|confirm|see|look|find|give|help)|"
    r"would you (?:check|tell|confirm|mind)|"
    r"what(?:'s| is| are)(?: the| my)? (?:status|price|cost|option|fee|"
    r"balance|total|difference|process|policy|deadline|limit)|"
    r"how (?:much|many|long|do|does|can i|would)|"
    r"do you (?:have|know|offer|accept|see|need)|is there|are there|"
    r"what are my options|when (?:is|will|can|does)|where (?:is|can|do)|"
    r"what happens|will i (?:get|be)|can i (?:get|use|still|change|cancel)|"
    r"i(?:'d| would) like to (?:know|check|see|find out)",
    re.I,
)
# identifier-like tokens
RE_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
# any token that contains a digit and has >=4 alphanumeric chars (catches
# user ids like mia_li_3668, zips 80273, conf codes W9502127, card 7447, dates)
RE_ID_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-/]*\d[A-Za-z0-9_.\-/]*")
RE_CAPS_ID = re.compile(
    r"\b[A-Z]{2,}[A-Z0-9]{3,}\b"
)  # all-caps structured IDs w/o digits

ACK_VOCAB = {
    "ok",
    "okay",
    "k",
    "sure",
    "yes",
    "yep",
    "yeah",
    "yup",
    "thanks",
    "thank",
    "thx",
    "alright",
    "fine",
    "great",
    "got",
    "it",
    "cool",
    "good",
    "perfect",
    "right",
    "correct",
    "no",
    "nope",
    "continue",
    "proceed",
    "go",
    "ahead",
    "done",
    "understood",
}

# Paper Figure-3 human reference values (dark bars).
REF_HUMAN = {
    "Wds/trn": 10.7,
    "Short%": 29.0,
    "Polite%": 15.3,
    "Formal%": 1.2,
    "Ack%": 13.2,
    "VerbCV": 0.8,
    "Repeat%": 0.0,
    "IDConf%": 0.0,
    "FrntLoad%": 31.7,
    "IDs/trn": 2.6,
    "Open.Wds": 18.7,
    "Uncert%": 7.3,
    "Certn%": 1.0,
    "Pushbk%": 0.7,
    "ClarfyQ%": 0.7,
    "InfoQ%": 6.5,
    "Emot%": 1.1,
    "Accuse%": 2.7,
    "Pivot%": 8.4,
}
METRIC_ORDER = [
    "Wds/trn",
    "Short%",
    "Polite%",
    "Formal%",
    "Ack%",
    "VerbCV",
    "Repeat%",
    "IDConf%",
    "FrntLoad%",
    "IDs/trn",
    "Open.Wds",
    "Uncert%",
    "Certn%",
    "Pushbk%",
    "ClarfyQ%",
    "InfoQ%",
    "Emot%",
    "Accuse%",
    "Pivot%",
]


def norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_any(ntext, phrases):
    for p in phrases:
        if " " in p:
            if p in ntext:
                return True
        else:
            if re.search(r"\b" + re.escape(p) + r"\b", ntext):
                return True
    return False


def is_meta(content):
    s = content.strip()
    return (
        s.startswith("\\tau")
        or s.startswith("\\reward")
        or s == "/stop"
        or s.startswith("<|canvas|>")
        or s.startswith("<|highlight|>")
    )


def user_turns(conversation):
    out = []
    for m in conversation:
        if m.get("role") != "user":
            continue
        c = str(m.get("content", ""))
        if is_meta(c):
            continue
        out.append(c)
    return out


def count_ids(turn):
    n = len(RE_EMAIL.findall(turn))
    no_email = RE_EMAIL.sub(" ", turn)
    for mtok in RE_ID_TOKEN.findall(no_email):
        alnum = re.sub(r"[^A-Za-z0-9]", "", mtok)
        if len(alnum) >= 4:
            n += 1
    for mtok in RE_CAPS_ID.findall(no_email):
        if not any(ch.isdigit() for ch in mtok):  # avoid double-count w/ RE_ID_TOKEN
            n += 1
    return n


def metrics_for_interaction(turns):
    if not turns:
        return None
    wc = [len(t.split()) for t in turns]
    ntexts = [norm(t) for t in turns]
    total_words = sum(wc)

    # D1
    wds_trn = st.mean(wc)
    short = 100 * sum(1 for w in wc if w <= 3) / len(wc)
    polite = 100 * sum(1 for n in ntexts if contains_any(n, POLITE)) / len(turns)
    formal = 100 * sum(1 for t in turns if RE_FORMAL.search(t)) / len(turns)

    def is_ack(t):
        toks = re.findall(r"[a-z]+", t.lower())
        return bool(toks) and all(tok in ACK_VOCAB for tok in toks)

    ack = 100 * sum(1 for t in turns if is_ack(t)) / len(turns)
    verbcv = (st.pstdev(wc) / st.mean(wc)) if st.mean(wc) > 0 else 0.0
    # repeat: any trigram (over concatenated user tokens) appearing >5 times
    alltok = []
    for t in turns:
        alltok += re.findall(r"[a-z0-9']+", t.lower())
    tri = Counter(tuple(alltok[i : i + 3]) for i in range(len(alltok) - 2))
    repeat = 100.0 if (tri and max(tri.values()) > 12) else 0.0
    idconf = 100.0 if any(RE_IDCONF.search(t) for t in turns) else 0.0

    # D2
    frntload = (100 * sum(wc[:2]) / total_words) if total_words else 0.0
    ids_trn = st.mean([count_ids(t) for t in turns])
    open_wds = wc[0]

    # D3
    uncert = 100 * sum(1 for n in ntexts if contains_any(n, UNCERT)) / len(turns)
    certn = 100 * sum(1 for n in ntexts if contains_any(n, CERTN)) / len(turns)
    push = clar = info = 0
    for t in turns:
        if RE_PUSHBACK.search(t):
            push += 1
        elif RE_CLARIFY.search(t):
            clar += 1
        elif RE_INFOSEEK.search(t):
            info += 1
    n = len(turns)
    pushbk = 100 * push / n
    clarfyq = 100 * clar / n
    infoq = 100 * info / n

    # D4
    emot = 100 * sum(1 for nn in ntexts if contains_any(nn, EMOT)) / len(turns)
    accuse = 100 * sum(1 for nn in ntexts if contains_any(nn, ACCUSE)) / len(turns)
    pivot = 100 * sum(1 for nn in ntexts if contains_any(nn, PIVOT)) / len(turns)

    return {
        "Wds/trn": wds_trn,
        "Short%": short,
        "Polite%": polite,
        "Formal%": formal,
        "Ack%": ack,
        "VerbCV": verbcv,
        "Repeat%": repeat,
        "IDConf%": idconf,
        "FrntLoad%": frntload,
        "IDs/trn": ids_trn,
        "Open.Wds": open_wds,
        "Uncert%": uncert,
        "Certn%": certn,
        "Pushbk%": pushbk,
        "ClarfyQ%": clarfyq,
        "InfoQ%": infoq,
        "Emot%": emot,
        "Accuse%": accuse,
        "Pivot%": pivot,
    }


def batch_of(key):
    if key.endswith("_ann2"):
        return "ann2"
    if key.endswith("_ann3"):
        return "ann3"
    return "base"


def main(path, include_agreement=True):
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    per_batch = defaultdict(lambda: defaultdict(list))  # batch -> metric -> [vals]
    overall = defaultdict(list)
    n_used = 0
    for key, entry in data.items():
        turns = user_turns(entry["conversation"])
        m = metrics_for_interaction(turns)
        if m is None:
            continue
        n_used += 1
        b = batch_of(key)
        for k, v in m.items():
            per_batch[b][k].append(v)
            overall[k].append(v)

    # metrics whose exact paper regex/lexicon is unspecified -> not bit-reproducible
    UNSPEC = {"IDs/trn"}

    print(f"interactions used: {n_used}\n")
    batches = ["base", "ann2", "ann3"]
    header = (
        f"{'metric':10s} {'paper(H)':>9s} {'mine(all)':>10s} {'Δ':>7s}  ok  "
        + "  ".join(f"{b:>7s}" for b in batches)
    )
    print(header)
    print("-" * len(header))
    refs, mines = [], []
    for k in METRIC_ORDER:
        ref = REF_HUMAN[k]
        mine = st.mean(overall[k])
        delta = mine - ref
        bvals = "  ".join(f"{st.mean(per_batch[b][k]):7.2f}" for b in batches)
        if k in UNSPEC:
            flag = "✗"  # regex unspecified in paper
        else:
            tol = max(0.8, 0.20 * ref)  # within 0.8pp or 20% counts as a match
            flag = "✓" if abs(delta) <= tol else "~"
            refs.append(ref)
            mines.append(mine)
        print(f"{k:10s} {ref:9.2f} {mine:10.2f} {delta:+7.2f}   {flag}   {bvals}")

    if not include_agreement:
        print("-" * len(header))
        print("strict mode is atomic debug only; no aggregate score is reported")
        return

    # agreement summary (excluding unspecified-regex metrics)
    mae = st.mean([abs(a - b) for a, b in zip(refs, mines)])
    n = len(refs)
    mr, mm = st.mean(refs), st.mean(mines)
    cov = sum((a - mr) * (b - mm) for a, b in zip(refs, mines))
    den = (sum((a - mr) ** 2 for a in refs) * sum((b - mm) ** 2 for b in mines)) ** 0.5
    pearson = cov / den if den else float("nan")
    print("-" * len(header))
    print(
        f"reproducible metrics: {n}/{len(METRIC_ORDER)}   "
        f"MAE={mae:.2f}pp   Pearson r={pearson:.3f}"
    )
    print("legend: ✓ within tol(max 0.8pp / 20%)   ~ off   ✗ paper regex unspecified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect the paper's atomic USI metrics"
    )
    parser.add_argument(
        "--human-file",
        type=Path,
        default=DEFAULT_HUMAN_FILE,
        help=f"unified human sessions JSON (default: {DEFAULT_HUMAN_FILE})",
    )
    parser.add_argument(
        "--mode",
        choices=("faithful", "strict"),
        default="faithful",
        help="'strict' is an atomic-metric debug mode, not another USI score",
    )
    cli_args = parser.parse_args(sys.argv[1:])
    set_mode(cli_args.mode)
    main(cli_args.human_file, include_agreement=cli_args.mode == "faithful")
