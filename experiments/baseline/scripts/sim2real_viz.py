#!/usr/bin/env python3
"""Visualize tau2 / human conversation trajectories as a self-contained HTML page.

Reads a tau2 results.json (sims) OR the human unified.json, renders each
trajectory with:
  - a header (task_id, trial, reward, termination, user-sim, persona, reason_for_call)
  - the turn-by-turn dialogue, colour-coded user / agent / tool-call / tool-result
  - per USER turn: which RULE behavioral metrics fired (Accuse/Pivot/Emot/Pushbk/...)
    + word count + #identifiers, so you can SEE why a turn is counted
  - optionally (--hybrid): the LLM-rubric labels for each user turn alongside the
    rule tags (reads the disk cache; only calls the LLM for uncached interactions)
  - a per-conversation metric strip + a filterable/sortable index at the top

Examples:
  python sim2real_viz.py data/simulations/gepa_certify/progcert_candidate/telecom/results.json
  python sim2real_viz.py <results.json> --sort reward --limit 20 --max-reward 0.5
  python sim2real_viz.py <results.json> --task task_042 --hybrid
  python sim2real_viz.py ~/Downloads/tau_bench_tasks_unified.json --sort adv --limit 30
"""

import argparse
import html
import json
import os
import sys
import webbrowser

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_r, "pyproject.toml")) and _r != "/":
    _r = os.path.dirname(_r)
sys.path.insert(0, os.path.join(_r, "src", "cresta", "evaluation", "usi"))
import sim2real_metrics as M
import sim2real_run_tau2 as R

# rule-metric -> chip colour
TAGCOL = {
    "Accuse": "#c0392b",
    "Pivot": "#8e44ad",
    "Emot": "#e67e22",
    "Pushbk": "#d35400",
    "Clarify": "#16a085",
    "InfoQ": "#2980b9",
    "Uncert": "#7f8c8d",
    "Certn": "#27ae60",
    "Polite": "#95a5a6",
    "IDConf": "#000000",
}


def turn_flags(t, mode):
    """Which rule metrics fire on a single user turn (mode-aware)."""
    n = M.norm(t)
    f = []
    if M.contains_any(n, M.UNCERT):
        f.append("Uncert")
    if M.contains_any(n, M.CERTN):
        f.append("Certn")
    if M.contains_any(n, M.EMOT):
        f.append("Emot")
    if M.contains_any(n, M.ACCUSE):
        f.append("Accuse")
    if M.contains_any(n, M.PIVOT):
        f.append("Pivot")
    if M.contains_any(n, M.POLITE):
        f.append("Polite")
    if M.RE_IDCONF.search(t):
        f.append("IDConf")
    if M.RE_PUSHBACK.search(t):
        f.append("Pushbk")
    elif M.RE_CLARIFY.search(t):
        f.append("Clarify")
    elif M.RE_INFOSEEK.search(t):
        f.append("InfoQ")
    return f, M.count_ids(t)


def llm_turn_tags(lab):
    """Compact tags from an LLM label dict."""
    out = []
    m = {
        "accusatory": "Accuse",
        "pivot": "Pivot",
        "emotion": "Emot",
        "uncertainty": "Uncert",
        "certainty": "Certn",
        "identity_confusion": "IDConf",
    }
    for k, name in m.items():
        if lab.get(k):
            out.append(name)
    qt = lab.get("question_type", "none")
    if qt == "pushback":
        out.append("Pushbk")
    elif qt == "clarify":
        out.append("Clarify")
    elif qt == "infoseek":
        out.append("InfoQ")
    return out, int(lab.get("num_identifiers") or 0)


# --------------------------------------------------------------------------- #
# Loading: normalize tau2 results.json AND human unified.json to a common shape
# --------------------------------------------------------------------------- #
def load(path):
    """Return (kind, sim_meta_str, list-of-trajectories).

    Each trajectory: dict(label, task_id, trial, reward, termination, persona,
    reason, known, messages=[(role, content, tool_calls)])."""
    d = json.load(open(path))

    if isinstance(d, dict) and "simulations" in d:  # tau2 results.json
        ui = d.get("info", {}).get("user_info", {})
        ai = d.get("info", {}).get("agent_info", {})
        meta = (
            f"user={ui.get('implementation')}:{ui.get('llm')} "
            f"persona={ui.get('llm_args', {}).get('personas')} | "
            f"agent={ai.get('llm')}"
        )
        tasks = {t["id"]: t for t in d.get("tasks", [])}
        trajs = []
        for s in d["simulations"]:
            t = tasks.get(s.get("task_id"), {})
            us = t.get("user_scenario", {})
            ins = (
                us.get("instructions", {})
                if isinstance(us.get("instructions"), dict)
                else {}
            )
            msgs = []
            for m in s.get("messages", []):
                msgs.append((m.get("role"), m.get("content"), m.get("tool_calls")))
            trajs.append(
                {
                    "label": f"{s.get('task_id')}  (trial {s.get('trial', 0)})",
                    "task_id": s.get("task_id"),
                    "trial": s.get("trial", 0),
                    "reward": (s.get("reward_info") or {}).get("reward"),
                    "termination": s.get("termination_reason"),
                    "persona": us.get("persona"),
                    "reason": ins.get("reason_for_call"),
                    "known": ins.get("known_info"),
                    "messages": msgs,
                }
            )
        return "tau2", meta, trajs

    # human unified.json: {id: {conversation, instance_id, survey, reward}} or list
    items = list(d.values()) if isinstance(d, dict) else d
    trajs = []
    for e in items:
        conv = e.get("conversation", [])
        msgs = [(m.get("role"), m.get("content"), m.get("tool_calls")) for m in conv]
        sv = e.get("survey", {})
        trajs.append(
            {
                "label": e.get("instance_id", "?"),
                "task_id": e.get("instance_id", "?"),
                "trial": 0,
                "reward": e.get("reward"),
                "termination": None,
                "persona": None,
                "reason": (sv.get("overall_score", {}) or {}).get("answer"),
                "known": None,
                "messages": msgs,
            }
        )
    return "human", f"human role-play sessions ({len(trajs)})", trajs


def adv_score(traj, mode):
    turns = [c for r, c, _ in traj["messages"] if r == "user" and c]
    turns = R.tau2_user_turns([{"role": "user", "content": c} for c in turns])
    mm = M.metrics_for_interaction(turns)
    if not mm:
        return 0.0
    return mm["Emot%"] + mm["Accuse%"] + mm["Pivot%"] + mm["Pushbk%"]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def esc(x):
    return html.escape(str(x)) if x is not None else ""


def chip(text, color, title=""):
    return (
        f'<span class="chip" style="background:{color}" title="{esc(title)}">'
        f"{esc(text)}</span>"
    )


def render_tool_calls(tcs):
    out = []
    for tc in tcs or []:
        name = tc.get("name") or (tc.get("function") or {}).get("name") or "tool"
        args = tc.get("arguments")
        if args is None:
            args = (tc.get("function") or {}).get("arguments")
        if isinstance(args, (dict, list)):
            args = json.dumps(args, ensure_ascii=False)
        out.append(
            f'<div class="toolcall">🔧 {esc(name)}(<span class="args">'
            f"{esc(args)}</span>)</div>"
        )
    return "".join(out)


def render_traj(sid, num, traj, mode, llm_labels):
    rw = traj["reward"]
    rwc = (
        "#888"
        if rw is None
        else ("#1e8449" if rw >= 0.999 else ("#b9770e" if rw and rw > 0 else "#922b21"))
    )
    head = [
        f'<div class="head"><b>#{num}</b> {esc(traj["label"])} '
        f'<span class="reward" style="background:{rwc}">reward='
        f"{('%.3f' % rw) if rw is not None else '-'}</span>"
    ]
    if traj["termination"]:
        head.append(f'<span class="term">{esc(traj["termination"])}</span>')
    head.append("</div>")
    meta = []
    if traj["persona"]:
        meta.append(
            f'<details><summary>persona</summary><div class="meta">'
            f"{esc(traj['persona'])}</div></details>"
        )
    if traj["reason"]:
        meta.append(f'<div class="meta"><b>reason:</b> {esc(traj["reason"])}</div>')
    if traj["known"]:
        meta.append(
            f'<details><summary>known_info</summary><div class="meta">'
            f"{esc(traj['known'])}</div></details>"
        )

    rows = []
    uturn = 0
    for role, content, tcs in traj["messages"]:
        if role == "tool":
            err = ""
            rows.append(
                f'<div class="row tool"><div class="bub toolres{err}">'
                f"↩ {esc(content)}</div></div>"
            )
            continue
        if role == "assistant":
            body = render_tool_calls(tcs)
            if content:
                body = f"<div>{esc(content)}</div>" + body
            if not body:
                continue
            rows.append(
                f'<div class="row agent"><div class="bub agentbub">{body}</div></div>'
            )
            continue
        if role == "user":
            if content is None:
                continue
            clean = R.tau2_user_turns([{"role": "user", "content": content}])
            if not clean:  # control-only turn
                rows.append(
                    f'<div class="row user"><div class="bub userbub ctl">'
                    f"{esc(content)}</div></div>"
                )
                continue
            txt = clean[0]
            flags, ids = turn_flags(txt, mode)
            wc = len(txt.split())
            tags = "".join(chip(f, TAGCOL.get(f, "#555")) for f in flags)
            if ids:
                tags += chip(f"IDs:{ids}", "#34495e")
            tags += f'<span class="wc">{wc}w</span>'
            llm_html = ""
            if llm_labels is not None and uturn < len(llm_labels):
                ltags, lids = llm_turn_tags(llm_labels[uturn])
                inner = "".join(chip(f, TAGCOL.get(f, "#555")) for f in ltags)
                if lids:
                    inner += chip(f"IDs:{lids}", "#34495e")
                llm_html = f'<div class="llmrow">LLM: {inner or "—"}</div>'
            rows.append(
                f'<div class="row user"><div class="bub userbub">'
                f'<div class="utext">{esc(content)}</div>'
                f'<div class="tags">rule: {tags}</div>{llm_html}</div></div>'
            )
            uturn += 1

    return (
        f'<section id="{sid}">'
        + "".join(head)
        + "".join(meta)
        + '<div class="dialog">'
        + "".join(rows)
        + "</div></section>"
    )


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7;color:#222}
#bar{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;z-index:10}
#bar input{padding:6px 10px;width:320px;border:1px solid #ccc;border-radius:6px}
.meta-top{color:#555;font-size:13px;margin-top:4px}
#index{padding:8px 16px}
#index a{display:inline-block;margin:2px 6px 2px 0;font-size:12px;text-decoration:none;
  padding:2px 8px;border-radius:10px;background:#e8eaed;color:#333}
section{background:#fff;margin:14px 16px;padding:12px 16px;border-radius:10px;
  box-shadow:0 1px 3px rgba(0,0,0,.08)}
.head{font-size:15px;margin-bottom:6px}
.reward{color:#fff;padding:1px 8px;border-radius:10px;font-size:12px;margin-left:8px}
.term{color:#888;font-size:12px;margin-left:8px}
.meta{font-size:13px;color:#444;white-space:pre-wrap;background:#fafafa;
  padding:6px 8px;border-radius:6px;margin:4px 0}
details summary{cursor:pointer;font-size:12px;color:#2980b9}
.dialog{margin-top:8px}
.row{display:flex;margin:6px 0}
.row.user{justify-content:flex-end}
.row.agent{justify-content:flex-start}
.row.tool{justify-content:center}
.bub{max-width:78%;padding:8px 11px;border-radius:12px;font-size:14px;line-height:1.4}
.userbub{background:#dbeafe;border-bottom-right-radius:3px}
.userbub.ctl{background:#fde2e2;color:#922b21;font-family:monospace;font-size:12px}
.agentbub{background:#eceff1;border-bottom-left-radius:3px}
.toolcall{font-family:monospace;font-size:12px;color:#5b6b7a;margin-top:3px}
.toolcall .args{color:#8a97a3}
.toolres{background:#eafaf1;color:#1e6b43;font-family:monospace;font-size:12px;
  border-radius:8px;max-width:88%}
.utext{white-space:pre-wrap}
.tags{margin-top:5px}
.llmrow{margin-top:3px;font-size:11px;color:#666}
.chip{color:#fff;font-size:10px;padding:1px 6px;border-radius:8px;margin-right:3px;
  display:inline-block}
.wc{color:#888;font-size:11px;margin-left:4px}
/* two-column compare */
.cols{display:flex;align-items:flex-start}
.col{flex:1;min-width:0;height:calc(100vh - 58px);overflow-y:auto;border-right:1px solid #ccc}
.colhead{position:sticky;top:0;background:#2c3e50;color:#fff;padding:7px 16px;
  font-size:13px;z-index:5}
.col section{margin:12px}
.col .bub{max-width:88%}
/* aligned pair-by-task layout */
.pair{background:#fff;margin:16px;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.1);
  overflow:hidden}
.pairhead{background:#34495e;color:#fff;padding:7px 14px;font-size:14px;font-weight:bold}
.paircols{display:flex;align-items:flex-start}
.paircol{flex:1;min-width:0;padding:4px 10px;border-right:1px solid #eee}
.paircol .sidelbl{font-size:11px;color:#fff;padding:2px 8px;border-radius:8px;
  display:inline-block;margin:6px 0}
.paircol section{margin:4px 0;box-shadow:none}
.paircol .head{font-size:13px}
.paircol .bub{max-width:92%}
"""


def select(path, task, trial, minr, maxr, sort, limit, mode):
    """Load + filter + sort + truncate one source -> (kind, meta, trajs)."""
    kind, meta, trajs = load(path)
    if task:
        trajs = [t for t in trajs if task in str(t["task_id"])]
    if trial is not None:
        trajs = [t for t in trajs if t["trial"] == trial]
    if minr is not None:
        trajs = [t for t in trajs if (t["reward"] or 0) >= minr]
    if maxr is not None:
        trajs = [
            t for t in trajs if (t["reward"] if t["reward"] is not None else 1) <= maxr
        ]
    if sort == "reward":
        trajs.sort(key=lambda t: t["reward"] if t["reward"] is not None else 1e9)
    elif sort == "adv":
        trajs.sort(key=lambda t: adv_score(t, mode), reverse=True)
    return kind, meta, trajs[:limit]


def hybrid_labels(trajs):
    import sim2real_llm_metrics as L

    tls = [
        R.tau2_user_turns(
            [
                {"role": r, "content": c}
                for r, c, _ in t["messages"]
                if r == "user" and c
            ]
        )
        for t in trajs
    ]
    return L.label_interactions(tls)


def build_column(trajs, prefix, mode, do_hybrid):
    labels = hybrid_labels(trajs) if do_hybrid else [None] * len(trajs)
    links, body = [], []
    for i, t in enumerate(trajs):
        sid = f"{prefix}{i}"
        links.append(
            f'<a href="#{sid}">#{i} {esc(t["task_id"])} '
            f"({('%.2f' % t['reward']) if t['reward'] is not None else '-'})</a>"
        )
        body.append(render_traj(sid, i, t, mode, labels[i]))
    return "".join(links), "".join(body)


FILTER_JS = (
    "<script>function f(){let q=document.getElementById('q').value.toLowerCase();"
    "let u=document.querySelectorAll('.pair');"
    "if(!u.length)u=document.querySelectorAll('section');"
    "for(let s of u){s.style.display=s.innerText.toLowerCase().includes(q)?'':"
    "'none';}}</script>"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "path", help="tau2 results.json or human unified.json (LEFT in compare)"
    )
    ap.add_argument(
        "--right",
        default=None,
        help="second source -> side-by-side compare (RIGHT column)",
    )
    ap.add_argument(
        "--out", default=None, help="output html (default: /tmp/<name>_viz.html)"
    )
    ap.add_argument("--mode", choices=["strict", "faithful"], default="strict")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--task", default=None, help="filter by task_id substring")
    ap.add_argument("--trial", type=int, default=None)
    ap.add_argument("--min-reward", type=float, default=None)
    ap.add_argument("--max-reward", type=float, default=None)
    ap.add_argument(
        "--sort",
        choices=["index", "reward", "adv"],
        default="index",
        help="LEFT sort: index | reward (hardest first) | adv (most adversarial first)",
    )
    # per-side overrides for compare (fall back to --sort/--limit/--task)
    ap.add_argument("--right-sort", choices=["index", "reward", "adv"], default=None)
    ap.add_argument("--right-limit", type=int, default=None)
    ap.add_argument("--right-task", default=None)
    ap.add_argument(
        "--align",
        action="store_true",
        help="with --right: pair the two sources by (task_id, trial) so the "
        "SAME task is shown side-by-side (e.g. seed vs candidate)",
    )
    ap.add_argument(
        "--labels",
        default="◀ left,▶ right",
        help="comma-separated short labels for the two aligned sides",
    )
    ap.add_argument(
        "--hybrid",
        action="store_true",
        help="overlay LLM-rubric labels per user turn (uses disk cache)",
    )
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    M.set_mode(args.mode)

    if (
        args.right and args.align
    ):  # ---- pair by (task_id, trial): same task L vs R ----
        _, lmeta, ltr = select(
            args.path,
            args.task,
            args.trial,
            args.min_reward,
            args.max_reward,
            "index",
            10**9,
            args.mode,
        )
        _, rmeta, rtr = select(
            args.right,
            args.right_task or args.task,
            args.trial,
            args.min_reward,
            args.max_reward,
            "index",
            10**9,
            args.mode,
        )
        lmap = {(t["task_id"], t["trial"]): t for t in ltr}
        rmap = {(t["task_id"], t["trial"]): t for t in rtr}
        keys = sorted(set(lmap) & set(rmap), key=lambda k: (str(k[0]), k[1]))[
            : args.limit
        ]
        if not keys:
            sys.exit("no common (task_id, trial) between the two sources to align")
        llab = (
            hybrid_labels([lmap[k] for k in keys])
            if args.hybrid
            else [None] * len(keys)
        )
        rlab = (
            hybrid_labels([rmap[k] for k in keys])
            if args.hybrid
            else [None] * len(keys)
        )
        sl, sr = (args.labels.split(",") + ["left", "right"])[:2]
        print(
            f"LEFT  {lmeta}\nRIGHT {rmeta}\naligned on {len(keys)} common "
            f"(task,trial) pairs",
            flush=True,
        )
        blocks = []
        for i, k in enumerate(keys):
            lcard = render_traj(f"l{i}", i, lmap[k], args.mode, llab[i])
            rcard = render_traj(f"r{i}", i, rmap[k], args.mode, rlab[i])
            blocks.append(
                f'<div class="pair"><div class="pairhead">{esc(k[0])} · trial {k[1]} '
                f"&nbsp; (L reward={lmap[k]['reward']}, R reward={rmap[k]['reward']})</div>"
                f'<div class="paircols">'
                f'<div class="paircol"><span class="sidelbl" style="background:#2980b9">'
                f"{esc(sl)}</span>{lcard}</div>"
                f'<div class="paircol"><span class="sidelbl" style="background:#c0392b">'
                f"{esc(sr)}</span>{rcard}</div></div></div>"
            )
        page = (
            f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style>"
            f"</head><body><div id='bar'><input id='q' placeholder='filter text… "
            f"(hides non-matching task pairs)' oninput=\"f()\"><div class='meta-top'>"
            f"L={esc(args.path)} &nbsp;|&nbsp; R={esc(args.right)}</div></div>"
            f"{''.join(blocks)}{FILTER_JS}</body></html>"
        )
        default_name = (
            os.path.splitext(os.path.basename(args.path))[0]
            + "_alignVS_"
            + os.path.splitext(os.path.basename(args.right))[0]
        )
    elif args.right:  # ---- side-by-side compare (independent columns) ----
        lk, lmeta, ltr = select(
            args.path,
            args.task,
            args.trial,
            args.min_reward,
            args.max_reward,
            args.sort,
            args.limit,
            args.mode,
        )
        rk, rmeta, rtr = select(
            args.right,
            args.right_task or args.task,
            args.trial,
            args.min_reward,
            args.max_reward,
            args.right_sort or args.sort,
            args.right_limit or args.limit,
            args.mode,
        )
        print(
            f"LEFT  [{lk}] {lmeta}\n  -> {len(ltr)} trajs (sort={args.sort})",
            flush=True,
        )
        print(
            f"RIGHT [{rk}] {rmeta}\n  -> {len(rtr)} trajs "
            f"(sort={args.right_sort or args.sort})",
            flush=True,
        )
        _, lbody = build_column(ltr, "l", args.mode, args.hybrid)
        _, rbody = build_column(rtr, "r", args.mode, args.hybrid)
        page = (
            f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style>"
            f"</head><body><div id='bar'><input id='q' placeholder='filter text… "
            f"(both columns)' oninput=\"f()\"></div><div class='cols'>"
            f"<div class='col'><div class='colhead'>◀ {esc(args.path)}<br>{esc(lmeta)}"
            f"</div>{lbody}</div>"
            f"<div class='col'><div class='colhead'>▶ {esc(args.right)}<br>{esc(rmeta)}"
            f"</div>{rbody}</div></div>{FILTER_JS}</body></html>"
        )
        default_name = (
            os.path.splitext(os.path.basename(args.path))[0]
            + "_vs_"
            + os.path.splitext(os.path.basename(args.right))[0]
        )
    else:  # ---- single column ----
        kind, meta, trajs = select(
            args.path,
            args.task,
            args.trial,
            args.min_reward,
            args.max_reward,
            args.sort,
            args.limit,
            args.mode,
        )
        print(
            f"[{kind}] {meta}\nrendering {len(trajs)} trajectories "
            f"(mode={args.mode}, sort={args.sort})",
            flush=True,
        )
        links, body = build_column(trajs, "t", args.mode, args.hybrid)
        page = (
            f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style>"
            f"</head><body><div id='bar'><input id='q' placeholder='filter text… "
            f"(live)' oninput=\"f()\"><div class='meta-top'>{esc(args.path)} — "
            f"{esc(meta)}</div></div><div id='index'>{links}</div>{body}{FILTER_JS}"
            "</body></html>"
        )
        default_name = os.path.splitext(os.path.basename(args.path))[0]

    out = args.out or f"/tmp/{default_name}_viz.html"
    with open(out, "w") as fh:
        fh.write(page)
    print(f"-> {out}")
    if not args.no_open:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
