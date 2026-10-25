"""Demo driver: run a few sharded multi-turn conversations per task, dump traces to MD."""
import json, os, sys, traceback
from datetime import datetime

# --- minimax routing: agent(minimax-m3)→:4114 openai→fireworks shim, user/system
#     (gpt-5.x)→标准 OpenAI。patch_model_openai() 替换 model_openai 模块单例 client，
#     按 model 名路由，故 assistant/user/system 可用不同 model。必须在
#     `from simulator_sharded import ...`（间接 import model_openai）之前生效。---
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
try:
    import minimax_routing
    minimax_routing.patch_model_openai()
except Exception as _e:  # noqa: BLE001
    print(f"[run_demo] WARNING: minimax_routing patch failed: {_e}", file=sys.stderr)

from simulator_sharded import ConversationSimulatorSharded

MODEL = "gpt-4o-mini"
ASSISTANT_MODEL = None  # if set, overrides MODEL for the assistant (agent under test)
USER_MODEL = None       # if set, overrides MODEL for the UserAgent
SYSTEM_MODEL = None     # if set, overrides MODEL for the SystemAgent (verify/extract)
N_PER_TASK = 2
LOG_FOLDER = "logs_demo"
USER_PERSONA = "prompts/user_agent.txt"
REASONING_EFFORT = None
ASSISTANT_REASONING_EFFORT = None  # if set, agent effort overrides REASONING_EFFORT
USER_REASONING_EFFORT = None       # if set, user-sim effort overrides REASONING_EFFORT
# database skipped: needs Spider DB files (not downloaded).
TASKS = ["math", "actions", "code", "data2text", "summary", "translation"]


def load_samples(task_name):
    fn = "data/sharded_translation.json" if task_name == "translation" else "data/sharded_instructions_600.json"
    with open(fn) as f:
        data = json.load(f)
    return [d for d in data if d["task"] == task_name], fn


def run_one(sample, fn):
    sim = ConversationSimulatorSharded(
        sample, assistant_model=(ASSISTANT_MODEL or MODEL), system_model=(SYSTEM_MODEL or MODEL),
        user_model=(USER_MODEL or MODEL),
        assistant_temperature=1.0, user_temperature=1.0, dataset_fn=fn, log_folder=LOG_FOLDER,
        user_persona=USER_PERSONA,
        reasoning_effort=REASONING_EFFORT,
        assistant_reasoning_effort=ASSISTANT_REASONING_EFFORT,
        user_reasoning_effort=USER_REASONING_EFFORT,
    )
    is_correct, score = sim.run(verbose=False, save_log=True)
    return sim.trace, is_correct, score


def trace_to_md(trace, task_id, is_correct, score):
    lines = [f"### {task_id}  —  correct={is_correct}  score={score}", ""]
    for msg in trace:
        role = msg["role"]
        content = msg["content"]
        ts = msg.get("timestamp", "")
        if role == "system":
            lines.append(f"**[system]** {content}")
            lines.append("")
        elif role == "user":
            lines.append(f"**[user]** {content}")
            lines.append("")
        elif role == "assistant":
            lines.append(f"**[assistant]**")
            lines.append("")
            lines.append("```")
            lines.append(content)
            lines.append("```")
            lines.append("")
        elif role == "log":
            t = content.get("type")
            if t == "shard_revealed":
                lines.append(f"> 🔵 log: shard revealed — id={content.get('shard_id')}")
            elif t == "hint_revealed":
                lines.append(f"> 🔵 log: hint revealed — id={content.get('shard_id')}")
            elif t == "system-verification":
                resp = content.get("response", {})
                lines.append(f"> 🟣 log: system-verification — {resp.get('response_type')}")
            elif t == "answer-evaluation":
                ea = content.get("exact_answer", "")
                lines.append(f"> 🟢 log: answer-evaluation — correct={content.get('is_correct')} score={content.get('score')}")
                lines.append("> extracted answer:")
                lines.append("```")
                lines.append(str(ea)[:2000])
                lines.append("```")
            elif t == "conversation-completed":
                lines.append(f"> ✅ log: conversation completed — correct={content.get('is_correct')}")
            else:
                lines.append(f"> log: {t}")
            lines.append("")
    return "\n".join(lines)


def main():
    global MODEL, ASSISTANT_MODEL, USER_MODEL, SYSTEM_MODEL, LOG_FOLDER, USER_PERSONA, REASONING_EFFORT, ASSISTANT_REASONING_EFFORT, USER_REASONING_EFFORT
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL, help="model for assistant/system/user (fallback when --assistant/--user/--system-model unset)")
    parser.add_argument("--assistant-model", default=None, help="model for the assistant (agent under test); overrides --model")
    parser.add_argument("--user-model", default=None, help="model for the UserAgent; overrides --model")
    parser.add_argument("--system-model", default=None, help="model for the SystemAgent; overrides --model")
    parser.add_argument("--tag", default=None, help="output filename tag (default: derived from model+persona+effort)")
    parser.add_argument("--n", type=int, default=N_PER_TASK)
    parser.add_argument("--tasks", nargs="+", default=TASKS, help="subset of tasks to run")
    parser.add_argument("--user_persona", default="prompts/user_agent.txt", help="path to user persona prompt")
    parser.add_argument("--reasoning_effort", default=None, choices=["low","medium","high","xhigh","max"], help="reasoning effort (shared fallback for assistant+user)")
    parser.add_argument("--assistant_reasoning_effort", default=None, choices=["low","medium","high","xhigh","max"], help="reasoning effort for the assistant (agent under test); overrides --reasoning_effort")
    parser.add_argument("--user_reasoning_effort", default=None, choices=["low","medium","high","xhigh","max"], help="reasoning effort for the UserAgent; overrides --reasoning_effort")
    parser.add_argument("--workers", type=int, default=1, help="parallel workers (1=serial)")
    parser.add_argument("--task-ids", default=None, help="comma-separated task_ids to run (overrides --n filtering)")
    parser.add_argument("--output-json", default=None, help="output json path (default: derived from tag)")
    parser.add_argument("--n-trials", type=int, default=1, help="repeat each task_id N times (multi-trial sampling; records get trial_idx)")
    args = parser.parse_args()

    MODEL = args.model
    ASSISTANT_MODEL = args.assistant_model
    USER_MODEL = args.user_model
    SYSTEM_MODEL = args.system_model
    USER_PERSONA = args.user_persona
    REASONING_EFFORT = args.reasoning_effort
    ASSISTANT_REASONING_EFFORT = args.assistant_reasoning_effort
    USER_REASONING_EFFORT = args.user_reasoning_effort
    persona_tag = os.path.basename(args.user_persona).replace("user_agent", "").replace(".txt", "").strip("_")
    if not persona_tag:
        persona_tag = "lazy"
    model_tag = (ASSISTANT_MODEL or MODEL).replace(".", "").replace("/", "").replace("-", "")
    _effort_for_tag = args.assistant_reasoning_effort or args.reasoning_effort
    effort_tag = f"_{_effort_for_tag}" if _effort_for_tag else ""
    tag = args.tag or f"{model_tag}{effort_tag}_{persona_tag}"
    LOG_FOLDER = f"logs_demo_{tag}"
    json_out = args.output_json or f"demo_runs_{tag}.json"
    md_out = f"conversations_demo_{tag}.md"

    task_id_set = set(t.strip() for t in args.task_ids.split(",") if t.strip()) if args.task_ids else None

    out = {"runs": [], "md": []}
    summary_rows = []
    todos = []
    n_trials = max(1, args.n_trials)
    for task_name in args.tasks:
        try:
            samples, fn = load_samples(task_name)
        except Exception as e:
            print(f"[{task_name}] load_samples failed: {e}")
            continue
        if task_id_set is not None:
            for sample in samples:
                if sample["task_id"] in task_id_set:
                    for ti in range(n_trials):
                        todos.append((task_name, sample, fn, ti))
        else:
            for sample in samples[:args.n]:
                for ti in range(n_trials):
                    todos.append((task_name, sample, fn, ti))

    # resume: read back already-completed (task_id, trial_idx) from existing json_out
    # so a timeout-restarted subprocess skips finished runs instead of redoing them.
    # Errored runs are dropped so they get retried (e.g. transient API slowness);
    # only successfully-completed runs are kept and skipped on rerun.
    done_keys: set = set()
    results: list = []
    if os.path.exists(json_out):
        try:
            _prev = json.load(open(json_out))
            _prev_runs = list(_prev.get("runs", []))
            results = [r for r in _prev_runs if "error" not in r]
            for _r in results:
                done_keys.add((_r.get("task_id"), _r.get("trial_idx", 0)))
        except Exception:
            results = []
    _n_before = len(todos)
    todos = [t for t in todos if (t[1]["task_id"], t[3]) not in done_keys]
    if done_keys:
        print(f"[resume] {len(done_keys)} runs already done; {_n_before - len(todos)}/{_n_before} skipped, {len(todos)} remaining", flush=True)

    def run_wrapper(todo):
        task_name, sample, fn, trial_idx = todo
        task_id = sample["task_id"]
        try:
            trace, is_correct, score = run_one(sample, fn)
            n_turns = sum(1 for m in trace if m["role"] == "assistant")
            n_shards = len(sample["shards"])
            return {"task": task_name, "task_id": task_id, "trial_idx": trial_idx,
                    "trace": trace, "is_correct": is_correct,
                    "score": score, "turns": n_turns, "shards": n_shards}
        except Exception as e:
            return {"task": task_name, "task_id": task_id, "trial_idx": trial_idx,
                    "error": str(e), "traceback": traceback.format_exc()}

    def save_partial():
        """Incremental save so a hung/kill mid-run keeps finished runs."""
        _d = os.path.dirname(json_out)
        if _d:
            os.makedirs(_d, exist_ok=True)
        srows = []
        for r in results:
            if "error" in r:
                srows.append({"task": r["task"], "task_id": r["task_id"], "error": r["error"]})
            else:
                srows.append({"task": r["task"], "task_id": r["task_id"], "turns": r.get("turns"),
                              "shards": r.get("shards"), "correct": r.get("is_correct"), "score": r.get("score")})
        with open(json_out, "w") as f:
            json.dump({"summary": srows, "runs": results}, f, indent=2, ensure_ascii=False)

    if args.workers and args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import tqdm
        results = list(results)  # keep resume-read runs; append new ones below
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_wrapper, t) for t in todos]
            for fut in tqdm.tqdm(as_completed(futs), total=len(todos)):
                r = fut.result()
                results.append(r)
                if "error" in r:
                    print(f"[{r['task']}] {r['task_id']} FAILED: {r['error']}")
                else:
                    print(f"[{r['task']}] {r['task_id']} done — turns={r.get('turns')} shards={r.get('shards')} correct={r.get('is_correct')} score={r.get('score')}")
                save_partial()
    else:
        results = list(results)  # keep resume-read runs; append new ones below
        for t in todos:
            r = run_wrapper(t)
            results.append(r)
            if "error" in r:
                print(f"[{r['task']}] {r['task_id']} FAILED: {r['error']}")
            else:
                print(f"[{r['task']}] {r['task_id']} done — turns={r.get('turns')} shards={r.get('shards')} correct={r.get('is_correct')} score={r.get('score')}")
            save_partial()

    for r in results:
        if "error" in r:
            summary_rows.append({"task": r["task"], "task_id": r["task_id"], "error": r["error"]})
        else:
            summary_rows.append({"task": r["task"], "task_id": r["task_id"], "turns": r.get("turns"),
                                 "shards": r.get("shards"), "correct": r.get("is_correct"), "score": r.get("score")})
        out["runs"].append(r)

    # write JSON
    _d = os.path.dirname(json_out)
    if _d:
        os.makedirs(_d, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump({"summary": summary_rows, "runs": out["runs"]}, f, indent=2, ensure_ascii=False)

    # build MD
    md = []
    md.append("# Lost in Conversation — 模拟对话样例\n")
    md.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append(f"模型: assistant/system/user 均为 `{MODEL}`  ")
    md.append(f"条件: sharded(多轮,逐轮揭示 shard)  ")
    md.append(f"User persona: `{USER_PERSONA}`  ")
    md.append(f"Reasoning effort: `{REASONING_EFFORT}`  ")
    md.append(f"每个任务样本数: {args.n}  ")
    md.append(f"任务: {', '.join(args.tasks)}\n")
    md.append("## 概览\n")
    md.append("| task | task_id | shard数 | 助手轮数 | 是否正确 | 得分 |")
    md.append("|---|---|---|---|---|---|")
    for r in summary_rows:
        if "error" in r:
            md.append(f"| {r['task']} | {r['task_id']} | — | — | ❌ 失败 | — |")
        else:
            md.append(f"| {r['task']} | {r['task_id']} | {r['shards']} | {r['turns']} | {r['correct']} | {r['score']} |")
    md.append("")
    md.append("## 完整对话\n")
    for run in out["runs"]:
        if "error" in run:
            md.append(f"### {run['task']} / {run['task_id']} — ❌ 运行失败\n")
            md.append("```")
            md.append(run.get("traceback", run["error"]))
            md.append("```\n")
            continue
        md.append(trace_to_md(run["trace"], run["task_id"], run["is_correct"], run["score"]))
        md.append("---\n")

    with open(md_out, "w") as f:
        f.write("\n".join(md))
    print(f"\n=== 写入 {md_out} 与 {json_out} ===")


if __name__ == "__main__":
    main()
