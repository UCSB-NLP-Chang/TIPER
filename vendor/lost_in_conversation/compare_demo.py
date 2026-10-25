"""Compare demo runs across models. Usage: python compare_demo.py tag1 tag2 ..."""
import json, sys, os

def load(tag):
    fn = f"demo_runs_{tag}.json"
    if not os.path.exists(fn):
        return None
    with open(fn) as f:
        return json.load(f)["summary"]

def main():
    tags = sys.argv[1:] or ["gpt4omini", "gpt54"]
    data = {t: load(t) for t in tags}
    data = {t: v for t, v in data.items() if v is not None}
    if not data:
        print("no demo_runs_*.json found"); return

    # build task_id -> {tag: row}
    by_id = {}
    for tag, rows in data.items():
        for r in rows:
            by_id.setdefault(r["task_id"], {})[tag] = r

    # group by task
    from collections import defaultdict
    by_task = defaultdict(list)
    for tag, rows in data.items():
        for r in rows:
            by_task[r["task"]].append(r)

    print(f"# 对比: {' vs '.join(data.keys())}\n")
    # per-sample table
    header = "| task | task_id | " + " | ".join(f"{t} score" for t in data) + " | " + " | ".join(f"{t} correct" for t in data) + " |"
    sep = "|" + "---|" * (2 + 2*len(data))
    print(header)
    print(sep)
    for tid, tagrows in sorted(by_id.items()):
        task = next(iter(tagrows.values()))["task"]
        scores = []
        corrects = []
        for t in data:
            r = tagrows.get(t)
            if r is None:
                scores.append("—"); corrects.append("—")
            elif "error" in r:
                scores.append("ERR"); corrects.append("ERR")
            else:
                s = r.get("score")
                scores.append(f"{s:.3f}" if isinstance(s, (int, float)) else str(s))
                corrects.append(str(r.get("correct")))
        print(f"| {task} | {tid} | " + " | ".join(scores) + " | " + " | ".join(corrects) + " |")

    # per-task averages
    print("\n## 每任务平均得分\n")
    print("| task | " + " | ".join(f"{t} 均分" for t in data) + " |")
    print("|---|" + "---|" * len(data))
    for task, rows in sorted(by_task.items()):
        cells = []
        for t in data:
            trs = [r for r in rows if r.get("task") == task and t in {tt: rr for tt, rr in [(t, r)] for r in [r]}]
            # simpler: filter data[t] for this task
            task_rows = [r for r in data[t] if r["task"] == task and "score" in r and isinstance(r["score"], (int,float))]
            if task_rows:
                avg = sum(r["score"] for r in task_rows) / len(task_rows)
                cells.append(f"{avg:.3f}")
            else:
                cells.append("—")
        print(f"| {task} | " + " | ".join(cells) + " |")

if __name__ == "__main__":
    main()
