import json, re, statistics

lines = open("data/backend_tasks/test.jsonl").readlines()
tasks = [json.loads(l) for l in lines]

def complexity(t):
    gt = t["ground_truth"]
    hidden = t["hidden_information"]
    # non-blank lines of reference function
    loc = len([l for l in gt.splitlines() if l.strip()])
    # branch / logic complexity proxy
    branch_tokens = ["if ", "elif ", "for ", "while ", " and ", " or ", "try:", "except", "not ", " in "]
    branches = sum(gt.count(tok) for tok in branch_tokens)
    # param count from signature
    m = re.search(r"def \w+\(([^)]*)\)", gt)
    params = len([p for p in (m.group(1).split(",") if m else []) if p.strip()]) if m else 0
    has_import = 1 if re.search(r"\bimport\b", gt) else 0
    chars = len(gt)
    hidden_chars = len(hidden)
    # composite difficulty score
    score = loc + branches * 3 + params * 2 + has_import * 6 + hidden_chars / 200.0
    return dict(loc=loc, branches=branches, params=params, has_import=has_import,
                chars=chars, hidden_chars=hidden_chars, score=score)

scored = []
for t in tasks:
    c = complexity(t)
    scored.append((c["score"], c, t))
scored.sort(key=lambda x: x[0], reverse=True)

# take top 110 hardest
top = scored[:110]
with open("data/backend_tasks/test_110_hard.jsonl", "w") as fb:
    for _, _, t in top:
        fb.write(json.dumps(t) + "\n")

scores = [s for s, _, _ in top]
locs = [c["loc"] for _, c, _ in top]
branches = [c["branches"] for _, c, _ in top]
params = [c["params"] for _, c, _ in top]
print(f"selected top-110 hardest -> data/backend_tasks/test_110_hard.jsonl")
print(f"score: min={scores[-1]:.1f} med={statistics.median(scores):.1f} max={scores[0]:.1f}")
print(f"ref LOC: med={statistics.median(locs)} max={max(locs)}")
print(f"ref branches: med={statistics.median(branches)} max={max(branches)}")
print(f"params: med={statistics.median(params)} max={max(params)}")
print(f"uses import: {sum(1 for _,c,_ in top if c['has_import'])}/110")
print("\n3 hardest examples:")
for s, c, t in top[:3]:
    m = re.search(r"def (\w+)\(", t["ground_truth"])
    print(f"  [score={s:.1f} loc={c['loc']} branches={c['branches']} params={c['params']}] {(m.group(1) if m else '?')}: {t['problem_description'][:110]}")
