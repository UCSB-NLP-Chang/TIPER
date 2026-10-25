import json, statistics

def stats(path, label):
    ds = [json.loads(l) for l in open(path)]
    turns = [len(d["dialogue_history"]) for d in ds]
    rewards = [d.get("reward") for d in ds if d.get("reward") is not None]
    turns_sorted = sorted(turns)
    n = len(turns)
    def pct(p): return turns_sorted[min(int(p/100*n), n-1)]
    print(f"=== {label} ===")
    print(f"  n={n}  turns: med={statistics.median(turns)} mean={statistics.mean(turns):.1f} min={min(turns)} max={max(turns)}")
    print(f"  turn percentiles: p25={pct(25)} p50={pct(50)} p75={pct(75)} p90={pct(90)} p95={pct(95)}")
    print(f"  >=20 turns: {sum(1 for x in turns if x>=20)}/{n} ({sum(1 for x in turns if x>=20)/n*100:.0f}%)")
    print(f"  >=30 turns: {sum(1 for x in turns if x>=30)}/{n}")
    if rewards:
        print(f"  reward: pass_rate={sum(rewards)/len(rewards):.3f} fully_correct={sum(1 for r in rewards if r==1.0)}/{len(rewards)}")

stats("out_hard_long40.jsonl", "110 hardest tasks, long-interrogation prompt, max_steps=40")
