"""Smoke-test weird personas: run a few real conversations per domain and check
(a) the user_agent LLM actually maintains the weird format, (b) validity holds
(all shards revealed, exact values), (c) whether weirdness moves difficulty.

Uses LICPersonaAdapter (same backend as pgepa) on a few TRAIN samples (does not
touch test split). Prints per-domain validity + correctness + sample user msgs.

Usage:
  source ./load_env.sh
  ../.venv/bin/python smoke_weird.py                 # 4 domains x 4 train samples x 1 trial
  ../.venv/bin/python smoke_weird.py --n 3 --domains math actions
"""
import sys, json, os, argparse

for _ln in open('../.env'):
    _ln = _ln.strip()
    if _ln and '=' in _ln and not _ln.startswith('#'):
        _k, _v = _ln.split('=', 1)
        if _v and not _v.startswith('<'):
            os.environ.setdefault(_k, _v)

sys.path.insert(0, '.'); sys.path.insert(0, '../scripts/persona_opt')
from lic_persona_adapter import LICPersonaAdapter, LICTask  # noqa: E402

WEIRD_FILES = {d: f'/tmp/weird_{d}.json' for d in ['math', 'code', 'database', 'actions']}


def load_train_samples(d, n):
    sp = json.load(open(f'data/lic_split_{d}.json'))
    tids = sp['train'][d][:n]
    fn = 'data/sharded_instructions_600.json'
    data = json.load(open(fn))
    tset = set(tids)
    samples = [x for x in data if x.get('task') == d and x.get('task_id') in tset]
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domains', nargs='+', default=['math', 'code', 'database', 'actions'])
    ap.add_argument('--n', type=int, default=4, help='samples per domain')
    ap.add_argument('--trials', type=int, default=1)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    for d in args.domains:
        if not os.path.exists(WEIRD_FILES[d]):
            print(f"[skip] {d}: no /tmp/weird_{d}.json (run test_weird_ctx.py first)")
            continue
        persona_text = json.load(open(WEIRD_FILES[d]))['persona_text']
        samples = load_train_samples(d, args.n)
        if len(samples) < args.n:
            print(f"[warn] {d}: only {len(samples)} train samples found")
        print(f"\n{'='*70}\n  {d}: {len(samples)} samples x {args.trials} trial  "
              f"(weird persona: {WEIRD_FILES[d]})\n{'='*70}")

        batch = [LICTask(s['task_id'], d) for s in samples]
        adapter = LICPersonaAdapter(
            persona_id=f'smoke_weird_{d}', agent_model='gpt-5.4',
            reasoning_effort='high', domains=[d], n_workers=args.workers,
            lic_dir='.', num_trials=args.trials)
        art = adapter.run_simulations(batch, {'persona_text': persona_text})

        # per-task: validity (all shards revealed?) + is_correct + user msg samples
        n_valid = n_correct = n_total = 0
        for tid, runs in art.runs_by_task.items():
            for r in runs:
                if 'error' in r:
                    print(f"  {tid}: INFRA_ERROR {str(r['error'])[:80]}")
                    continue
                n_total += 1
                # count shard_revealed vs expected nshards
                revealed = sum(1 for m in r.get('trace', [])
                               if m.get('role') == 'log'
                               and (m.get('content') or {}).get('type') == 'shard_revealed')
                nshards = r.get('shards') or len([s for s in samples if s['task_id'] == tid][0].get('shards', []))
                valid = revealed >= nshards
                correct = r.get('is_correct')
                n_valid += int(valid)
                n_correct += int(correct is True)
                flag = '' if valid else ' [INVALID: revealed {}/{} shards]'.format(revealed, nshards)
                print(f"  {tid}: correct={correct} revealed={revealed}/{nshards}{flag}")

        print(f"  -> validity {n_valid}/{n_total}  agent correct {n_correct}/{n_total}")

        # dump 1-2 user messages to inspect weird-format maintenance
        print(f"  --- sample user messages (weird format check) ---")
        shown = 0
        for tid, runs in art.runs_by_task.items():
            if shown >= 2: break
            for r in runs:
                if 'error' in r: continue
                for m in r.get('trace', []):
                    if m.get('role') == 'user':
                        print(f"  [{tid}] {str(m.get('content',''))[:200]}")
                        break
                shown += 1
                break


if __name__ == '__main__':
    main()
