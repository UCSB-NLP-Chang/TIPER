"""Test-set evaluation: best persona vs seed_lazy baseline, n_trials=3.

For each domain, runs the best persona (from pgepa_state.json) and the seed_lazy
baseline on the test split with 3 trials per sample, then computes:
  - avg@3  : mean over samples of (fraction of 3 trials the agent got correct)
            = agent's average correctness under this persona
  - pass^3 : fraction of samples where the agent got ALL 3 trials correct
A lower number = the persona is better at making the agent fail (harder).

Usage:
  python test_set_eval.py                      # all 4 domains, best vs seed
  python test_set_eval.py --domains math code  # subset
  python test_set_eval.py --n-trials 3 --workers 6
"""
import sys, json, os, argparse

# load ../.env (skip <placeholder> values)
for _ln in open('../.env'):
    _ln = _ln.strip()
    if _ln and '=' in _ln and not _ln.startswith('#'):
        _k, _v = _ln.split('=', 1)
        if _v and not _v.startswith('<'):
            os.environ.setdefault(_k, _v)

sys.path.insert(0, '.'); sys.path.insert(0, '../scripts/persona_opt')
from lic_persona_adapter import LICPersonaAdapter, LICTask  # noqa: E402

DOMAINS = ['math', 'code', 'database', 'actions']
RUN_DIR_TMPL = 'data/persona_opt/pgepa_runs/lic_{d}{suffix}'


def load_split_test(d):
    s = json.load(open(f'data/lic_split_{d}.json'))
    return s['test'][d]


def load_best_persona(d, suffix=''):
    s = json.load(open(f'{RUN_DIR_TMPL.format(d=d, suffix=suffix)}/pgepa_state.json'))
    bi = s['best_idx']
    bc = next(c for c in s['candidates'] if c['idx'] == bi)
    return bc['persona_text'], bi, bc.get('gen', '?')


def load_test_samples(d, task_ids):
    fn = ('data/sharded_translation.json' if d == 'translation'
          else 'data/sharded_instructions_600.json')
    with open(fn) as f:
        data = json.load(f)
    tids = set(task_ids)
    return [x for x in data if x['task'] == d and x['task_id'] in tids]


def eval_persona(d, persona_text, tag, n_trials, workers):
    task_ids = load_split_test(d)
    samples = load_test_samples(d, task_ids)
    missing = set(task_ids) - {s['task_id'] for s in samples}
    if missing:
        print(f"  [warn] {d}: {len(missing)} test ids not found in dataset: {list(missing)[:3]}")
    batch = [LICTask(s["task_id"], d) for s in samples]
    adapter = LICPersonaAdapter(
        persona_id=f'test_{d}_{tag}', agent_model='gpt-5.4',
        reasoning_effort='high', domains=[d], n_workers=workers,
        lic_dir='.', num_trials=n_trials)
    art = adapter.run_simulations(batch, {'persona_text': persona_text})
    # tid -> list of is_correct (one per trial), excluding errored runs
    results = {}
    for tid, runs in art.runs_by_task.items():
        cs = [r.get('is_correct') for r in runs if 'error' not in r]
        cs = [c if c is not None else False for c in cs]
        results[tid] = cs
    return results


def metrics(results, n_trials):
    avg3, pass3, passat3 = [], [], []
    for tid, cs in results.items():
        if not cs:
            continue
        rate = sum(cs) / len(cs)
        avg3.append(rate)
        pass3.append(1.0 if all(cs) else 0.0)
        passat3.append(1.0 if any(cs) else 0.0)
    n = len(avg3)
    return {
        'n_samples': n,
        'n_runs': sum(len(cs) for cs in results.values()),
        'avg@3': (sum(avg3) / n) if n else None,        # mean correctness (higher = agent better)
        'pass^3': (sum(pass3) / n) if n else None,      # all-3-correct fraction
        'pass@3': (sum(passat3) / n) if n else None,    # any-1-correct fraction
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domains', nargs='+', default=DOMAINS)
    ap.add_argument('--n-trials', type=int, default=3)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed-file', default='seed_lazy.txt')
    ap.add_argument('--run-suffix', default='', help='suffix on run-dir (e.g. _v2)')
    ap.add_argument('--out', default=None, help='output json (default by suffix)')
    args = ap.parse_args()

    if args.out is None:
        args.out = f'data/persona_opt/test_set_results{args.run_suffix}.json'

    seed_text = open(args.seed_file).read()
    rows = []
    for d in args.domains:
        print(f"\n{'='*60}\n  DOMAIN: {d}  (test split, {args.n_trials} trials, run{args.run_suffix})\n{'='*60}")
        best_text, best_idx, best_gen = load_best_persona(d, args.run_suffix)
        print(f"  best persona: idx{best_idx} (gen {best_gen})")

        # personas to evaluate: best vs no-persona (neutral) baseline.
        # neutral = empty persona_text -> wrap_persona yields pure _INVARIANTS (neutral user)
        tags = [('best', best_text, best_idx)]
        tags.append(('neutral', '', None))

        for tag, ptext, bidx in tags:
            print(f"\n  --- {d} / {tag} ---")
            res = eval_persona(d, ptext, tag, args.n_trials, args.workers)
            m = metrics(res, args.n_trials)
            print(f"    n={m['n_samples']} runs={m['n_runs']}  "
                  f"avg@3={m['avg@3']:.3f}  pass^3={m['pass^3']:.3f}  pass@3={m['pass@3']:.3f}")
            rows.append({'domain': d, 'persona': tag, 'best_idx': bidx,
                         **m, 'results': {k: v for k, v in res.items()}})

    # summary table
    print(f"\n{'='*70}\n  SUMMARY (agent correctness — lower = persona harder)\n{'='*70}")
    print(f"  {'domain':<10} {'persona':<8} {'n':<4} {'avg@3':<8} {'pass^3':<8} {'pass@3':<8}")
    for r in rows:
        print(f"  {r['domain']:<10} {r['persona']:<8} {r['n_samples']:<4} "
              f"{r['avg@3']:.3f}    {r['pass^3']:.3f}    {r['pass@3']:.3f}")
    # deltas vs neutral (no-persona baseline)
    print(f"\n  delta vs neutral (neutral - best) >0 means best persona made agent worse (good):")
    for d in args.domains:
        s = next((r for r in rows if r['domain'] == d and r['persona'] == 'neutral'), None)
        if not s: continue
        b = next((r for r in rows if r['domain'] == d and r['persona'] == 'best'), None)
        if b:
            print(f"  {d:<10} avg@3 Δ={s['avg@3']-b['avg@3']:+.3f}  "
                  f"pass^3 Δ={s['pass^3']-b['pass^3']:+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n  written -> {args.out}")


if __name__ == '__main__':
    main()
