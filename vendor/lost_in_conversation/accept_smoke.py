"""Accept-path smoke: verify the aggregate_artifacts calls made by
_run_full_valset_confirm (4a difficulty-only + 4b eval_existing_with_validity)
work end-to-end on the LIC adapter with the tau-compat kwargs."""
import sys, json, os
for _envp in ('../.env', '../../.env'):
    if os.path.exists(_envp):
        for _ln in open(_envp):
            _ln = _ln.strip()
            if _ln and '=' in _ln and not _ln.startswith('#'):
                _k, _v = _ln.split('=', 1)
                if _v and not _v.startswith('<'):
                    os.environ.setdefault(_k, _v)
        break
sys.path.insert(0, '.'); sys.path.insert(0, '../scripts/persona_opt')
from lic_persona_adapter import LICPersonaAdapter, LICTask

s = json.load(open('data/lic_split.json'))
# 2 val tasks (math + code) to exercise the accept-path aggregate calls
batch = [LICTask(s['val']['math'][0], 'math'), LICTask(s['val']['code'][0], 'code')]

a = LICPersonaAdapter(persona_id='acc_smoke', agent_model='gpt-5.4',
                     reasoning_effort='high', domains=['math', 'code'],
                     n_workers=2, lic_dir='.')
cand = {'persona_text': 'You tend to answer in vague, brief terms and let the '
        'assistant pull details out of you one at a time; you never volunteer '
        'constraints unless directly asked.'}
print(f"accept-smoke: 2 tasks, running sim ONCE then both aggregate paths", flush=True)

# one sim pass, reused for both aggregate paths (mirrors _run_full_valset_confirm)
art = a.run_simulations(batch, cand)
print(f"  run_simulations ok: {len(art.runs_by_task)} runs", flush=True)

# 4a: difficulty-only (capture_traces=False, should_judge=False, failed_only=False)
eb1 = a.aggregate_artifacts(batch, art, capture_traces=False,
                            should_judge=False, failed_only=False)
print(f"  4a difficulty-only: scores={eb1.scores} feas={[o.get('feasible') for o in eb1.objective_scores]}", flush=True)

# 4b: eval_existing_with_validity (capture_traces=True, should_judge=True, failed_only=True)
eb2 = a.aggregate_artifacts(batch, art, capture_traces=True,
                            should_judge=True, failed_only=True)
ntraj = len(eb2.trajectories or [])
print(f"  4b existing-with-validity: scores={eb2.scores} n_trajectories={ntraj}", flush=True)
assert len(eb2.scores) == len(batch)
assert ntraj == len(batch), "capture_traces=True should yield one trajectory per task"
print("ACCEPT-PATH SMOKE PASSED: both aggregate calls work; trajectories captured", flush=True)
