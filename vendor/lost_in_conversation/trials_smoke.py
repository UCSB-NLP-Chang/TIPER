"""Quick smoke: num_trials=2 -> run_demo produces 2 runs/task, adapter aggregates."""
import sys, json, os
for _ln in open('../.env'):
    _ln = _ln.strip()
    if _ln and '=' in _ln and not _ln.startswith('#'):
        _k, _v = _ln.split('=', 1)
        if _v and not _v.startswith('<'):
            os.environ.setdefault(_k, _v)
sys.path.insert(0, '.'); sys.path.insert(0, '../scripts/persona_opt')
from lic_persona_adapter import LICPersonaAdapter, LICTask

s = json.load(open('data/lic_split.json'))
batch = [LICTask(s['train']['math'][0], 'math')]
a = LICPersonaAdapter(persona_id='trials_smoke', agent_model='gpt-5.4',
                     reasoning_effort='high', domains=['math'], n_workers=2,
                     lic_dir='.', num_trials=2)
cand = {'persona_text': open('seed_lazy.txt').read()}
print(f"trials smoke: 1 math task x2 trials, task={batch[0].task_id}", flush=True)
art = a.run_simulations(batch, cand)
print(f"  runs_by_task: {{k: len(v) for k,v in art.runs_by_task.items()}} = "
      f"{ {k: len(v) for k,v in art.runs_by_task.items()} }", flush=True)
eb = a.aggregate_artifacts(batch, art, capture_traces=True,
                           should_judge=False, failed_only=False)
print(f"  scores={eb.scores} n_traj={len(eb.trajectories or [])} "
      f"obj={eb.objective_scores}", flush=True)
assert all(len(v) == 2 for v in art.runs_by_task.values()), "expected 2 trials per task"
assert len(eb.scores) == 1
print("TRIALS SMOKE PASSED: 2 runs/task, aggregated to 1 score", flush=True)
