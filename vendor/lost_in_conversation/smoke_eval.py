"""Smoke test: evaluate one seed persona on 1 task/domain via LICPersonaAdapter."""
import sys, json, os
# Load ../.env manually (zsh `source` chokes on <placeholder> values like <your-key>).
for _ln in open('../.env'):
    _ln = _ln.strip()
    if _ln and '=' in _ln and not _ln.startswith('#'):
        _k, _v = _ln.split('=', 1)
        if _v and not _v.startswith('<'):  # skip placeholder values
            os.environ.setdefault(_k, _v)
sys.path.insert(0, '.')
sys.path.insert(0, '../scripts/persona_opt')
from lic_persona_adapter import LICPersonaAdapter, LICTask

s = json.load(open('data/lic_split.json'))
batch = [LICTask(task_id=s['train'][d][0], domain=d)
         for d in ['math', 'code', 'database', 'actions']]

a = LICPersonaAdapter(persona_id='smoke1', agent_model='gpt-5.4',
                     reasoning_effort='high',
                     domains=['math', 'code', 'database', 'actions'],
                     n_workers=4, lic_dir='.',
                     always_judge_minibatch=False)  # cheap smoke: rule gate only
cand = {'persona_text': open('seed_lazy.txt').read()}
print(f"smoke: 4 tasks, model={a._agent_model} effort={a._reasoning_effort}", flush=True)
eb = a.evaluate(batch, cand, capture_traces=True)
print("\n=== SMOKE RESULT ===", flush=True)
print(f"n scores={len(eb.scores)} n obj={len(eb.objective_scores)}", flush=True)
for t, o, ob in zip(batch, eb.outputs, eb.objective_scores):
    print(f"  {t.domain:9} {t.task_id:40} outcome={o.outcome.value:24} "
          f"is_correct={o.is_correct} diff={ob['difficulty']} valid={ob['validity']}", flush=True)
infeas = sum(1 for ob in eb.objective_scores if ob.get("feasible", 1) == 0)
print(f"infeasible-batch={infeas>0}  mean_diff={sum(ob['difficulty'] for ob in eb.objective_scores)/len(eb.objective_scores):.3f}", flush=True)
