"""Smoke-test one task for a domain through VitaPersonaAdapter (empty persona).
Verifies task loading + environment + run_task + evaluator all work end-to-end
before launching a full 15-iter PGEPA run. Usage: python _smoke_cross.py <domain> <task_id>
"""
import sys, json
sys.path.insert(0, "scripts")
from vita_persona_adapter import VitaPersonaAdapter, VitaTask

domain = sys.argv[1]
task_id = sys.argv[2]

ad = VitaPersonaAdapter(
    persona_id="smoke",
    agent_llm="gpt-5.4", agent_llm_args={"reasoning_effort": "high"},
    user_llm="gpt-5.4", user_llm_args={"reasoning_effort": "high"},
    max_steps=100, max_errors=3, max_duration=600,
    num_trials=1, evaluator_llm="gpt-4.1", evaluator_llm_args={},
    enable_think=False, language="chinese", domain=domain,
    max_concurrency=2, seed_base=300,
    online_judge=False,  # smoke: skip judge to keep it cheap
)
print(f"smoke: domain={domain} run_domain={ad._run_domain} task={task_id}")
batch = [VitaTask(task_id=task_id, domain=domain)]
eb = ad.evaluate(batch, {"persona_text": ""}, capture_traces=True)
print("=== outputs ===")
for o in eb.outputs:
    print(json.dumps({"task_id": o.task_id, "reward": o.reward,
                      "outcome": str(o.outcome), "domain": o.domain}, ensure_ascii=False))
print("=== scores ===", list(eb.scores) if hasattr(eb, "scores") else "n/a")
