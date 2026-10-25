import json

def render(d, title):
    out = [f"## {title}", "", "**Problem:** " + d["task"]["problem_description"], ""]
    out.append("| turn | role | content |")
    out.append("|---|---|---|")
    for i, t in enumerate(d["dialogue_history"]):
        c = t["content"].replace("|", "\\|").replace("\n", "<br>")
        out.append(f"| {i} | {t['role']} | {c} |")
    ans = str(d.get("answer", "")).strip()
    if "```" in ans:
        code_block = ans
    else:
        code_block = "```python\n" + ans + "\n```"
    out += ["", "**Final answer (agent):**", code_block,
            "", f"**reward:** {d.get('reward')}", ""]
    return "\n".join(out)

mine = [json.loads(l) for l in open("out_openai.jsonl")]
off = [json.loads(l) for l in open("data/colbench_code_offline_15k_llama8b.jsonl")]

parts = []
parts.append(render(mine[0], "Conversation 1 — gpt-4o-mini, 4 turns (my run, task 0, reward 0.0)"))
parts.append(render(mine[5], "Conversation 2 — gpt-4o-mini, 10 turns hit max_steps (my run, task 5, reward 0.7)"))
# pick an offline conversation with a decent length and nonzero reward
for d in off:
    if len(d["dialogue_history"]) >= 8 and d.get("reward", 0) == 1.0:
        parts.append(render(d, "Conversation 3 — Llama-3.1-8B offline, 8+ turns (offline 15k, reward 1.0)"))
        break

open("conversations.md", "w").write("\n---\n\n".join(parts))
print("wrote conversations.md")
