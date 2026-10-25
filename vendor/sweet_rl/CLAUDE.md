# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> This repo is `facebookresearch/sweet_rl` — SWEET-RL and ColBench (Collaborative Agent Bench). It is independent of the `tau2` framework in the parent directory; do not mix the two.

## What this is

SWEET-RL trains multi-turn LLM agents on **Collaborative Reasoning** tasks, where an LLM agent converses with a simulated human user over multiple turns to solve a realistic task. ColBench has two task types:

- **Backend Programming (`code`)** — agent must elicit requirements from the human, then produce a Python function matching hidden test cases. Evaluated by executing the function against test cases (pass rate).
- **Frontend Design (`html`)** — agent iterates on HTML/Tailwind designs based on visual feedback from the human, who compares the agent's rendered screenshot against a ground-truth design. Evaluated by CLIP image cosine similarity.

The agent signals it's done by emitting `I WANT TO ANSWER:`; otherwise the loop ends at `max_steps` (default 10). For `html`, the agent's current design is wrapped in an `<html>` tag and rendered to a screenshot the human simulator can see.

## Setup

```bash
pip install -e .                      # this package (fire, selenium, openai, gradio)
git clone https://github.com/YifeiZhou02/collab_openrlhf
cd collab_openrlhf && pip install -e .   # custom openrlhf fork for multi-turn DPO + length normalization
```

Frontend Design additionally needs Firefox + GeckoDriver in `$PATH`:
```bash
geckodriver --version   # verify; v0.35.0 used by the authors
```

Data is downloaded from HuggingFace:
```bash
huggingface-cli download facebook/collaborative_agent_bench \
    backend_tasks/train.jsonl backend_tasks/test.jsonl colbench_code_offline_15k_llama8b.jsonl
```
Frontend (`html`) data is built from `HuggingFaceM4/WebSight` via the snippet in README.md using `sweet_rl.utils.webpage_utils.replace_urls`.

There is **no test suite, Makefile, or lint config in this repo**. `CONTRIBUTING.md` references `autoformat.sh`, `lint_changed.sh`, and `pytest -m unit`, but none of those exist here — ignore those instructions. Coding style is PEP-based; just keep imports consistent with surrounding files.

## Common commands

All entry points use `fire` (CLI args map to function params). Every script hardcodes `/fsx-ram/...` default paths from the authors' cluster — **always pass explicit paths** instead of relying on defaults.

### Evaluate a model on ColBench

The human simulator runs as a **vLLM OpenAI-compatible server** (started separately), pointed at by `--hostname`/`--port`. Code uses Llama-3.1-70B as the human; html uses Qwen2-VL-72B.

```bash
# 1. Start the human-simulator vLLM server (separate shell/GPUs)
python -m vllm.entrypoints.openai.api_server --model /path/to/llama3.1-70b-instruct \
    --port 8000 --tensor-parallel-size 8 --max-model-len 16384 --enforce-eager --trust-remote-code

# 2. Run the multi-turn simulation (agent runs locally via vLLM)
python scripts/simulate_interactions.py \
    --agent_model /path/to/Llama-3.1-8B-Instruct \
    --hostname localhost --port 8000 \
    --task_type code --num_tasks 1000 \
    --input_path backend_tasks/test.jsonl --output_path out.jsonl \
    --env_model /path/to/llama3.1-70b-instruct

# 3. Score
python scripts/evaluate_code.py out.jsonl          # prints success rate + test-pass %
python scripts/evaluate_html.py out_html.jsonl     # prints avg CLIP cosine similarity
```

Visualize trajectories:
```bash
python visualizers/visualize_dialogue_histories.py out.jsonl            # code (gradio)
python visualizers/visualize_design_dialogue_histories.py out_html.jsonl # html (gradio)
```

### SWEET-RL training pipeline (offline, multi-stage)

The SWEET-RL algorithm is a **multi-stage script pipeline**, not a single training command. It produces step-level preference pairs for DPO training via the `collab_openrlhf` fork (`deepspeed --module openrlhf.cli.train_dpo`). Stages, in order:

1. **Build trajectory-level preferences with training-time info** — `evaluate_code.py $DATA_PATH --k 3 --ground_truth_preference_path …` reruns offline trajectories `k` times, scores them, and writes preference pairs where each pair's prompt is prefixed with `"In light that the final answer is: {ground_truth}."` (the training-time information that gives SWEET-RL its name).
2. **Train the step-level reward model** — DPO train an 8B model on those ground-truth preferences (`--mean_log_prob`, `--response_template "<|start_header_id|>assistant<|end_header_id|>"`). Saved to `$REWARD_PATH`.
3. **Sample N completions per step** — `sample_best_of_n.py $DATA_PATH $SAMPLED_PATH --data_fraction 0.1` (uses vLLM `n=16`, also records reference logprobs).
4. **Rank completions with the reward model** — `rank_best_of_n.py --model_id $REWARD_PATH --input_path $SAMPLED_PATH --output_path $RANKED_PATH`. Ranks by reward-model logprob minus reference logprob.
5. **Build step-level preference pairs** — `generate_random_pairs_from_ranks.py $RANKED_PATH $RANDOM_PAIRS_PATH --no_prompt --num_pairs 4` (chosen from top half, rejected from bottom half).
6. **DPO-train the policy** on those pairs, then run `simulate_interactions.py` + `evaluate_code.py` to evaluate.

For `html`, the offline-data stage uses `simulate_interactions.py --task_type html --best_of_n 6 --train --to_continue`.

## Architecture

```
sweet_rl/
  environments/        — the world the agent acts in (the "human" is an LLM via OpenAI API)
    human_interaction_env.py        — HumanInteractionEnv: code task. OpenAI client = human simulator.
    human_design_interaction_env.py — HumanDesignInteractionEnv: html task. Subclass that renders
                                      agent HTML + ground-truth HTML to screenshots via Selenium/Firefox
                                      and sends both as images to the (vision) human simulator.
  models/vllm_agent.py — VLLMAgent: runs the policy locally with vLLM, batched get_action().
  utils/
    code_utils.py      — check_correctness(): execs agent's Python against test cases in a subprocess
                         with a 1s timeout; blocks os/sys/print/open/transformers/sudo/exit.
    webpage_utils.py   — get_driver() (headless Firefox), render_full_html() (screenshot),
                         extract_html_snippet(), replace_urls() (normalizes images to picsum id 48).
prompts/               — system prompts for agent + human simulator (code vs html). Selected by
                         --task_type when --{user,agent}_prompt_path == "auto".
scripts/               — the pipeline entry points (fire CLI).
visualizers/           — gradio UIs to browse recorded dialogue_history.
```

### Core loop (`simulate_interactions.py` → `batch_interact_environment`)

A `HumanInteractionEnv` instance is created per concurrent task, each holding its own OpenAI client pointed at the human-simulator vLLM server. `VLLMAgent` holds one local vLLM `LLM` and batches **all** environments' actions per step:

1. `env.reset(problem_description, ground_truth)` seeds the dialogue with the user's initial vague request (ground_truth is the hidden info the human simulator sees but the agent must extract).
2. Loop `max_steps+1` times: `agent.get_action(batch_obs)` → for each env `env.step(response, formatted_prompt)`.
3. Inside `step`: the agent's text is appended as `assistant`; if not done, `invoke_model()` calls the human-simulator server and appends its reply as `user` (truncated to `HUMAN_RESPONSE_CHARACTER_LIMIT`, 400 for code / 1000 for html). The `done` flag fires on `I WANT TO ANSWER:` or step limit; the final answer is parsed from after that marker.
4. Each recorded dialogue turn stores `{"role","content","input","output"}` — `input` is the formatted chat-template prompt, `output` the raw model text. These are what the SWEET-RL preference-pair scripts later slice into `chosen`/`rejected`.

### Data format

Tasks (input jsonl): `{"problem_description", "ground_truth", "test_cases"}` (code) or `{"problem_description", "ground_truth"}` where ground_truth is HTML (html).
Simulation output: `{"task": <original>, "dialogue_history": [...], "answer": <final agent output>}`. `evaluate_*` adds a `"reward"` field and rewrites the file in place.

## Gotchas

- **No `torch` import in `simulate_interactions.py`** but it calls `torch.cuda.device_count()` (line ~91). Importing `vllm` does not put `torch` in the module namespace — this will `NameError` unless torch is imported. Add `import torch` if you hit this.
- `evaluate_code.py` **rewrites `saved_path` in place** adding `reward` fields; if `reward` already exists on every record it skips re-evaluation. Pass a copy if you want to preserve raw output.
- `code_utils.check_correctness` uses `exec`/`eval` on LLM-generated code in a subprocess with a 1s SIGALRM timeout. It blocks obvious dangerous tokens (`import os`, `open(`, `print(`, `sudo`, `transformers`, `exit(`) but this is not a real sandbox — only run on untrusted model output in an isolated environment.
- `filter_websites.py` imports `render_html` from `webpage_utils`, but that module only defines `render_full_html` — the script is stale/broken as committed.
- `VLLMAgent` builds prompts with `tokenizer.apply_chat_template` using `system = agent_prompt` + the dialogue history; the prompts in `prompts/*_agent_*.txt` still contain a `{dialogue_history}` placeholder that is **not** used in this code path (leftover from an older format). The chat template handles history instead.
- All `--env_model`/`--agent_model` defaults and `auto` resolutions point at the authors' `/fsx-ram/shared/...` paths — always override with local paths.
- `html` simulation forces `batch_size=100` regardless of the CLI flag (more Firefox drivers than that is unstable); remember to `env.driver.quit()` (done at the end of `simulate_interactions.py`).
- CC-By-NC licensed — non-commercial use only.
