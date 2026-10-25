# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Lost in Conversation is a benchmark for evaluating LLMs on **multi-turn, underspecified** conversation. It "shards" single-turn instructions into ordered atomic pieces revealed one turn at a time, so single-turn and multi-turn performance can be compared on identical underlying tasks. Companion to the paper "LLMs Get Lost in Multi-Turn Conversation" (arXiv:2505.06120).

## Setup

```sh
pip install -r requirements.txt      # numpy, tqdm, tiktoken, pandas, streamlit, GitPython, nltk, sqlparse, pymongo, sacrebleu
```

API access requires one of these environment variables set before running:
- `OPENAI_API_KEY`, **or**
- `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`

`model_openai.py` auto-selects Azure when both Azure vars are present, otherwise OpenAI. There is no `.env` auto-loader in the run path — export the vars in your shell (a `load_env_vars()` helper exists in `utils.py` but is not called by the simulators).

## Common commands

```sh
# Run simulations (all defaults: gpt-4o-mini, math task, 1 run each of full/concat/sharded)
python run_simulations.py
python run_simulations.py --help                       # see all knobs

# Typical experiment invocation
python run_simulations.py \
  --models gpt-4o-mini gpt-4o \
  --tasks math code database actions data2text summary translation \
  --N_sharded_runs 5 --N_full_runs 5 --N_concat_runs 5 \
  --N_workers 8 --log_folder logs

# One-off single conversation (each simulator is independently runnable with its own __main__)
python simulator_sharded.py --task math --assistant_model gpt-4o-mini --verbose
python simulator_full.py    --task math --run_concat --verbose
python simulator_snowball.py --task database
python simulator_recap.py   --task database --models gpt-4o-mini

# Inspect simulated conversations in a browser
streamlit run app_conv_viewer.py
```

There is no test suite, linter config, or Makefile in this repo.

## Architecture

### Three conversation types (the core experimental axis)

`run_simulations.py` dispatches on `conv_type` to one of three simulators. All three share the same task, system-prompt, and evaluator; they differ in how the instruction reaches the assistant:

- **`full`** (`simulator_full.py`) — single-turn baseline. The fully-specified prompt (`task.populate_fully_specific_prompt`) is sent once; response evaluated once.
- **`concat`** (`simulator_full.py`, `run_concat=True`) — all shards concatenated into one user message. Controls for total information while staying single-turn.
- **`sharded`** (`simulator_sharded.py`) — the multi-turn condition. Shards are revealed one per turn by a `UserAgent`; the assistant responds each turn; a `SystemAgent` classifies each response and evaluates answer attempts.

Two intervention simulators reuse the sharded loop and are run from their own `__main__`:
- **`snowball`** (`simulator_snowball.py`) — each user turn reiterates all hints revealed so far.
- **`recap`** (`simulator_recap.py`) — takes a completed sharded log, appends a recapitulation message with the full/concat prompt, and re-evaluates. Reads existing `logs/<task>/sharded/` to find source conversations.

### Agent roles (all LLM-driven, via `model_openai.generate`/`generate_json`)

- **Assistant** — the system under test (`assistant_model`). Plain `generate()` call on the extracted conversation.
- **UserAgent** (`user_agent.py`) — `user_model`. For most tasks it reads `prompts/user_agent.txt`, decides what to say next, and returns which `shard_id` to reveal. For `translation`, `summary`, `data2text` it instead delegates to `task.populate_sharded_prompt(sample, turn_index)` — these tasks have fixed, deterministic shard disclosure (no LLM user).
- **SystemAgent** (`system_agent.py`) — `system_model`. Two jobs: (1) `verify_system_response` classifies the last assistant turn as `answer_attempt` vs `clarification`/`discussion` using `prompts/system_turn_categorization.txt` (summary/totto/translation skip this and are always treated as answer attempts); (2) `extract_answer` pulls the answer out of the assistant turn using the task's `answer_extraction_strategy` (`full_response`, `prefix_suffix`, `gen`, or `task_specific`).

### The trace

Every simulator builds a `trace`: a list of `{"role", "content", "timestamp", ...}` dicts. Roles include `system`, `user`, `assistant`, and `log`. `log` entries carry typed events: `shard_revealed`/`hint_revealed`, `system-verification`, `answer-evaluation`, `conversation-completed`. `utils.extract_conversation()` projects the trace down to just `system/user/assistant` (the real LLM-visible conversation) — the `log` entries are bookkeeping never sent to the model. When working on a simulator, preserve this separation: never feed `log` entries to `generate()`.

### Tasks (`tasks/` + `task_base.py`)

`task_base.py:Task` defines the abstract interface every task implements: `get_task_name`, `get_dataset_file`, `get_samples`, `get_answer_description`, `generate_system_prompt`, `evaluator_function`, `populate_fully_specific_prompt`, `populate_concat_prompt`, `process_original_sample`. `tasks/tasks.py:get_task(task_name)` is the registry/factory — **add new tasks here**. Each task lives in `tasks/<name>/task_<name>.py` with a per-task `eval_*.py` for scoring.

The seven tasks: `code` (HumanEval/LCB), `database` (Spider), `actions` (BFCL function-calling), `math` (GSM8K), `data2text` (ToTTo), `summary` (SummHay), `translation` (WMT). `evaluator_function` must return a `dict` with `score` and/or `is_correct` (score `1.0` ⇒ correct).

### Prompts (`prompts/`)

Plain `.txt` files loaded at runtime. Placeholder substitution uses `[[KEY]]` syntax — either via `model_openai.format_messages` (for the last user message) or direct `.replace()` in task/simulator code. Lazification prompts (`*_lazification_*.txt`) were used to build the sharded dataset (see paper appendix), not at simulation time.

### Data & logging

- Input: `data/sharded_instructions_600.json` (the 600 sharded instructions; default `--dataset_file`). Per-task raw datasets live in `data/sharded_<task>.json`.
- Output: `logs/<task>/<conv_type>/<conv_type>_<task>_<model>.jsonl`, one JSON record per conversation. Records include `conv_id`, `git_version`, the full `trace`, `is_correct`, `score`, and `dataset_fn`. Large log files are auto-split into `__N.jsonl` chunks by `utils_log.split_large_file` (30 MB).
- **Resumability**: `run_simulations.py` calls `utils_log.get_run_counts` to count existing per-`task_id` logs and only queues the missing runs. Re-running the same command resumes instead of duplicating. `--log_folder` / dataset_fn / temperature suffixes are part of the dedup key.

## Gotchas

- **Shard text field is `shard`, not `shard_text`.** The README's schema snippet says `shard_text`, but the code reads `sample["shards"][i]["shard"]` everywhere. `shard_id` is the integer key.
- **`task_base.Task.__init__` is dead code.** It calls `self._get_task_name()` (underscore-prefixed) but the abstract methods are named `get_task_name` (no underscore), and no concrete task calls `super().__init__()`. Each task subclass sets its own fields in `__init__`. Don't rely on the base initializer; don't "fix" the mismatch without updating all subclasses.
- **`code` task runs only on Unix** (the eval executes generated Python). **`database` task requires the Spider DBs** — download them and place under `data/spider/databases/` (see `data/spider/readme.txt`) before running.
- **Snowball task-name aliases differ**: `simulator_snowball.py` asserts membership in `["database", "wmt", "summary", "math", "apis", "data2text", "python"]` — older internal names (`wmt`→translation, `apis`→actions, `python`→code) that don't match the `tasks/` registry names. The snowball simulator is a legacy path; prefer `simulator_sharded.py` for new work.
- **Cost tracking is hardcoded** in `model_openai.OpenAI_Model.cost_calculator` — adding a new model family means adding its pricing there or it raises "pricing unknown".
- **Reasoning models** (`o1`/`o3`/`deepseek-r1`) get higher `max_tokens` (10000 vs 1000) and have their system message folded into the first user message (o1 doesn't accept system role).
- The `.msr/project.md` file is internal Microsoft Research metadata, not project instructions.
