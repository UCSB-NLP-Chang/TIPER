# show_matrix.sh — API Settings for All 18 Cells

Records the **actual** agent / user `llm` and `llm_args` for every cell that
`scripts/show_matrix.sh` aggregates (3 agents × 3 users × 2 domains = 18 cells),
**preserving each value's original format as saved**.

## Source of truth (grounded, not guessed)

Each cell's `llm_args` was extracted from files, not inferred:

- **Measured (15 cells)**: `<run_dir>/results.json` → `info.agent_info` / `info.user_info`.
  - This is what the tau2 runner persists for reproducibility. `llm_args` is the **raw dict
    passed to the runner** (before the LiteLLM mapping layer).
- **Intended (3 cells — results.json missing)**: variable definitions in
  `scripts/run_cccon_agent_matrix_experiment.sh`.
  - Missing: `banking_gpt54_cc_con`, `banking_sonnet45_gpt52user`, `banking_sonnet45_sonnet46user`.
  - These are "what the script would pass", NOT confirmed by a results.json.

Common settings (15 measured cells): `num_trials=3`, `max_steps=200`, `max_errors=10`,
`git_commit=24a1bdf`.

---

## The three API routes

| Route | How to tell | Shape of llm_args |
|-------|-------------|-------------------|
| **OpenAI API** | `llm` is `gpt-5.2` / `gpt-5.4` | uses `reasoning_effort` directly (OpenAI native) |
| **Bedrock** | `llm` has `bedrock/...` prefix | carries `aws_region_name`; Claude uses `thinking` or `reasoning_effort` |
| **CC → Bedrock** | `implementation=cc_user_con`, `llm=claude-sonnet-4-6` (no prefix) | `use_bedrock:true` + `bedrock_profile`; thinking/effort are NOT passed to the CLI, fixed inside CC |

---

## All 18 cells (original format preserved)

### Agent-side llm_args

All cells use `implementation=llm_agent`. The same agent label can carry **different
llm_args formatting** across cells (see notes), so they are listed per cell.

| agent | user | domain | agent.llm | agent.llm_args (verbatim) | source |
|-------|------|--------|-----------|---------------------------|--------|
| gpt-5.2 | gpt52user | telecom | `gpt-5.2` | `{"temperature":1.0,"reasoning_effort":"high"}` | results.json |
| gpt-5.2 | gpt52user | banking | `gpt-5.2` | `{"reasoning_effort":"high","temperature":1.0}` | results.json |
| gpt-5.2 | sonnet46user | telecom | `gpt-5.2` | `{"reasoning_effort":"high"}` | results.json |
| gpt-5.2 | sonnet46user | banking | `gpt-5.2` | `{"reasoning_effort":"high","temperature":1.0}` | results.json |
| gpt-5.2 | cc_con | telecom | `gpt-5.2` | `{"reasoning_effort":"high","temperature":1.0}` | results.json |
| gpt-5.2 | cc_con | banking | `gpt-5.2` | `{"reasoning_effort":"high","temperature":1.0}` | results.json |
| gpt-5.4 | gpt52user | telecom | `gpt-5.4` | `{"reasoning_effort":"high"}` | results.json |
| gpt-5.4 | gpt52user | banking | `gpt-5.4` | `{"reasoning_effort":"high","temperature":1.0}` | results.json |
| gpt-5.4 | sonnet46user | telecom | `gpt-5.4` | `{"reasoning_effort":"high"}` | results.json |
| gpt-5.4 | sonnet46user | banking | `gpt-5.4` | `{"reasoning_effort":"high","temperature":1.0}` | results.json |
| gpt-5.4 | cc_con | telecom | `gpt-5.4` | `{"reasoning_effort":"high","temperature":1.0}` | results.json |
| gpt-5.4 | cc_con | banking | `gpt-5.4` | `{"reasoning_effort":"high","temperature":1.0}` | ⚠ script only (results.json missing) |
| sonnet-4.5 | gpt52user | telecom | `bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0` | `{"temperature":1.0,"aws_region_name":"us-west-2","reasoning_effort":"high"}` | results.json |
| sonnet-4.5 | gpt52user | banking | `bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0` | `{"thinking":{"type":"enabled","budget_tokens":4096},"temperature":1.0,"allowed_openai_params":["thinking"],"aws_region_name":"us-west-2"}` | ⚠ script only (results.json missing) |
| sonnet-4.5 | sonnet46user | telecom | `bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0` | `{"temperature":1.0,"aws_region_name":"us-west-2","reasoning_effort":"high"}` | results.json |
| sonnet-4.5 | sonnet46user | banking | `bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0` | `{"thinking":{"type":"enabled","budget_tokens":4096},"temperature":1.0,"allowed_openai_params":["thinking"],"aws_region_name":"us-west-2"}` | ⚠ script only (results.json missing) |
| sonnet-4.5 | cc_con | telecom | `bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0` | `{"thinking":{"type":"enabled","budget_tokens":4096},"temperature":1.0,"allowed_openai_params":["thinking"],"aws_region_name":"us-west-2"}` | results.json |
| sonnet-4.5 | cc_con | banking | `bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0` | `{"thinking":{"type":"enabled","budget_tokens":4096},"temperature":1.0,"allowed_openai_params":["thinking"],"aws_region_name":"us-west-2"}` | results.json |

### User-side llm_args (only 3 distinct)

The user config is determined by the user label. Original format:

| user_label | implementation | user.llm | user.llm_args (verbatim) | API route |
|-----------|----------------|----------|--------------------------|-----------|
| **gpt52user** | `user_simulator` | `gpt-5.2` | `{"temperature":1.0}` | OpenAI API |
| **sonnet46user** | `user_simulator` | `bedrock/us.anthropic.claude-sonnet-4-6` | `{"temperature":1.0,"aws_region_name":"us-west-2"}` | Bedrock |
| **cc_con** | `cc_user_con` | `claude-sonnet-4-6` | `{"use_bedrock":true,"bedrock_profile":"us-west-2-staging_BedrockBatchInference"}` | CC → Bedrock |

> User-side values are identical across all 15 measured cells (only the 3 patterns above).
> The 3 missing cells use the same values per script (`GPT52_USER_ARGS` / `SONNET46_USER_ARGS`
> / `CCCON_USER_ARGS`).

---

## Important notes (format variance & grounding)

1. **gpt-5.x agent `temperature` presence varies by cell.** Some are `{"reasoning_effort":"high"}`
   only, others `{"reasoning_effort":"high","temperature":1.0}`. These are the raw values from
   results.json (different launch scripts ran them separately). `reasoning_effort` is an
   **OpenAI-native arg** — no conversion.

2. **sonnet-4.5 agent has two formats that may NOT be equivalent in practice.**
   - `{"...","reasoning_effort":"high"}` (telecom gpt52user / sonnet46user cells)
   - `{"thinking":{"type":"enabled","budget_tokens":4096},...,"allowed_openai_params":["thinking"]}` (cc_con cells & banking cells)

   They are *intended* to be equivalent (`reasoning_effort:"high"` → `thinking` budget 4096).
   tau2's OWN manual conversion (`llm_utils.py:518`) only fires for `claude-` ids, **NOT for the
   `bedrock/` prefix** — which is why the script comment
   (`run_cccon_agent_matrix_experiment.sh:15-17`) says sonnet-4.5 must pass explicit `thinking`.
   **BUT — verified empirically — LiteLLM 1.86.2 itself maps `reasoning_effort:"high"` →
   `thinking={"type":"enabled","budget_tokens":4096}` for `bedrock/` Claude** even when tau2
   does not (probed via `litellm.get_optional_params`, see Hidden Defaults §V2 below).
   → **So the telecom `reasoning_effort`-form cells DID get thinking budget 4096** under the
   installed litellm; they ARE equivalent to the explicit-thinking cells.
   ⚠ Caveat: this relies on litellm ≥ ~1.82.7 (older versions silently drop it — the reason
   tau2 added its manual map). If those runs predate that version, they could have had no
   thinking. results.json records `git_commit` but not the litellm version at run time, so this
   can't be confirmed from the saved data alone — only that the current reproducible behavior is
   "equivalent".

3. **cc_con (cc_user_con) thinking is not controllable via llm_args.** The CC backbone uses a
   fixed internal config: `thinking:{"type":"adaptive"}` + `output_config.effort:"high"` +
   `max_tokens:32000` (measured in `CC_USER_BACKBONE_API.md`). The `use_bedrock`/`bedrock_profile`
   args only select the route. Note: the `claude` CLI itself does have
   `--effort {low,medium,high,xhigh,max}`, but the tau2 wrapper does not wire it up
   (`claude_code_user_simulator.py:635-657` passes only `--model`).

4. **Saved llm_args is the pre-LiteLLM raw dict.** It is a different layer from the actual HTTP
   body sent to Bedrock / OpenAI (Bedrock converts it into the Converse API
   `additionalModelRequestFields`). See `API_LAYERS_EXPLAINED.md`.

5. **The 3 missing-results.json cells** (⚠) were not run or live elsewhere. Their values above are
   the intended settings from `scripts/run_cccon_agent_matrix_experiment.sh`, unconfirmed by a
   results.json.

---

## Hidden / implicit defaults (parameters you did NOT explicitly set)

Below: things that have a default value despite not appearing in `llm_args`. Split into
**✅ VERIFIED** (checked against this repo / probed empirically just now) and
**📄 EXTERNAL** (documented provider/SDK behavior, NOT verified against this repo — audit before
trusting). Environment: `litellm 1.86.2`, `litellm.drop_params=True`, tau2 sets no `max_tokens`
anywhere (`config.py` / `llm_utils.py` grep clean; only `num_retries` defaults).

### ✅ VERIFIED in this repo

| # | Param | Finding | How verified |
|---|-------|---------|--------------|
| V1 | `max_tokens` (Bedrock Claude) | tau2 sets none → **LiteLLM injects `maxTokens=8192`** (NOT 4096). So `thinking.budget_tokens=4096 < 8192` — **no collision**, which is why the sonnet-4.5 thinking cells run fine. | `litellm.get_optional_params(model="us.anthropic.claude-sonnet-4-5...", custom_llm_provider="bedrock", thinking={budget 4096})` → `maxTokens=8192` |
| V2 | `reasoning_effort → thinking` (Bedrock) | LiteLLM 1.86.2 **natively maps** `reasoning_effort:"high"` → `thinking={"type":"enabled","budget_tokens":4096}` for `bedrock/` Claude, independent of tau2's manual map. → telecom (reasoning_effort form) == banking/cc_con (explicit form). | same probe with `reasoning_effort="high"`, no thinking → output had `thinking budget 4096` + `maxTokens=8192` |
| V3 | `max_tokens` (OpenAI gpt-5.x) | LiteLLM injects **nothing** → model default cap (~128k for gpt-5.x). Only `reasoning_effort` + `temperature` passed; `extra_body={}`. | probe `model="gpt-5.2", custom_llm_provider="openai", reasoning_effort="high"` |
| V4 | `num_retries` | Defaults to `DEFAULT_MAX_RETRIES = 3` when not in `llm_args`. | `llm_utils.py:498-499` |
| V5 | `temperature=1.0` + thinking (Bedrock) | This is the **correct/required** combo, not a bug: Anthropic extended thinking requires temperature=1 (other values error). LiteLLM keeps `temperature:1.0` alongside thinking — no conflict. | earlier live call: thinking enabled + temp 1.0 on bedrock sonnet-4-6 returned OK |

> Correction vs the pasted notes: the worry that "LiteLLM injects `max_tokens=4096` and collides
> with budget 4096" is **not what happens** — it injects **8192**. And "temperature with thinking
> is forbidden" is too strong — temperature **=1.0** is required/fine; only non-1 values error.

### 📄 EXTERNAL (documented behavior — NOT verified against this repo; audit if it matters)

**OpenAI route (gpt-5.x):**
- `verbosity` — defaults to `"medium"` (GPT-5 family). Not set by tau2 → all gpt-5.x cells run at
  medium verbosity, which affects output-token volume in cross-cell comparisons.
- `reasoning_effort` tier — `high` is set, but a higher `xhigh` tier exists for gpt-5.2; that
  headroom is unused.
- No `prompt_cache_key` set → prefix caching still auto-applies but isn't trackable per run.

**Bedrock route (sonnet-4.5):**
- `top_p` / `top_k` — not in `llm_args`; Converse uses model defaults. (When thinking is on,
  Anthropic ignores/forbids top_p/top_k anyway.)
- thinking `display` — Claude 4 defaults to `summarized`; thinking content you see is a summary,
  not raw thinking tokens.
- AWS SDK (boto3) read timeout — default ~60s; long Claude-4 turns can exceed it unless the
  underlying client raises it. Not checked whether litellm's boto3 client overrides this.

**CC → Bedrock route (cc_user_con):**
- Interleaved thinking — CC backbone can think between tool calls; direct-Bedrock sonnet-4.5 does
  not interleave by default → different behavior even at the same nominal "effort".
- `MAX_THINKING_TOKENS` / `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING` env vars — if set on the
  experiment host, they change the Sonnet-4.6 backbone's thinking mode (adaptive → fixed). Not
  audited on the run host.
- Heavy CC system prompt — CC injects its own large system prompt (tool docs, permissions,
  CLAUDE.md, etc.). This makes cc_con's **input token distribution fundamentally different** from
  the `sonnet46user` direct-Bedrock user — the real reason the two aren't directly comparable.

---

## Suggested audit follow-ups

1. Pull the **actual request body** from a saved run (`data/api_first_requests/` or a Bedrock
   invocation log) to confirm `inferenceConfig.maxTokens=8192` reaches the wire (the only fully
   reliable check of V1).
2. `grep` the run host for `CLAUDE_CODE_*` / `MAX_THINKING_TOKENS` / `OPENAI_*` env vars to rule
   out leftover overrides.
3. (Optional) Pin `verbosity:"medium"` explicitly on gpt-5.x cells for documentation clarity.

---

## Referenced files

- `scripts/show_matrix.sh` — the 18 cell path definitions
- `scripts/run_cccon_agent_matrix_experiment.sh` — launch-time arg definitions (`GPT_AGENT_ARGS`,
  `SONNET45_AGENT_ARGS`, `GPT52_USER_ARGS`, `SONNET46_USER_ARGS`, `CCCON_USER_ARGS`)
- `<run_dir>/results.json` → `info.agent_info` / `info.user_info` — measured ground truth
- `src/tau2/utils/llm_utils.py:518-530` — reasoning_effort→thinking budget_map
- `src/tau2/user/claude_code_user_simulator.py:635-657` — CC CLI args (thinking not supported)
- `CC_USER_BACKBONE_API.md` — measured CC backbone fixed thinking/effort
