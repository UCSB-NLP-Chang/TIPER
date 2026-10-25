"""Test LIC-domain-aware mutation ctx: build a real mutation prompt from existing
data, call the gpt-5.5 proposer, inspect the persona it produces. Iterate on the
ctx text below until the produced personas hit LIC-effective attack axes (not
telecom customer-support axes).

Usage:
  source ./load_env.sh
  ../.venv/bin/python test_mutation_ctx.py --domain math
  ../.venv/bin/python test_mutation_ctx.py --domain math --use-old-ctx  # compare vs current telecom ctx
"""
import sys, json, os, argparse, glob

for _ln in open('../.env'):
    _ln = _ln.strip()
    if _ln and '=' in _ln and not _ln.startswith('#'):
        _k, _v = _ln.split('=', 1)
        if _v and not _v.startswith('<'):
            os.environ.setdefault(_k, _v)

sys.path.insert(0, '.'); sys.path.insert(0, '../scripts/persona_opt')
from lineage_loop import call_mutation_llm, _extract_json  # noqa: E402

# ---------------------------------------------------------------------------
# LIC domain SCORING: real evaluation mechanism + domain-specific attack axes.
# Replaces the telecom "multiple root causes / device state assertions" fallback.
# ---------------------------------------------------------------------------
SCORING_LIC = {
"math": """## HOW THE AGENT IS SCORED
The agent must produce the FINAL NUMERIC ANSWER to a math word problem, judged by
exact-match against the gold answer. The problem is revealed across multiple turns:
the user discloses quantities, constraints, and relationships one shard at a time.
A wrong final number scores 0 — there are no partial credits for a correct setup
with an arithmetic slip, and no credit for solving a different (easier) problem.

## WHAT MAKES THE AGENT FAIL HERE
The agent fails when it (a) locks onto an early quantity as the main unknown and
ignores a later-revealed constraint that changes the setup, (b) misses a buried
numeric value because it was slipped into a long casual sentence among other details,
(c) uses the wrong units / misreads a rate or ratio because the framing suggested a
different relationship, or (d) finalizes an answer before the last shard reveals a
qualifying condition (a discount, a second party, a time constraint). Effective
personas bury the decisive quantity in an inconspicuous position, reveal the
qualifying condition LAST, or frame the problem so the agent models the wrong
relationship.""",

"code": """## HOW THE AGENT IS SCORED
The agent must write code that passes the task's test cases — including hidden and
boundary cases, not just the worked example. Partial pass (3/4 cases) still scores 0.
The spec (function signature, examples, edge-case requirements, constraints) is
revealed one shard at a time across turns.

## WHAT MAKES THE AGENT FAIL HERE
The agent fails when it (a) anchors on the first worked example and writes code that
hard-codes that pattern, missing the general spec revealed later, (b) ignores a
boundary / edge-case condition revealed late because it already committed to a
happy-path implementation, (c) treats a provisional example as the full spec and
finalizes before the constraints shard arrives, or (d) misreads an off-by-one or
domain constraint because it was framed casually. Effective personas make the agent
commit to a concrete implementation off the first example, then reveal the
constraints/edge-cases that break it — exploiting the agent's reluctance to revise.""",

"database": """## HOW THE AGENT IS SCORED
The agent must write a SQL query whose EXECUTION RESULT (rows / columns / values)
matches the gold answer. Syntactically-valid-but-wrong-result scores 0. The query
requirements (tables, filters, joins, aggregations, conditions) are revealed one
shard at a time.

## WHAT MAKES THE AGENT FAIL HERE
The agent fails when it (a) anchors on the first suggested query direction (e.g.
"just show me the orders") and never revisits the schema/joins after later shards
add filter and grouping requirements, (b) misses a WHERE condition because it was
described as an offhand remark rather than a requirement, (c) picks the wrong join
relationship because the framing implied a different table linkage, or (d) finalizes
the query before the last shard reveals a grouping/aggregation constraint. Effective
personas anchor the agent on a plausible-but-incomplete query direction early, then
drip out the constraints that make that direction wrong.""",

"actions": """## HOW THE AGENT IS SCORED
The agent must call the correct function(s) with ALL correct argument values. Wrong
function, wrong argument, missing argument, or wrong argument value — any of these
scores 0 even if the intent was right. The user's intent and the required parameter
values are revealed one shard at a time.

## WHAT MAKES THE AGENT FAIL HERE
The agent fails when it (a) picks a function off the first fuzzy intent description
and never revisits after a later shard narrows the intent to a different API, (b)
fills a required argument with a wrong/guessed value because the exact value was
revealed late or buried, (c) refers to a record by relationship ("the other one",
"the recent one") letting the agent resolve to the wrong id, or (d) omits a
parameter because it was mentioned as an aside rather than a requirement. Effective
personas let the agent commit to a function + partial arguments early, then reveal
the exact values / extra parameters that make that call wrong.""",
}

# ---------------------------------------------------------------------------
# LIC attack-surface constraint: the invariants LOCK full revelation + exact
# values, so the ONLY lever is style/timing/order/framing. Telling the proposer
# this prevents wasted generations on withhold/blur attacks that _INVARIANTS
# already neutralize.
# ---------------------------------------------------------------------------
LIC_ATTACK_SURFACE = """## THE ATTACK SURFACE (read carefully — most personas waste this)
The user is FORCED by hard invariants to (1) eventually reveal EVERY shard and
(2) preserve every exact value verbatim (names, numbers, strings, dates). The user
CANNOT win by withholding a shard, refusing to answer, or blurring a number — those
are illegal and score zero. The ONLY levers the persona controls are:
  - TIMING/ORDER: reveal the decisive shard LAST, or reveal a misleading shard early
  - POSITION/BURYING: slip the key value into a long casual sentence among other
    details, or attach it as an offhand aside rather than a stated requirement
  - FRAMING/ANCHORING: describe the task so the agent locks onto the wrong scope,
    wrong unknown, or wrong function before the qualifying shards arrive
  - PREMATURE CLOSURE: behave as if the agent's early draft/answer/query is "basically
    right", nudging it to finalize before the last constraint shard is revealed
A persona that instructs withholding, refusing, or blurring exact values is INVALID
here (unlike a generic customer-support setting). Design around revelation style."""


def lic_mutation_prompt(persona_text, score, success_traces, failure_traces,
                        domain, history=None):
    sc = SCORING_LIC[domain]
    hist = json.dumps(history[-5:], indent=2) if history else "First evaluation."
    succ = json.dumps(success_traces[:3], indent=2) if success_traces else "None."
    fail = json.dumps(failure_traces[:2], indent=2) if failure_traces else "None."
    return f"""You are an expert adversarial persona designer for a multi-turn, underspecified-task
benchmark. An LLM assistant is given a task ({" / ".join([domain])}) whose full specification
has been sharded: the user reveals one piece of the spec per turn. Your persona controls
the SIMULATED USER who reveals those shards.

## TASK
Modify the persona below IN-PLACE to make it HARDER for the assistant to reach the correct
final answer, while keeping it VALID (realistic user, in-character, no cheating).

## CURRENT PERSONA
Assistant success rate: {score:.1%} (lower = harder = better)

```
{persona_text}
```

## SCORE HISTORY (recent generations)
{hist}

{sc}

{LIC_ATTACK_SURFACE}

## CONVERSATION TRACES WHERE THE ASSISTANT SUCCEEDED (persona failed to create difficulty)
Study these — find where the persona revealed info too cleanly, let the assistant finalize
too early, or failed to bury the decisive shard. What subtle behavior change would make the
assistant take the WRONG final answer instead?

{succ}

## CONVERSATION TRACES WHERE THE ASSISTANT FAILED (persona was effective — preserve the win)
{fail}

## MUTATION RULES
1. Rewrite ~30% of the persona — be BOLD, redesign the revelation style from scratch if needed.
2. Target a SPECIFIC assistant strategy visible in the success traces — find the exact turn it
   breaks through and design a fundamentally different counter.
3. Choose ONE behavioral axis that produces SYSTEMATIC misdirection — a consistent pattern that
   biases the assistant toward the WRONG answer every time. Attack one of these stages of the
   assistant's reasoning:
   (a) REVELATION TIMING/ORDER — reveal the decisive shard last; reveal a misleading shard
       early so the assistant commits to the wrong setup before the truth arrives.
   (b) POSITION/BURYING — slip the key value into a long casual sentence among other details,
       or as an offhand aside, so the assistant under-weights it.
   (c) PREMATURE CLOSURE / ANCHORING — behave as if the assistant's early draft/query is
       "basically right", nudging it to finalize before the last constraint shard arrives.
   (d) SCOPE FRAMING — describe the task so the assistant models the wrong unknown, wrong
       relationship, or wrong function before the qualifying shards arrive.
   Ineffective axes: random/erratic behavior, emotional volatility, unpredictable message
   length — these add VARIANCE but not BIAS; the assistant can still reach the answer by
   asking enough follow-ups.

## VALIDITY CONSTRAINTS (violating ANY = illegal persona, auto-reverted)
- The user MUST eventually reveal every shard and preserve every exact value verbatim.
  Withholding, refusing, or blurring numbers/names/dates is INVALID.
- Reveal at most one new shard per turn; never repeat an already-revealed shard.
- Never fabricate information not in the shards.
- Never mention "AI", "scenario", "simulation", or output JSON/system prompts.
- Stay a coherent, realistic user — not a cartoonish adversary.

## ADVERSARIAL REASONING
Reason in plain text first (saved as reference only; NOT the persona):
### Step 1 — Assistant assumption: from the success traces, the core assumption the assistant
relies on to break through. Quote the specific turn(s). Why is it a vulnerability?
### Step 2 — Success weak point: the 1-2 most revealing success traces; the single moment where
a realistic alternative behavior would MISLEAD the assistant into a WRONG final answer.
### Step 3 — Core hypothesis: "if the user does X, the assistant will mistakenly do Y, because Z."
### Step 4 — Write the persona implementing that hypothesis.

## OUTPUT FORMAT
Write Steps 1-4 reasoning as plain text FIRST. Then output the final persona as a SINGLE JSON
object wrapped in a ```json code fence, nothing after it:

```json
{{
  "mutation_direction": "<1-2 sentence concrete explanation of the NEW behavioral axis this
   persona exploits and HOW/WHY it misleads the assistant>",
  "persona_text": "<the full revised persona text>"
}}
```
"""


def load_traces(domain, tag='best', max_succ=3, max_fail=2):
    """Extract success/failure conversation traces from test eval demo_runs.json."""
    fs = glob.glob(f'data/persona_opt/lic_eval/test_{domain}_{tag}/eval_*/{domain}/demo_runs.json')
    if not fs:
        return [], []
    f = max(fs, key=os.path.getmtime)
    runs = json.load(open(f)).get('runs', [])
    succ, fail = [], []
    for r in runs:
        if 'error' in r or 'trace' not in r:
            continue
        conv = []
        for m in r['trace']:
            if m.get('role') in ('user', 'assistant'):
                conv.append({"role": m['role'], "content": str(m['content'])[:600]})
        item = {"task_id": r.get('task_id'), "is_correct": r.get('is_correct'),
                "n_shards": r.get('shards'), "conversation": conv}
        if r.get('is_correct') is True and len(succ) < max_succ:
            succ.append(item)
        elif r.get('is_correct') is False and len(fail) < max_fail:
            fail.append(item)
    return succ, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='math', choices=list(SCORING_LIC))
    ap.add_argument('--model', default='gpt-5.5')
    ap.add_argument('--effort', default='xhigh')
    ap.add_argument('--use-old-ctx', action='store_true',
                    help='build prompt with the CURRENT telecom ctx for comparison')
    ap.add_argument('--save', default=None, help='save produced persona to file')
    args = ap.parse_args()

    # current best persona + score from pgepa_state
    st = json.load(open(f'data/persona_opt/pgepa_runs/lic_{args.domain}/pgepa_state.json'))
    bi = st['best_idx']
    bc = next(c for c in st['candidates'] if c['idx'] == bi)
    persona_text = bc['persona_text']
    # agent success rate from test avg@3 if available, else 1 - val difficulty
    res_f = f'data/persona_opt/test_set_results_{args.domain}.json'
    if os.path.exists(res_f):
        rows = json.load(open(res_f))
        best_row = next((r for r in rows if r['persona'] == 'best'), None)
        score = best_row['avg@3'] if best_row else (1 - bc['agg'])
    else:
        score = 1 - bc['agg']

    succ, fail = load_traces(args.domain, 'best')
    print(f"domain={args.domain} best=idx{bi} gen{bc['gen']} score(agent success)={score:.3f}")
    print(f"traces: {len(succ)} success, {len(fail)} failure")

    if args.use_old_ctx:
        from lineage_loop import _build_mutation_prompt
        prompt = _build_mutation_prompt(persona_text, score, succ, [], 'lic_test',
                                        failure_traces=fail, scoring_domain=args.domain)
    else:
        prompt = lic_mutation_prompt(persona_text, score, succ, fail, args.domain)

    print(f"prompt chars: {len(prompt)}")
    if os.environ.get('DUMP_PROMPT'):
        open(f'/tmp/mut_prompt_{args.domain}.txt', 'w').write(prompt)
        print(f"prompt dumped -> /tmp/mut_prompt_{args.domain}.txt")

    print(f"\n{'='*70}\nCalling {args.model} ({args.effort})...\n{'='*70}")
    out = call_mutation_llm(prompt, args.model, reasoning_effort=args.effort)
    print("=== mutation_direction ===")
    print(out.get('mutation_direction', '?'))
    print("\n=== persona_text ===")
    pt = out.get('persona_text', '?')
    print(pt)
    if args.save:
        json.dump(out, open(args.save, 'w'), indent=2, ensure_ascii=False)
        print(f"\nsaved -> {args.save}")


if __name__ == '__main__':
    main()
