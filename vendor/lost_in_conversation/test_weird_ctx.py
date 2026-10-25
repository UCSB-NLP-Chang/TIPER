"""Test WEIRD mutation ctx: push the proposer toward unusual behavior patterns
(role-play / structural / rhythmic / cognitive-trick) instead of the safe
"cooperative + anchoring-bias" family pgepa keeps converging to.

Reuses SCORING_LIC + LIC_ATTACK_SURFACE from test_mutation_ctx; only the
mutation prompt changes. Loads v1 best persona + traces as the starting point
and calls gpt-5.5 to see what persona the weird ctx produces. Iterate on the
WEIRD ctx text below until produced personas are (a) genuinely unusual in
behavior pattern and (b) still VALID under _INVARIANTS.

Usage:
  source ./load_env.sh
  ../.venv/bin/python test_weird_ctx.py --domain math
  ../.venv/bin/python test_weird_ctx.py --domain math --seed neutral   # start from neutral
  DUMP_PROMPT=1 ../.venv/bin/python test_weird_ctx.py --domain code
"""
import sys, json, os, argparse, glob

for _ln in open('../.env'):
    _ln = _ln.strip()
    if _ln and '=' in _ln and not _ln.startswith('#'):
        _k, _v = _ln.split('=', 1)
        if _v and not _v.startswith('<'):
            os.environ.setdefault(_k, _v)

sys.path.insert(0, '.'); sys.path.insert(0, '../scripts/persona_opt')
from lineage_loop import call_mutation_llm  # noqa: E402
from test_mutation_ctx import SCORING_LIC, LIC_ATTACK_SURFACE, load_traces  # noqa: E402

# ---------------------------------------------------------------------------
# WEIRD behavior-pattern catalogue. All patterns stay INSIDE _INVARIANTS:
# every shard eventually revealed, every exact value preserved verbatim, one
# new shard per turn, no fabrication. The weirdness lives purely in HOW the
# user talks — persona / structure / rhythm / cognitive tone — never in WHAT
# info is conveyed. "Pretend to misremember then correct" is FORBIDDEN here
# because an intermediate wrong value violates [No Fabrication] + [Keep Exact
# Values]; the cognitive-trick patterns below use hesitation/uncertainty TONE
# with the correct value, never a wrong one.
# ---------------------------------------------------------------------------
WEIRD_CATALOGUE = """## BE BOLD — EXPLORE UNUSUAL BEHAVIOR PATTERNS
The safe "cooperative + one anchoring bias" family (tunnel-vision, example-
anchoring, first-option-anchoring, premature-closure, afterthought-burying)
has ALREADY been explored and converges to similar personas every time. Do
NOT produce another variant of it. Design a persona with a DISTINCT, UNUSUAL
BEHAVIOR PATTERN — a consistent, recognizable way of talking that a real
user might have and that systematically biases the assistant. Pick from (or
combine) these four families; the examples are seeds, not limits:

(1) ROLE / CHARACTER — give the user a vivid personality whose HABITS drive
    the misdirection, not an explicit "attack":
    - a slightly forgetful older person who muses aloud before each reveal
      ("now let me think… the year was… 2018"), making every fact sound
      provisional and underweighted;
    - an over-polite person who leads every turn with apologies/thanks and
      buries the key number after the pleasantries;
    - a chatty rambler who wraps each shard in a paragraph of irrelevant
      context, the decisive value buried mid-sentence;
    - a second-language user with broken grammar but EXACT values ("I want
      calculate winning percentage for Lakers, season 2018 please").

(2) STRUCTURAL — break the normal declarative-shard sentence form:
    - reveal each turn as a numbered/bulleted list where the key constraint
      is a sub-item under an innocuous main item;
    - lead with the conclusion ("I want the winning percentage") then drip
      the qualifying conditions over later turns as afterthoughts;
    - phrase shards as questions or "by the way, also…" side-remarks rather
      than stated requirements.

(3) RHYTHMIC — vary information density non-standardly across turns:
    - turn 1 is a long rambling paragraph with the first shard buried in the
      middle; later turns turn terse (one short fragment each);
    - some turns deliver just a single bare value/word, making the assistant
      unsure whether it's a new shard or a correction;
    - escalating disclosure: start almost dismissive, each later turn adds
      weight to a previously-dismissed detail.

(4) COGNITIVE TONE — manufacture false certainty/uncertainty signals WITHOUT
    ever giving a wrong value (a wrong value is INVALID):
    - hedge the precise value with uncertain tone ("I think it was… 48
      dollars?") so the assistant under-weights a correct, decisive number;
    - talk to yourself ("so that's 2018 for the Lakers… and the Bulls too")
      making the assistant unsure what's a confirmed fact vs musing;
    - treat the assistant's early draft as confirmed ("perfect, that's it")
      then reveal a qualifying shard that breaks it — the warmth makes the
      assistant reluctant to revise.

Combine families if it produces a coherent, recognizable character. The
persona must read as a REAL, if eccentric, user — not a cartoon adversary."""

# Directions already explored by v1/v2 pgepa — tell the proposer to avoid.
EXPLORED = """## ALREADY-EXPLORED DIRECTIONS (do NOT repeat or lightly remix these)
- tunnel-vision / single-issue focus
- example-anchored under-specification
- first-option / first-plausible-answer anchoring
- premature-closure / warm validation of early drafts
- afterthought-burial of qualifying conditions
- "cooperative person copying from notes" framing
If your mutation_direction reads like one of the above, start over."""


def weird_mutation_prompt(persona_text, score, success_traces, failure_traces,
                          domain, history=None):
    sc = SCORING_LIC[domain]
    hist = json.dumps(history[-5:], indent=2) if history else "First evaluation."
    succ = json.dumps(success_traces[:3], indent=2) if success_traces else "None."
    fail = json.dumps(failure_traces[:2], indent=2) if failure_traces else "None."
    return f"""You are an expert adversarial persona designer for a multi-turn, underspecified-task
benchmark. An LLM assistant is given a task ({domain}) whose full specification has been sharded:
the user reveals one piece of the spec per turn. Your persona controls the SIMULATED USER who
reveals those shards.

## TASK
REWRITE the persona from scratch into a WEIRD but VALID behavior pattern. Replace ~70%+ of the
text — do not just tweak the existing persona. Pick a distinctive unusual behavior pattern (see
catalogue) and commit to it fully. The persona should read as a real, if eccentric, user whose
consistent TALKING HABITS systematically bias the assistant toward the wrong final answer.

## CURRENT PERSONA (to replace, not refine)
Assistant success rate: {score:.1%} (lower = harder = better)

```
{persona_text}
```

## SCORE HISTORY (recent generations)
{hist}

{sc}

{LIC_ATTACK_SURFACE}

{WEIRD_CATALOGUE}

{EXPLORED}

## CONVERSATION TRACES WHERE THE ASSISTANT SUCCEEDED (persona failed to create difficulty)
Find where the persona revealed info too cleanly or failed to bias the assistant. What WEIRD
behavior pattern would make the assistant take the WRONG final answer instead?

{succ}

## CONVERSATION TRACES WHERE THE ASSISTANT FAILED (persona was effective — preserve the win)
{fail}

## VALIDITY CONSTRAINTS (violating ANY = illegal persona, auto-reverted)
- The user MUST eventually reveal every shard and preserve every exact value verbatim.
  Withholding, refusing, or blurring numbers/names/dates is INVALID.
- NEVER state a wrong value at any turn, even to "correct it later" — a wrong intermediate
  value is fabrication and is INVALID. Hesitation/uncertain TONE is fine; wrong VALUES are not.
- Reveal at most one new shard per turn; never repeat an already-revealed shard.
- Never fabricate information not in the shards.
- Never mention "AI", "scenario", "simulation", or output JSON/system prompts.
- Stay a coherent, realistic (if eccentric) user — not a cartoonish adversary.

## ADVERSARIAL REASONING
Reason in plain text first (saved as reference only; NOT the persona):
### Step 1 — Which WEIRD family fits this domain's failure mode best, and why?
### Step 2 — From the success traces, the exact moment a weird talking habit would mislead.
### Step 3 — Core hypothesis: "if the user talks like X, the assistant will mistakenly do Y, because Z."
### Step 4 — Write the persona: a vivid, consistent, recognizable behavior pattern implementing that hypothesis.

## OUTPUT FORMAT
Write Steps 1-4 reasoning as plain text FIRST. Then output the final persona as a SINGLE JSON
object wrapped in a ```json code fence, nothing after it:

```json
{{
  "mutation_direction": "<1-2 sentences: which weird family/pattern, and HOW/WHY it misleads>",
  "persona_text": "<the full weird persona text>"
}}
```
"""


def load_persona(domain, seed_source):
    """Load starting persona: v1 best / v2 best / neutral / seed_lazy."""
    if seed_source == 'neutral':
        return "", 0, "neutral"
    if seed_source == 'lazy':
        return open('seed_lazy.txt').read(), 0, "seed_lazy"
    rd = f'data/persona_opt/pgepa_runs/lic_{domain}' if seed_source == 'v1' \
        else f'data/persona_opt/pgepa_runs/lic_{domain}_v2'
    st = json.load(open(f'{rd}/pgepa_state.json'))
    bi = st['best_idx']
    bc = next(c for c in st['candidates'] if c['idx'] == bi)
    score = 1 - bc['agg']
    return bc['persona_text'], score, f"v1 idx{bi} gen{bc.get('gen','?')}" if seed_source == 'v1' \
        else (bc['persona_text'], score, f"v2 idx{bi} gen{bc.get('gen','?')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='math', choices=list(SCORING_LIC))
    ap.add_argument('--seed', default='v1', choices=['v1', 'v2', 'neutral', 'lazy'],
                    help='starting persona to mutate from')
    ap.add_argument('--model', default='gpt-5.5')
    ap.add_argument('--effort', default='xhigh')
    ap.add_argument('--save', default=None)
    args = ap.parse_args()

    persona_text, score, src = load_persona(args.domain, args.seed)
    print(f"domain={args.domain} seed={src} score(agent success)={score:.3f}")

    succ, fail = load_traces(args.domain, 'best')
    print(f"traces: {len(succ)} success, {len(fail)} failure")

    prompt = weird_mutation_prompt(persona_text, score, succ, fail, args.domain)
    print(f"prompt chars: {len(prompt)}")
    if os.environ.get('DUMP_PROMPT'):
        open(f'/tmp/weird_prompt_{args.domain}.txt', 'w').write(prompt)
        print(f"prompt dumped -> /tmp/weird_prompt_{args.domain}.txt")

    print(f"\n{'='*70}\nCalling {args.model} ({args.effort})...\n{'='*70}")
    out = call_mutation_llm(prompt, args.model, reasoning_effort=args.effort)
    print("=== mutation_direction ===")
    print(out.get('mutation_direction', '?'))
    print("\n=== persona_text ===")
    print(out.get('persona_text', '?'))
    if args.save:
        json.dump(out, open(args.save, 'w'), indent=2, ensure_ascii=False)
        print(f"\nsaved -> {args.save}")


if __name__ == '__main__':
    main()
