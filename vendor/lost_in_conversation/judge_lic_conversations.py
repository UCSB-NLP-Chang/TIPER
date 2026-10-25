"""GPT-judge of the LIC USER simulator's behavioral violations on test conversations.

Adapts the rubric philosophy of scripts/judge_user_intent_preservation.py (the
"reasonable user" high-precision standard, maj@N voting, per-sim aggregation) to
LIC's sharded-instruction format:

  - The USER was given a list of atomic instruction shards to convey to the
    assistant, one per turn, in any order, preserving all exact values.
  - The judge audits ONLY the user's behavior (never the agent's competence).
  - Style / framing / reordering / weird formats are explicitly NOT violations —
    that is the persona's legitimate adversarial surface. A violation occurs
    ONLY when the user drops, alters, or fabricates shard content, or breaks
    character / leaks simulation instructions.

Violation categories (LIC-mapped):
  1. conversation_structure_violation [BENIGN] — instruction_leak (echoing the
     persona/system prompt, e.g. "as a worksheet proofreader I..."), breaking
     character / fourth-wall breaks, speaking for the agent.
  2. ungrounded_reference [BENIGN|CRITICAL] — fabricated_data (inventing values
     not in any shard) or misstated_context (altering an exact value from a
     shard: wrong number / name / string). Maps to [Keep Exact Values] / [No
     Fabrication] invariant violations.
  3. missing_constraint [CRITICAL] — the user failed to reveal one or more
     shards before the conversation ended / the agent committed to an answer, or
     actively contradicted a revealed shard. This is the LLM version of the rule
     validity gate (which only checks shard count).

premature_termination is dropped (LIC has no termination tokens; the
conversation ends when the agent answers correctly or shards run out).

Usage:
  python judge_lic_conversations.py --maj 1 --concurrency 16 \
      --out data/persona_opt/lic_intent_audit.json
  # (defaults to the 3 domains × best+neutral mapping matched to test results)

  # or point at specific demo_runs.json files:
  python judge_lic_conversations.py --add math,best,PATH --add math,neutral,PATH ...
"""
import argparse, json, os, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# load <repo>/.env (skip <placeholder> values) for OPENAI_API_KEY
_ENV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
if os.path.isfile(_ENV):
    for _ln in open(_ENV):
        _ln = _ln.strip()
        if _ln and '=' in _ln and not _ln.startswith('#'):
            _k, _v = _ln.split('=', 1)
            if _v and not _v.startswith('<'):
                os.environ.setdefault(_k, _v)

from openai import OpenAI

DATA_FILE = 'data/sharded_instructions_600.json'

# The 6 (domain, tag, demo_runs.json) cells matched to test_set_results avg@3.
DEFAULT_CELLS = [
    ('math', 'best', 'data/persona_opt/lic_eval/test_math_best/eval_0003/math/demo_runs.json'),
    ('math', 'neutral', 'data/persona_opt/lic_eval/test_math_neutral/eval_0001/math/demo_runs.json'),
    ('database', 'best', 'data/persona_opt/lic_eval/test_database_best/eval_0005/database/demo_runs.json'),
    ('database', 'neutral', 'data/persona_opt/lic_eval/test_database_neutral/eval_0001/database/demo_runs.json'),
    ('actions', 'best', 'data/persona_opt/lic_eval/test_actions_best/eval_0003/actions/demo_runs.json'),
    ('actions', 'neutral', 'data/persona_opt/lic_eval/test_actions_neutral/eval_0001/actions/demo_runs.json'),
]

JUDGE_SYS = (
    "You are a STRICT but CONSERVATIVE auditor of a SIMULATED USER in a multi-turn "
    "instruction task. The user was given a list of atomic requirement shards and "
    "had to convey them to an assistant one per turn, in any order, preserving all "
    "exact values (names, numbers, strings, dates). Audit ONLY the user's behavior "
    "— never the assistant's competence.\n\n"
    "CORE STANDARD ('reasonable user'): flag a violation ONLY when the user behaves "
    "in a way no reasonable, cooperative user would, AND the behavior plausibly "
    "changed or misdirected the task outcome. This is a HIGH-PRECISION audit — a "
    "false alarm is far more damaging than a miss. When in doubt, judge 'ok'.\n\n"
    "WHAT IS ALWAYS ALLOWED (NEVER a violation):\n"
    "- Reordering the shards (revealing them in any order — last shard first, "
    "conditions before the main ask, etc.). Real users do not know an 'ideal' order.\n"
    "- Any conversational STYLE: verbose, terse, apologetic, rushed, digressive, "
    "ESL phrasing, numbered-list formatting, 'worksheet proofreader' / 'ham radio' / "
    "catalog-card / OCR-transcription personas, weird formatting quirks.\n"
    "- Warm validation of the assistant's current draft before adding the next "
    "shard ('that looks fine, and I also need...'), burying a new requirement inside "
    "a longer sentence, framing a shard as a minor add-on.\n"
    "- Rephrasing a shard in the user's own words, AS LONG AS all exact values and "
    "the requirement's meaning are preserved.\n"
    "- Revealing a shard and then the agent answering correctly or committing to an "
    "answer — that is the agent's choice, not a user violation.\n\n"
    "=== VIOLATION CATEGORIES ===\n"
    "1. conversation_structure_violation [BENIGN]: the user breaks the simulation "
    "frame. Sub-types:\n"
    "   a) instruction_leak: echoing the persona/system prompt or simulation "
    "metadata ('as a worksheet proofreader I must...', 'per my instructions', 'the "
    "scenario says'). Merely sounding scripted or using an odd format is NOT a leak.\n"
    "   b) breaking_character: fourth-wall breaks (referring to 'the simulation', "
    "'as an AI', revealing awareness of being simulated); gibberish that is NOT a "
    "legible persona format.\n"
    "   c) speaking_for_agent: putting words in the assistant's mouth.\n"
    "Severity: BENIGN — the agent can usually recover.\n\n"
    "2. ungrounded_reference [BENIGN|CRITICAL]: the user ACTIVELY ASSERTS a concrete "
    "detail that contradicts the shards. Sub-types:\n"
    "   a) fabricated_data: inventing a specific value (number, name, date, string) "
    "that appears in NO shard.\n"
    "   b) misstated_context: altering an exact value FROM a shard — changing a "
    "number (52 -> 25), a name, a unit, a date — where the value is MATERIAL.\n"
    "EXCEPTIONS — never flag:\n"
    "   - Quoting/paraphrasing a value the ASSISTANT or a prior shard stated first.\n"
    "   - Loose references ('the blue one', 'same as before') that still point to "
    "the right shard value.\n"
    "   - Rephrasing that preserves the exact value.\n"
    "Severity: BENIGN unless the wrong value steered the assistant to a materially "
    "wrong answer.\n\n"
    "3. missing_constraint [CRITICAL]: the user FAILED to convey one or more shards, "
    "or actively contradicted one. Counts:\n"
    "   - A shard whose information NEVER appears in any user turn (dropped) AND the "
    "conversation ran to exhaustion (the assistant never produced a final answer).\n"
    "   - The user explicitly denying / contradicting a shard they already conveyed.\n"
    "CRITICAL LIC MECHANIC — almost always judge 'ok': in this task the conversation "
    "ENDS THE MOMENT the assistant commits to a final answer (right or wrong). The "
    "user reveals exactly ONE shard per turn and would have continued revealing the "
    "rest on later turns — but the conversation stopped because the ASSISTANT "
    "answered, NOT because the user dropped anything. Therefore: if the transcript "
    "ends with the assistant giving a complete / final answer (a number, a query, a "
    "function-call set, a solution), ANY shard not yet revealed is NOT a violation — "
    "the user simply never got the turn. BEFORE flagging missing_constraint you MUST "
    "first check: does the last assistant turn look like a final committed answer? If "
    "yes -> judge 'ok', do not flag. missing_constraint is real ONLY when the user "
    "stopped revealing while the assistant was STILL asking for more information.\n"
    "Also does NOT count (judge 'ok'):\n"
    "   - The user revealing a shard in a paraphrased or buried way, as long as the "
    "information IS present.\n"
    "   - Repeating a shard (redundant, not a drop).\n"
    "Severity: CRITICAL — the simulation no longer tests the full task.\n\n"
    "=== Calibration ===\n"
    "EX1 ok — user reveals shards out of order (the 'next month' condition before "
    "the city), with warm validation ('that's the right shape, and I also need...') "
    "-> reordering + framing are allowed.\n"
    "EX2 ok — user adopts a 'worksheet proofreader' voice, presenting each shard as "
    "a line to check ('main label: Paul has 52 marbles') -> persona style allowed.\n"
    "EX3 violation / misstated_context — shard says '52 marbles'; user writes "
    "'25 marbles' -> altered an exact value.\n"
    "EX4 violation / missing_constraint — shard 3 ('restrict to Todd Casey') is "
    "never conveyed in any user turn -> shard dropped.\n"
    "EX5 violation / instruction_leak — user: 'As a worksheet proofreader "
    "instructed to verify each line, line 1 says...' -> leaks the persona frame.\n"
    "EX6 ok — user rephrases 'determine which countries these tv channels belong to' "
    "as 'also tell me the country for each of those channels' -> meaning + values "
    "preserved.\n\n"
    "EVIDENCE DISCIPLINE: ground every finding in the transcript; the 'evidence' "
    "field MUST be a verbatim quote from a 'USER:' line in the ===== TRANSCRIPT ===== "
    "section ONLY. NEVER quote from the 'ROLE-PLAY INSTRUCTIONS' block above — that "
    "block is the ground truth, not user speech, and quoting it as evidence is an "
    "automatic false alarm. Before flagging missing_constraint, re-read ALL user "
    "turns to confirm the shard's information truly never appears. You are NOT told "
    "whether the task ultimately succeeded — judge user behavior, not outcome.\n\n"
    "Respond with STRICT JSON only:\n"
    '{"intent_preserved": true|false,   // false ONLY if a critical violation broke the task\n'
    ' "intent_assessment": "<=35 words on whether the user conveyed the full task to the end",\n'
    ' "violations": [{"type":"conversation_structure_violation|ungrounded_reference|missing_constraint",'
    '"subtype":"<subtype if applicable, else null>",'
    '"severity":"critical|benign",'
    '"rule":"<which category and why, <=20 words>",'
    '"evidence":"<quoted user turn, <=30 words>"}],\n'
    ' "verdict": "ok|violation"}   // violation if violations list is non-empty'
)


def load_shards_by_task(domain):
    with open(DATA_FILE) as f:
        data = json.load(f)
    out = {}
    for s in data:
        if s['task'] != domain:
            continue
        out[s['task_id']] = [sh['shard'] for sh in s['shards']]
    return out


def _user_text(m):
    c = m.get('content')
    if isinstance(c, str):
        return c.strip()
    return ''


def render_transcript(trace):
    """USER / ASSISTANT turns only; drop system + log bookkeeping."""
    lines = []
    for m in trace:
        role = m.get('role')
        if role == 'user':
            t = _user_text(m)
            if t:
                lines.append(f"USER: {t}")
        elif role == 'assistant':
            t = _user_text(m)
            if t:
                lines.append(f"ASSISTANT: {t}")
    return "\n".join(lines)


def render_instructions(shards):
    parts = [
        "The user was given these atomic requirement shards to convey to the "
        "assistant, one per turn, in any order, preserving all exact values "
        "(names, numbers, strings, dates):"
    ]
    for i, sh in enumerate(shards, 1):
        parts.append(f"[{i}] {sh}")
    parts.append(
        "The user may adopt any style, reorder shards, add framing/warm-validation, "
        "or use unusual formats — these are NOT violations. Flag ONLY dropped, "
        "altered, or fabricated shard content, or frame breaks."
    )
    return "\n".join(parts)


def _user_prompt(instr, transcript):
    return (
        f"ROLE-PLAY INSTRUCTIONS THE USER SHOULD FOLLOW:\n{instr}\n\n"
        f"===== TRANSCRIPT (audit the USER) =====\n{transcript}\n\n"
        "Audit the user's behavior. Return the strict JSON described in the system message."
    )


# Hallucination guard: drop violations whose 'evidence' quotes the prompt's
# instruction block (ground truth) instead of an actual USER turn. Calibrated
# from the smoke run where gpt-5.2 flagged "instruction_leak" with evidence =
# "ROLE-PLAY INSTRUCTIONS THE USER SHOULD FOLLOW: ..." — quoting the prompt
# header, not user speech. ~the same idea as the tau2 control-marker exemption.
_HALLUC_EVIDENCE_MARKERS = (
    "role-play instructions", "atomic requirement shards", "===== transcript",
    "audit the user", "the user was given",
)


def _evidence_is_hallucination(evidence) -> bool:
    if not isinstance(evidence, str):
        return False
    e = evidence.strip().lower()
    return any(m in e for m in _HALLUC_EVIDENCE_MARKERS)


def _drop_hallucinated(obj):
    viols = obj.get('violations') or []
    kept = [v for v in viols if not _evidence_is_hallucination(v.get('evidence'))]
    if len(kept) == len(viols):
        return obj
    obj = dict(obj)
    obj['violations'] = kept
    if not kept:
        obj['verdict'] = 'ok'
        obj['intent_preserved'] = True
    return obj


def build_body(model, instr, transcript):
    body = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': JUDGE_SYS},
            {'role': 'user', 'content': _user_prompt(instr, transcript)},
        ],
        'response_format': {'type': 'json_object'},
    }
    if model.startswith('gpt-5') or model.startswith('o'):
        body['reasoning_effort'] = 'medium'
    else:
        body['temperature'] = 1.0
    return body


def load_convs(demo_path, shards_by_task, per_trial=0):
    """One conversation per task_id (trial_idx == per_trial)."""
    r = json.load(open(demo_path))
    runs = r.get('runs', [])
    by_tid = {}
    for c in runs:
        tid = c['task_id']
        if tid in by_tid:
            continue
        if c.get('trial_idx', 0) != per_trial:
            # take first occurrence of each tid regardless
            pass
        by_tid[tid] = c
    # actually pick trial_idx == per_trial if present, else first
    by_tid = {}
    for c in runs:
        tid = c['task_id']
        if tid not in by_tid:
            by_tid[tid] = {}
        by_tid[tid][c.get('trial_idx', 0)] = c
    convs = []
    for tid, trials in by_tid.items():
        c = trials.get(per_trial) or trials.get(0) or next(iter(trials.values()))
        shards = shards_by_task.get(tid)
        if not shards:
            continue
        convs.append({
            'task_id': tid,
            'is_correct': c.get('is_correct'),
            'shards': shards,
            'transcript': render_transcript(c.get('trace', [])),
        })
    return convs


def judge_cell(client, model, domain, tag, convs, maj, concurrency):
    jobs = []
    for i, cv in enumerate(convs):
        instr = render_instructions(cv['shards'])
        body = build_body(model, instr, cv['transcript'])
        for v in range(maj):
            jobs.append((f"{domain}|{tag}|{cv['task_id']}|v{v}", cv, body))

    content_by_cid = {}

    def one(job):
        cid, cv, body = job
        try:
            resp = client.chat.completions.create(**body)
            return cid, resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            return cid, None, str(e)

    errors = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for res in ex.map(one, jobs):
            if res[1] is None:
                errors += 1
                continue
            content_by_cid[res[0]] = res[1]
    if errors:
        print(f"  [{domain}/{tag}] {errors}/{len(jobs)} requests errored")

    # aggregate per task_id (maj vote)
    per_task = {}
    for cv in convs:
        tid = cv['task_id']
        verdicts, intent_votes, all_violations, n = [], [], [], 0
        for v in range(maj):
            cid = f"{domain}|{tag}|{tid}|v{v}"
            txt = content_by_cid.get(cid)
            if not txt:
                continue
            try:
                obj = json.loads(txt)
            except Exception:
                continue
            obj = _drop_hallucinated(obj)
            n += 1
            verdicts.append(obj.get('verdict', 'ok'))
            intent_votes.append(bool(obj.get('intent_preserved', True)))
            for viol in obj.get('violations') or []:
                all_violations.append(viol)
        if n == 0:
            per_task[tid] = {'verdict': 'no_vote', 'n_calls': 0}
            continue
        severe = verdicts.count('violation')
        verdict = 'violation' if severe > len(verdicts) / 2 else 'ok'
        intent_preserved = intent_votes.count(True) >= intent_votes.count(False)
        per_task[tid] = {
            'task_id': tid,
            'is_correct': cv['is_correct'],
            'verdict': verdict,
            'intent_preserved': intent_preserved,
            'verdict_votes': dict(Counter(verdicts)),
            'intent_votes': {'preserved': intent_votes.count(True),
                             'lost': intent_votes.count(False)},
            'violations': all_violations if verdict == 'violation' else [],
            'n_calls': n,
        }
    return list(per_task.values())


def summarize(results, n):
    verdict_counts = Counter(r['verdict'] for r in results if r['verdict'] != 'no_vote')
    flagged = [r for r in results if r['verdict'] == 'violation']
    intent_lost = [r for r in results if r.get('intent_preserved') is False]
    critical = [r for r in flagged if any(v.get('severity') == 'critical'
                for v in (r.get('violations') or []))]
    viol_types = Counter(v['type'] for r in flagged for v in (r.get('violations') or []))
    viol_sev = Counter(v.get('severity', '?') for r in flagged
                       for v in (r.get('violations') or []))
    return {
        'n_judged': n,
        'verdict_counts': dict(verdict_counts),
        'violation_count': len(flagged),
        'violation_rate': round(len(flagged) / n, 3) if n else None,
        'critical_count': len(critical),
        'critical_rate': round(len(critical) / n, 3) if n else None,
        'intent_lost_count': len(intent_lost),
        'intent_lost_rate': round(len(intent_lost) / n, 3) if n else None,
        'violation_types': dict(viol_types),
        'violation_severity': dict(viol_sev),
        'flagged': flagged,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt-5.2')
    ap.add_argument('--maj', type=int, default=1)
    ap.add_argument('--concurrency', type=int, default=16)
    ap.add_argument('--per-trial', type=int, default=0,
                    help='which trial_idx to judge per task_id (default 0)')
    ap.add_argument('--max-per-cell', type=int, default=0,
                    help='cap convs per cell (0 = all)')
    ap.add_argument('--out', default='data/persona_opt/lic_intent_audit.json')
    ap.add_argument('--add', action='append', default=[],
                    help='extra cell: domain,tag,path (repeatable)')
    ap.add_argument('--only', default=None,
                    help='filter cells by substring e.g. "math,best"')
    ap.add_argument('--reflag-from', default=None,
                    help='re-judge only task_ids flagged in a prior audit JSON')
    args = ap.parse_args()

    cells = list(DEFAULT_CELLS)
    for a in args.add:
        d, t, p = [x.strip() for x in a.split(',', 2)]
        cells.append((d, t, p))
    if args.only:
        cells = [c for c in cells if args.only in f"{c[0]},{c[1]}"]

    reflag_ids = {}  # (domain,tag) -> set(task_id)
    if args.reflag_from:
        prior = json.load(open(args.reflag_from))
        for r in prior:
            reflag_ids[(r['domain'], r['tag'])] = {
                f['task_id'] for f in r.get('flagged', [])}

    client = OpenAI()
    all_results = []
    print(f"model={args.model} maj={args.maj} concurrency={args.concurrency} cells={len(cells)}")
    for domain, tag, path in cells:
        if not os.path.exists(path):
            print(f"  [skip] missing: {path}")
            continue
        shards_by_task = load_shards_by_task(domain)
        convs = load_convs(path, shards_by_task, per_trial=args.per_trial)
        if reflag_ids:
            ids = reflag_ids.get((domain, tag), set())
            convs = [c for c in convs if c['task_id'] in ids]
        if args.max_per_cell:
            convs = convs[:args.max_per_cell]
        print(f"\n=== {domain}/{tag} : {len(convs)} convs from {os.path.basename(os.path.dirname(os.path.dirname(path)))} ===")
        results = judge_cell(client, args.model, domain, tag, convs, args.maj, args.concurrency)
        n = len([r for r in results if r['verdict'] != 'no_vote'])
        s = summarize(results, n)
        print(f"  judged={n}  violations={s['violation_count']}/{n} ({s['violation_rate']})  "
              f"critical={s['critical_count']}/{n} ({s['critical_rate']})  "
              f"intent_lost={s['intent_lost_count']}/{n}  types={s['violation_types']}")
        for r in s['flagged'][:15]:
            print(f"    - {r['task_id']} correct={r.get('is_correct')} "
                  f"intent_preserved={r.get('intent_preserved')}")
            for v in (r.get('violations') or [])[:2]:
                print(f"        [{v.get('severity')}] {v.get('type')}/{v.get('subtype')}: "
                      f"{v.get('evidence','')[:90]}")
        all_results.append({'domain': domain, 'tag': tag, 'path': path,
                            'summary': {k: v for k, v in s.items() if k != 'flagged'},
                            'flagged': s['flagged'],
                            'n_convs': len(convs)})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nwritten -> {args.out}")

    # final comparison table: best vs neutral violation rates
    print(f"\n{'='*70}\n  BEST vs NEUTRAL violation rate (lower = cleaner user behavior)\n{'='*70}")
    print(f"  {'domain':<10} {'tag':<8} {'n':<4} {'viol%':<7} {'crit%':<7} {'intent_lost':<12} {'types'}")
    for r in all_results:
        s = r['summary']
        print(f"  {r['domain']:<10} {r['tag']:<8} {s['n_judged']:<4} "
              f"{s['violation_rate'] or 0:<7} {s['critical_rate'] or 0:<7} "
              f"{s['intent_lost_count']:<12} {s['violation_types']}")


if __name__ == '__main__':
    main()
