"""Stream filter for vita_pgepa logs. Emits only meaningful events and suppresses
the high-volume benign noise:

  * sim DEBUG spam (orchestrator step / tool response / cost lines)
  * benign Tracebacks: tool calls with bad args (bad id/address/NameError) raise
    ValueError inside environment.get_response, which CATCHES them, prints them
    via traceback.print_exc(), returns "Error: ..." to the agent, and the sim
    CONTINUES. These are not fatal. The sim DEBUG logging interleaves with
    traceback.print_exc output, so block-based traceback reconstruction is
    unreliable -- instead we filter line-by-line:
      - drop "Traceback (most recent call last):" headers and "  File ..." frames
      - drop ValueError lines and "Error: ..." response lines (benign tool errors)
      - emit a line ONLY if it names a FATAL exception type (KeyError, TypeError,
        AttributeError, RuntimeError, AssertionError, ImportError, RecursionError,
        NotImplementedError, "Exception in thread", etc.) -- real crashes.

Meaningful single-line events that always pass:
  gepa Iteration milestones, Selected candidate, BATCH INFEASIBLE (severe gate),
  test-phase signals (evaluate_reward / test_report / TEST EVAL), gepa start,
  Loading gepa state, best_agg / val_difficulty.
Online-judge lines pass only when a SEVERE finding or gating is reported.
"""
import sys, re

KEY = re.compile(
    r"Iteration [0-9]+:|Selected program|Selected candidate|BATCH INFEASIBLE|"
    r"Exception in thread|Error processing|"
    r"KeyError|not found in registry|MAXIMUM.*REACHED|Stopper|"
    r"evaluate_reward|TEST EVAL|test_trials|test_report|"
    r"Loading gepa state|best_agg|val_difficulty"
)
JUDGE = re.compile(r"\[online judge\]")
SEVERE = re.compile(r"([1-9][0-9]*) severe|BATCH INFEASIBLE|severe_rate")
# lines that are part of a (likely benign) traceback: headers + indented frames
TB_NOISE = re.compile(r"^Traceback \(most recent call last\):|^  File |^\s+\^+|^\s+~~~")
# benign caught-tool exceptions (get_response catches these -> sim continues)
BENIGN_EXC = re.compile(r"^ValueError:|^Error: |Response: Error:")
# fatal exception types that indicate a REAL crash (not a caught tool error)
FATAL_EXC = re.compile(
    r"^(KeyError|TypeError|AttributeError|RuntimeError|AssertionError|"
    r"ImportError|ModuleNotFoundError|RecursionError|NotImplementedError|"
    r"OverflowError|ZeroDivisionError|FileNotFoundError|PermissionError|"
    r"ConnectionError|TimeoutError|MemoryError|SystemExit|"
    r"Exception in thread|concurrent\.futures\._base\.CancelledError):"
)

for line in sys.stdin:
    if TB_NOISE.search(line):
        continue
    if BENIGN_EXC.search(line):
        continue
    if FATAL_EXC.search(line):
        sys.stdout.write(line)
        sys.stdout.flush()
        continue
    if KEY.search(line) or (JUDGE.search(line) and SEVERE.search(line)):
        sys.stdout.write(line)
        sys.stdout.flush()
