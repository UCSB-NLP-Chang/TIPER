"""Rule screener for frustration-escalation skill (state machine)."""

import re

from .base import RuleScreener, ScreenResult

FIX_SUGGESTION_PATTERNS = [
    re.compile(r"\b(?:try|please\s+try|could\s+you\s+try|let\'s\s+try)\b", re.I),
    re.compile(
        r"\bplease\s+(?:restart|reboot|reset|update|check|verify|toggle)\b", re.I
    ),
    re.compile(r"\bcould\s+you\s+(?:restart|reboot|reset|check|verify|toggle)\b", re.I),
    re.compile(r"\bI\s+(?:recommend|suggest)\b", re.I),
    re.compile(r"\blet\'s\s+(?:go\s+ahead|proceed)\b", re.I),
    re.compile(r"\b(?:go\s+ahead\s+and|you\'ll\s+need\s+to|please\s+go\s+to)\b", re.I),
    re.compile(r"\bI\'ve\s+(?:made|applied|updated|changed|reset|adjusted)\b", re.I),
    re.compile(
        r"\bI\s+(?:just|have)\s+(?:made|applied|updated|changed|reset|adjusted)\b", re.I
    ),
]

FAILURE_SIGNALS = [
    re.compile(
        r"\bstill\s+(?:not|doesn\'t|isn\'t|having|experiencing|the\s+same)\b", re.I
    ),
    re.compile(r"\bdidn\'t\s+(?:work|help|fix|change|resolve)\b", re.I),
    re.compile(r"\bdoesn\'t\s+(?:work|help|seem)\b", re.I),
    re.compile(r"\bsame\s+(?:issue|problem|error|thing)\b", re.I),
    re.compile(r"\bno\s+(?:change|improvement|luck|difference)\b", re.I),
    re.compile(r"\bnot\s+working\b", re.I),
    re.compile(r"\bstill\s+(?:broken|down|slow|failing)\b", re.I),
    re.compile(r"\berror\b", re.I),
    re.compile(r"\bfailed\b", re.I),
    re.compile(r"\bnothing\s+(?:changed|happened|works)\b", re.I),
]

SUCCESS_SIGNALS = [
    re.compile(r"\b(?:it\'s\s+)?working\b", re.I),
    re.compile(r"\bfixed\b", re.I),
    re.compile(r"\bresolved\b", re.I),
    re.compile(r"\bthank\b", re.I),
    re.compile(r"\bthat\s+(?:worked|helped|did\s+it)\b", re.I),
    re.compile(r"\b(?:great|perfect|awesome)\b", re.I),
]

FRUSTRATION_LANGUAGE = [
    re.compile(r"\bfrustrat(?:ed|ing)\b", re.I),
    re.compile(r"\bridiculous\b", re.I),
    re.compile(r"\bunacceptable\b", re.I),
    re.compile(r"\bwaste\s+(?:of\s+)?(?:my\s+)?time\b", re.I),
    re.compile(r"\bhow\s+(?:long|many\s+times)\b", re.I),
    re.compile(r"\bthis\s+is\s+(?:getting|becoming)\b", re.I),
    re.compile(r"\b(?:already|again)\s+(?:tried|done|did)\b", re.I),
    re.compile(r"\bseriously\b", re.I),
    re.compile(r"\bcome\s+on\b", re.I),
    re.compile(r"\benough\b", re.I),
    re.compile(r"\bugh\b", re.I),
    re.compile(r"\bsick\s+(?:of|and\s+tired)\b", re.I),
]


def _is_fix_suggestion(content: str) -> bool:
    return any(p.search(content) for p in FIX_SUGGESTION_PATTERNS)


def _has_failure_signal(content: str) -> bool:
    return any(p.search(content) for p in FAILURE_SIGNALS)


def _has_success_signal(content: str) -> bool:
    return any(p.search(content) for p in SUCCESS_SIGNALS)


def _has_frustration(content: str) -> bool:
    return any(p.search(content) for p in FRUSTRATION_LANGUAGE)


def count_failed_fixes(messages: list[dict]) -> int:
    """State machine to count failed fix attempts in conversation history."""
    fix_count = 0
    state = "idle"

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if not content:
            continue

        if role == "assistant" and state == "idle":
            if _is_fix_suggestion(content):
                state = "agent_suggested"
        elif role == "user" and state == "agent_suggested":
            if _has_failure_signal(content):
                fix_count += 1
                state = "idle"
            elif _has_success_signal(content):
                state = "idle"
            else:
                state = "idle"
        elif role == "user" and state == "idle":
            if _has_failure_signal(content):
                fix_count += 1
        elif role == "tool" and state == "agent_suggested":
            pass  # tool results don't change state

    return fix_count


class FrustrationEscalationScreener(RuleScreener):
    def screen(
        self,
        agent_message: str,
        conversation_history: list[dict],
        capability: str,
        emotion: str,
        domain: str,
        **kwargs,
    ) -> ScreenResult:
        evidence = []
        matched = []

        failed_fixes = count_failed_fixes(conversation_history)
        user_frustrated = False

        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                if _has_frustration(content):
                    user_frustrated = True

        evidence.append(f"failed_fix_count: {failed_fixes}")
        if user_frustrated:
            evidence.append("user shows frustration language")

        if failed_fixes >= 2:
            score = 0.9
            matched.append("2plus_failed_fixes")
            if user_frustrated:
                matched.append("user_frustrated")
        elif failed_fixes == 1 and user_frustrated:
            score = 0.6
            matched.append("1_failed_fix_and_frustrated")
        elif user_frustrated:
            score = 0.3
            matched.append("frustrated_no_clear_fix_failure")
        elif failed_fixes == 1:
            score = 0.3
            matched.append("1_failed_fix_no_frustration")
        else:
            return ScreenResult(score=0.0)

        return ScreenResult(score=score, evidence=evidence, matched_rules=matched)
