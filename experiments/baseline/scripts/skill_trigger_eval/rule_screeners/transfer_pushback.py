"""Rule screener for transfer-pushback skill."""

import re

from .base import RuleScreener, ScreenResult

TRANSFER_OFFER_PATTERNS = [
    (re.compile(r"\btransfer\s+(you|your\s+call)\b", re.I), "transfer_you"),
    (re.compile(r"\bescalat(e|ing|ion)\b", re.I), "escalation"),
    (re.compile(r"\bconnect\s+you\s+(to|with)\b", re.I), "connect_you"),
    (re.compile(r"\bput\s+you\s+through\b", re.I), "put_through"),
    (re.compile(r"\bsupervisor\b", re.I), "supervisor"),
    (re.compile(r"\bspecialist\b", re.I), "specialist"),
    (re.compile(r"\bmanager\b", re.I), "manager"),
    (re.compile(r"\btechnical\s+support\b", re.I), "tech_support"),
    (re.compile(r"\bhigher.?level\s+support\b", re.I), "higher_level"),
    (re.compile(r"\bdifferent\s+(department|team|agent)\b", re.I), "different_dept"),
]

OFFER_FORM_PATTERNS = [
    re.compile(r"\bwould\s+you\s+like\b", re.I),
    re.compile(r"\bshall\s+I\b", re.I),
    re.compile(r"\bI\s+can\s+transfer\b", re.I),
    re.compile(r"\blet\s+me\s+transfer\b", re.I),
    re.compile(r"\blet\s+me\s+connect\b", re.I),
    re.compile(r"\?"),
]

USER_REQUESTED_TRANSFER = [
    re.compile(r"\btransfer\s+me\b", re.I),
    re.compile(r"\bspeak\s+to\s+(a\s+)?(supervisor|manager)\b", re.I),
    re.compile(r"\bescalate\b", re.I),
    re.compile(r"\bget\s+me\s+(a\s+)?(supervisor|manager)\b", re.I),
    re.compile(r"\bsomeone\s+else\b", re.I),
]

TRANSFER_EXECUTED = [
    re.compile(r"YOU\s+ARE\s+BEING\s+TRANSFERRED", re.I),
    re.compile(r"transferring\s+you\s+now", re.I),
    re.compile(r"I\'ve\s+transferred\s+you", re.I),
]


class TransferPushbackScreener(RuleScreener):
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
        score = 0.0

        if not agent_message:
            return ScreenResult(score=0.0)

        # Check if this is a transfer execution announcement (not an offer)
        for pat in TRANSFER_EXECUTED:
            if pat.search(agent_message):
                evidence.append("transfer_executed: announcement, not offer")
                return ScreenResult(
                    score=0.1,
                    evidence=evidence,
                    matched_rules=["transfer_executed_only"],
                )

        # Check for transfer/escalation keywords in agent message
        keyword_hits = []
        for pat, name in TRANSFER_OFFER_PATTERNS:
            if pat.search(agent_message):
                keyword_hits.append(name)
                evidence.append(f"matched: '{name}' in agent msg")

        if not keyword_hits:
            return ScreenResult(score=0.0)

        matched.append("transfer_keyword_hit")

        # Check if agent is offering (question/offer form)
        is_offer_form = any(p.search(agent_message) for p in OFFER_FORM_PATTERNS)
        if is_offer_form:
            score = 1.0
            matched.append("offer_form")
            evidence.append("agent uses offer/question form")
        else:
            score = 0.7
            evidence.append("declarative form (no explicit offer)")

        # Check if user previously requested transfer (reduces relevance for pushback)
        user_requested = False
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                if any(p.search(content) for p in USER_REQUESTED_TRANSFER):
                    user_requested = True
                    break

        if user_requested:
            score = max(score - 0.4, 0.1)
            matched.append("user_previously_requested")
            evidence.append(
                "user previously requested transfer → less likely pushback trigger"
            )

        return ScreenResult(score=score, evidence=evidence, matched_rules=matched)
