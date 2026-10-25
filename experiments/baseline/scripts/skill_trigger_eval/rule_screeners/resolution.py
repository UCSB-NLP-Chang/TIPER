"""Rule screener for resolution / resolution-banking skill."""

import re

from .base import RuleScreener, ScreenResult

TELECOM_RESOLUTION_PATTERNS = [
    (re.compile(r"speed\s*test.*excellent", re.I), "speed_test_excellent"),
    (
        re.compile(r"signal\s*strength.*(?:strong|excellent|good)", re.I),
        "signal_strong",
    ),
    (re.compile(r"issue.*(?:resolved|fixed)", re.I), "issue_resolved"),
    (
        re.compile(
            r"successfully\s+(?:activated|configured|updated|enabled|reset|added|removed)",
            re.I,
        ),
        "success_action",
    ),
    (re.compile(r"problem.*(?:resolved|fixed|solved)", re.I), "problem_resolved"),
    (
        re.compile(r"(?:service|connection|data|network).*(?:restored|working)", re.I),
        "service_restored",
    ),
    (
        re.compile(r"everything.*(?:look|seem).*(?:good|fine|normal)", re.I),
        "looks_good",
    ),
    (
        re.compile(
            r"has\s+been\s+(?:added|enabled|activated|configured|updated|reset|removed|applied)",
            re.I,
        ),
        "has_been_action",
    ),
    (
        re.compile(
            r"(?:data|plan|feature|roaming)\s+(?:has\s+been\s+)?(?:added|enabled|activated)",
            re.I,
        ),
        "feature_added",
    ),
    (re.compile(r"(?:GB|gigabyte|data)\s+(?:has\s+been\s+)?added", re.I), "data_added"),
    (re.compile(r"now\s+(?:active|enabled|working|configured)", re.I), "now_active"),
    (
        re.compile(r"(?:setting|configuration).*(?:updated|changed|corrected)", re.I),
        "setting_updated",
    ),
    (
        re.compile(r"(?:airplane|flight)\s+mode.*(?:off|disabled|turned)", re.I),
        "airplane_off",
    ),
]

BANKING_RESOLUTION_PATTERNS = [
    (
        re.compile(r"transaction.*(?:posted|completed|processed|successful)", re.I),
        "transaction_posted",
    ),
    (re.compile(r"fee.*(?:reversed|waived|refunded|removed)", re.I), "fee_reversed"),
    (
        re.compile(r"account.*(?:unlocked|reactivated|unfrozen|opened|created)", re.I),
        "account_action",
    ),
    (
        re.compile(r"dispute.*(?:filed|submitted|opened|initiated)", re.I),
        "dispute_filed",
    ),
    (
        re.compile(r"card.*(?:reissued|replaced|activated|unblocked|ordered)", re.I),
        "card_action",
    ),
    (
        re.compile(r"transfer.*(?:completed|processed|successful|initiated)", re.I),
        "transfer_completed",
    ),
    (re.compile(r"balance.*(?:updated|corrected|adjusted)", re.I), "balance_updated"),
    (
        re.compile(r"payment.*(?:processed|posted|completed|successful)", re.I),
        "payment_processed",
    ),
    (
        re.compile(r"(?:issue|problem|concern).*(?:resolved|fixed|addressed)", re.I),
        "issue_resolved",
    ),
    (
        re.compile(
            r"successfully\s+(?:reversed|processed|updated|submitted|filed|ordered|created|opened)",
            re.I,
        ),
        "success_action",
    ),
    (
        re.compile(
            r"has\s+been\s+(?:filed|submitted|ordered|processed|reversed|waived|updated|created|opened|activated|approved)",
            re.I,
        ),
        "has_been_action",
    ),
    (
        re.compile(
            r"(?:replacement|new)\s+(?:card|account).*(?:ordered|created|issued|activated)",
            re.I,
        ),
        "replacement_ordered",
    ),
    (re.compile(r"referral\s+(?:link|code|program)", re.I), "referral_info"),
    (
        re.compile(r"credit\s+limit.*(?:increased|approved|updated)", re.I),
        "credit_limit",
    ),
    (re.compile(r"loan.*(?:approved|processed|submitted)", re.I), "loan_action"),
    (
        re.compile(r"request.*(?:submitted|filed|processed|approved)", re.I),
        "request_submitted",
    ),
    (
        re.compile(
            r"here\s+(?:is|are)\s+(?:the|your)\s+(?:details|information|summary)", re.I
        ),
        "details_provided",
    ),
    (re.compile(r"(?:Dispute|Request|Application)\s+ID", re.I), "id_provided"),
]

POSITIVE_CLOSING_PATTERNS = [
    re.compile(r"(?:is\s+there\s+)?anything\s+else\s+I\s+can\s+help", re.I),
    re.compile(r"glad\s+(?:I\s+could|to)\s+help", re.I),
    re.compile(r"happy\s+to\s+assist", re.I),
    re.compile(r"(?:that|this)\s+should\s+(?:resolve|fix|take\s+care)", re.I),
    re.compile(r"you\'?re?\s+(?:all\s+)?set", re.I),
]

TOOL_RESULT_SUCCESS_PATTERNS = [
    re.compile(r'"status"\s*:\s*"(?:success|completed|active)"', re.I),
    re.compile(r'"result"\s*:\s*"(?:excellent|strong|good|pass)"', re.I),
    re.compile(r"successfully", re.I),
    re.compile(r"verification.*successful", re.I),
]


class ResolutionScreener(RuleScreener):
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

        if not agent_message:
            return ScreenResult(score=0.0)

        patterns = (
            BANKING_RESOLUTION_PATTERNS
            if domain == "banking"
            else TELECOM_RESOLUTION_PATTERNS
        )

        resolution_hits = []
        for pat, name in patterns:
            if pat.search(agent_message):
                resolution_hits.append(name)
                evidence.append(f"resolution signal: {name}")

        tool_success = False
        for msg in reversed(conversation_history[-4:]):
            if msg.get("role") == "tool":
                content = msg.get("content") or ""
                for pat in TOOL_RESULT_SUCCESS_PATTERNS:
                    if pat.search(content):
                        tool_success = True
                        evidence.append("tool result indicates success")
                        break
                if tool_success:
                    break

        if resolution_hits and tool_success:
            score = 0.9
            matched.append("resolution_signal_with_tool_confirmation")
        elif resolution_hits:
            score = 0.7
            matched.append("resolution_signal_text_only")
        elif tool_success and any(
            p.search(agent_message) for p in POSITIVE_CLOSING_PATTERNS
        ):
            score = 0.5
            matched.append("tool_success_with_positive_closing")
        elif any(p.search(agent_message) for p in POSITIVE_CLOSING_PATTERNS):
            turn_idx = kwargs.get("turn_idx", 0)
            n_user_turns = kwargs.get("n_user_turns", 1)
            if n_user_turns > 0 and turn_idx / n_user_turns > 0.7:
                score = 0.4
                matched.append("positive_closing_late_conversation")
                evidence.append(
                    f"turn {turn_idx}/{n_user_turns} (late in conversation)"
                )
            else:
                score = 0.2
                matched.append("positive_closing_early")
        else:
            return ScreenResult(score=0.0)

        return ScreenResult(score=score, evidence=evidence, matched_rules=matched)
