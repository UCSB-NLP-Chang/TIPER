"""Rule screener for jargon-hit skill."""

import re

from .base import RuleScreener, ScreenResult

JARGON_LEXICONS: dict[str, dict[str, list[str]]] = {
    "telecom": {
        "tech-basic": [
            "APN",
            "VPN protocol",
            "IMEI",
            "LTE band",
            "network mode",
            "baseband",
            "ICCID",
            "SIM toolkit",
            "carrier aggregation",
        ],
        "tech-illiterate": [
            "APN",
            "VPN",
            "network mode",
            "IMEI",
            "LTE",
            "proxy",
            "DNS",
            "bandwidth",
            "latency",
            "throughput",
            "SSID",
            "IP address",
            "subnet",
            "firmware",
            "baseband",
            "ICCID",
            "SIM toolkit",
            "carrier aggregation",
            "QoS",
            "packet loss",
        ],
    },
    "banking": {
        "banking-basic": [
            "ACH",
            "wire transfer routing number",
            "BIN",
            "SWIFT code",
            "MCC",
            "EFT batch",
            "chargeback",
            "interchange fee",
            "NACHA",
            "ISO 20022",
            "clearing house",
        ],
        "banking-illiterate": [
            "ACH",
            "wire",
            "routing number",
            "APR",
            "overdraft fee",
            "grace period",
            "statement cycle",
            "MCC",
            "chargeback",
            "interchange",
            "NACHA",
            "ISO 20022",
            "clearing house",
            "SWIFT",
            "BIN",
            "EFT",
            "yield",
            "amortization",
        ],
    },
}

EXPERT_CAPABILITIES = {
    "tech-savvy",
    "tech-expert",
    "banking-expert",
    "banking-savvy",
}

EXPLANATION_PATTERNS = [
    re.compile(r"\(.*?{term}.*?\)", re.I),
    re.compile(r"{term}\s*[-–—]\s*\w", re.I),
    re.compile(r"{term},?\s+which\s+(is|means|refers)", re.I),
    re.compile(r"{term},?\s+also\s+known\s+as", re.I),
    re.compile(r"that\s+is,?\s+{term}", re.I),
    re.compile(r"{term}\s*\(", re.I),
]


def _term_is_explained(agent_msg: str, term: str) -> bool:
    """Check if the agent explains the term in the same message."""
    for pat_template in EXPLANATION_PATTERNS:
        pat = re.compile(pat_template.pattern.replace("{term}", re.escape(term)), re.I)
        if pat.search(agent_msg):
            return True
    return False


def _find_jargon_in_text(text: str, terms: list[str]) -> list[str]:
    """Find which jargon terms appear in the text."""
    hits = []
    for term in terms:
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.I)
        if pattern.search(text):
            hits.append(term)
    return hits


class JargonHitScreener(RuleScreener):
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

        if capability in EXPERT_CAPABILITIES:
            evidence.append(
                f"capability={capability} is expert-level, skip jargon check"
            )
            return ScreenResult(
                score=0.0, evidence=evidence, matched_rules=["expert_skip"]
            )

        lexicon = JARGON_LEXICONS.get(domain, {}).get(capability, [])
        if not lexicon:
            all_terms = []
            for domain_lexicons in JARGON_LEXICONS.values():
                for terms in domain_lexicons.values():
                    all_terms.extend(terms)
            lexicon = list(set(all_terms))

        hits = _find_jargon_in_text(agent_message, lexicon)
        if not hits:
            return ScreenResult(score=0.0)

        matched.append("jargon_lexicon_hit")
        matched.append("capability_not_expert")

        unexplained = []
        explained = []
        for term in hits:
            if _term_is_explained(agent_message, term):
                explained.append(term)
            else:
                unexplained.append(term)

        if unexplained:
            score = 0.9
            evidence.append(f"unexplained jargon: {unexplained}")
            matched.append("jargon_unexplained")
        elif explained:
            score = 0.6
            evidence.append(f"jargon found but explained: {explained}")
            matched.append("jargon_explained")
        else:
            score = 0.4

        evidence.append(f"capability: {capability}")
        evidence.append(f"domain: {domain}")

        return ScreenResult(score=score, evidence=evidence, matched_rules=matched)
