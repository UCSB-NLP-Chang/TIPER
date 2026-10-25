from .base import RuleScreener
from .frustration_escalation import FrustrationEscalationScreener
from .jargon_hit import JargonHitScreener
from .resolution import ResolutionScreener
from .transfer_pushback import TransferPushbackScreener

SCREENERS: dict[str, type[RuleScreener]] = {
    "transfer-pushback": TransferPushbackScreener,
    "jargon-hit": JargonHitScreener,
    "resolution": ResolutionScreener,
    "resolution-banking": ResolutionScreener,
    "frustration-escalation": FrustrationEscalationScreener,
}


def get_screener(skill: str) -> RuleScreener:
    cls = SCREENERS.get(skill)
    if cls is None:
        raise ValueError(f"No screener for skill: {skill}")
    return cls()
