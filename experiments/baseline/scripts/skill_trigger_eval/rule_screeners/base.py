"""Base class for rule-based skill screeners."""

from dataclasses import dataclass, field


@dataclass
class ScreenResult:
    score: float  # 0.0-1.0
    evidence: list[str] = field(default_factory=list)
    matched_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "screen_score": round(self.score, 2),
            "screen_evidence": self.evidence,
            "matched_rules": self.matched_rules,
        }


class RuleScreener:
    """Base class. Subclasses implement screen()."""

    def screen(
        self,
        agent_message: str,
        conversation_history: list[dict],
        capability: str,
        emotion: str,
        domain: str,
        **kwargs,
    ) -> ScreenResult:
        raise NotImplementedError
