"""
Recommendation Engine

Evaluates product candidates against customer profile constraints using the Strategy Pattern.
Applies hard salary constraints first, followed by soft constraint purpose scoring.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, List

from app.conversation.models import (
    ConversationEntity,
    EntityType,
    RecommendationCandidate,
    UserProfile,
)

logger = logging.getLogger(__name__)


class BaseScoringStrategy(ABC):
    """Abstract base scoring strategy for recommendation soft constraints."""

    @abstractmethod
    def score(self, card: ConversationEntity, profile: UserProfile) -> float:
        pass


class TravelScoringStrategy(BaseScoringStrategy):
    def score(self, card: ConversationEntity, profile: UserProfile) -> float:
        meta = card.metadata or {}
        score = 1.0
        lounges = meta.get("lounge_access_count", 0)
        if lounges > 0:
            score += lounges * 0.15
        if meta.get("foreign_markup", 0.05) < 0.03:
            score += 0.20
        return score


class CashbackScoringStrategy(BaseScoringStrategy):
    def score(self, card: ConversationEntity, profile: UserProfile) -> float:
        meta = card.metadata or {}
        score = 1.0
        cb_rate = meta.get("cashback_rate", 0.0)
        score += cb_rate * 20.0
        return score


class GeneralScoringStrategy(BaseScoringStrategy):
    def score(self, card: ConversationEntity, profile: UserProfile) -> float:
        meta = card.metadata or {}
        score = 1.0
        min_salary = meta.get("min_salary", 0)
        score += (min_salary / 10000.0) * 0.1
        return score


class RecommendationEngine:
    """
    Ranks product candidates deterministically against user profile slots.
    """

    def __init__(self) -> None:
        self.strategies: Dict[str, BaseScoringStrategy] = {
            "travel": TravelScoringStrategy(),
            "cashback": CashbackScoringStrategy(),
            "general": GeneralScoringStrategy(),
        }

    def recommend(self, profile: UserProfile, candidates: List[ConversationEntity]) -> List[RecommendationCandidate]:
        """
        Ranks *candidates* against *profile*. Hard salary constraint applied first.
        Returns sorted list of RecommendationCandidates.
        """
        salary = profile.salary_egp or 0.0
        purpose = profile.primary_purpose or "general"
        strategy = self.strategies.get(purpose, self.strategies["general"])

        ranked: List[RecommendationCandidate] = []

        for card in candidates:
            meta = card.metadata or {}
            min_salary = meta.get("min_salary", 0)

            # Hard Constraint Filter: Salary Requirement
            if salary > 0 and salary < min_salary:
                ranked.append(
                    RecommendationCandidate(
                        entity=card,
                        score=0.0,
                        hard_constraint_passed=False,
                        matching_features=[],
                        trade_offs=[f"Requires minimum salary of {min_salary:,} EGP (Current: {salary:,.0f} EGP)"],
                    )
                )
                continue

            # Soft Constraint Strategy Scoring
            score_val = strategy.score(card, profile)
            matches = []
            trade_offs = []

            lounges = meta.get("lounge_access_count", 0)
            if lounges > 0:
                matches.append(f"Includes {lounges} complimentary airport lounge visits")
            elif purpose == "travel":
                trade_offs.append("No complimentary lounge access")

            cb_rate = meta.get("cashback_rate", 0.0)
            if cb_rate > 0:
                matches.append(f"Offers {cb_rate*100:.1f}% cashback")

            ranked.append(
                RecommendationCandidate(
                    entity=card,
                    score=round(score_val, 2),
                    hard_constraint_passed=True,
                    matching_features=matches,
                    trade_offs=trade_offs,
                )
            )

        ranked.sort(key=lambda x: x.score, reverse=True)
        logger.info("RecommendationEngine: Scored %d candidates. Top recommendation: %s (score: %.2f)",
                    len(ranked), ranked[0].entity.display_name if ranked else "None", ranked[0].score if ranked else 0.0)
        return ranked
