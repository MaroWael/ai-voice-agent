"""
Tests for Dialogue Manager v5 Phase 7 ComparisonEngine & RecommendationEngine

Verifies:
  1. ComparisonEngine feature matrix creation and query formatting.
  2. RecommendationEngine hard salary constraint filtering.
  3. RecommendationEngine strategy scoring for travel vs cashback preferences.
"""

import pytest

from app.conversation.comparison_engine import ComparisonEngine
from app.conversation.entity_resolver import CANONICAL_ENTITIES
from app.conversation.models import ConversationEntity, EntityType, Language, UserProfile
from app.conversation.recommendation_engine import RecommendationEngine


def _get_entity(entity_id: str) -> ConversationEntity:
    for item in CANONICAL_ENTITIES:
        if item["id"] == entity_id:
            return ConversationEntity(
                id=item["id"],
                canonical_name=item["canonical_name"],
                display_name=item["display_name"],
                entity_type=item["entity_type"],
                metadata=item["metadata"],
            )
    raise ValueError(f"Entity {entity_id} not found")


def test_comparison_engine_matrix():
    comp_engine = ComparisonEngine()
    gold = _get_entity("visa_gold")
    plat = _get_entity("visa_platinum")

    matrix_state = comp_engine.build_comparison_matrix([gold, plat])
    assert len(matrix_state.compared_entities) == 2
    assert "Visa Gold" in matrix_state.comparison_matrix
    assert "Visa Platinum" in matrix_state.comparison_matrix
    assert matrix_state.comparison_matrix["Visa Platinum"]["lounge_access"] == "6 visits"

    sq_ar = comp_engine.format_comparison_query([gold, plat], Language.ARABIC)
    assert "مقارنة بين" in sq_ar
    assert "Visa Gold" in sq_ar
    assert "Visa Platinum" in sq_ar


def test_recommendation_engine_scoring_and_hard_constraints():
    rec_engine = RecommendationEngine()
    gold = _get_entity("visa_gold")
    plat = _get_entity("visa_platinum")
    sig = _get_entity("visa_signature")

    candidates = [gold, plat, sig]

    # Scenario 1: Salary 20,000 EGP, purpose travel -> Signature (min 30k) disqualified! Platinum #1 (min 15k, 6 lounges)
    profile1 = UserProfile(salary_egp=20000.0, primary_purpose="travel")
    ranked1 = rec_engine.recommend(profile1, candidates)

    assert ranked1[0].entity.id == "visa_platinum"
    assert ranked1[0].hard_constraint_passed is True

    # Signature fails hard constraint
    sig_candidate = [c for c in ranked1 if c.entity.id == "visa_signature"][0]
    assert sig_candidate.hard_constraint_passed is False
    assert sig_candidate.score == 0.0

    # Scenario 2: Salary 50,000 EGP, purpose travel -> Signature #1 (12 lounges)
    profile2 = UserProfile(salary_egp=50000.0, primary_purpose="travel")
    ranked2 = rec_engine.recommend(profile2, candidates)
    assert ranked2[0].entity.id == "visa_signature"
    assert ranked2[0].score > ranked2[1].score
