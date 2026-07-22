"""
Unknown Answer Detection — Factory

Constructs a fully wired UnknownAnswerDetector from settings.
Thresholds are read from settings; no magic numbers live here or in the detector.
"""

from app.config.settings import settings
from app.unknown_detection.interfaces import UnknownAnswerDetector
from app.unknown_detection.rule_based import RuleBasedUnknownDetector


def build_unknown_detector() -> UnknownAnswerDetector:
    """
    Return a fully wired RuleBasedUnknownDetector.

    Thresholds are sourced exclusively from settings so they can be tuned
    without touching any logic.
    """
    return RuleBasedUnknownDetector(
        min_score=settings.UNKNOWN_DETECTOR_MIN_SCORE,
        min_results=settings.UNKNOWN_DETECTOR_MIN_RESULTS,
        mean_threshold=settings.UNKNOWN_DETECTOR_MEAN_THRESHOLD,
    )
