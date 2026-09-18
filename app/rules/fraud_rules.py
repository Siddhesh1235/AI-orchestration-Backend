"""
Multi-Signal Fraud & Risk Scoring Rules for WardMitra AI.
Calculates a normalized fraud score (0.0 to 1.0) and assigns REVIEW_REQUIRED status.
"""

from typing import Dict, Any, List, Optional
from app.config.settings import settings


def compute_fraud_risk_score(
    is_gibberish: bool = False,
    is_spam_word: bool = False,
    is_out_of_bounds: bool = False,
    text_image_inconsistent: bool = False,
    is_irrelevant_evidence: bool = False,
    is_duplicate: bool = False,
    repeat_frequency_high: bool = False
) -> Dict[str, Any]:
    """
    Computes a normalized fraud risk score from multiple signals without
    jumping to conclusions on a single weak signal.
    """
    score = 0.0
    signals: List[str] = []

    if is_gibberish:
        score += 0.65
        signals.append("Gibberish or repetitive text pattern")

    if is_spam_word:
        score += 0.55
        signals.append("Explicit test or spam word detected")

    if is_out_of_bounds:
        score += 0.45
        signals.append("Location coordinates outside PCMC boundary")

    if text_image_inconsistent:
        score += 0.35
        signals.append("Image evidence does not match complaint category")

    if is_irrelevant_evidence:
        score += 0.30
        signals.append("Uploaded evidence is irrelevant to civic problem")

    if is_duplicate:
        score += 0.20
        signals.append("Potential duplicate submission detected")

    if repeat_frequency_high:
        score += 0.30
        signals.append("Suspiciously high submission frequency")

    normalized_score = min(1.0, round(score, 2))

    if normalized_score >= 0.50:
        status = "REVIEW_REQUIRED"
        is_fraud = True
        reason = f"Suspicious activity detected (Risk Score: {normalized_score}): " + "; ".join(signals)
    elif normalized_score >= 0.30:
        status = "REVIEW_REQUIRED"
        is_fraud = False
        reason = f"Elevated risk signals (Risk Score: {normalized_score}): " + "; ".join(signals)
    else:
        status = "APPROVED"
        is_fraud = False
        reason = "Authentic grievance"

    return {
        "fraud_score": normalized_score,
        "is_fraud": is_fraud,
        "status": status,
        "reason": reason,
        "signals": signals
    }
