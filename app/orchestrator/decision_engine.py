"""
Verification & Decision Engine for WardMitra AI.
Implements Box 12 (Verification Engine) & Box 13 (Deterministic Output Validation).
Guarantees that only APPROVED complaints automatically proceed to registration,
while unverified, suspicious, or duplicate complaints are routed to human review.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("pcms.verification_engine")

VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
VALID_PRIORITIES = {"P1", "P2", "P3", "P4"}


class VerificationEngine:
    def verify(
        self,
        category: str,
        image_confidence: float = 0.0,
        evidence_valid: bool = True,
        is_duplicate: bool = False,
        fraud_risk: float = 0.0,
        severity: str = "MEDIUM",
        is_emergency: bool = False,
        priority: str = "P3",
        has_photo: bool = False,
        is_fraud: bool = False,
        department: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Requirements 12 & 13: Final verification step before complaint creation.
        Returns deterministic verification status ('APPROVED' or 'REVIEW_REQUIRED').
        """
        reasons: List[str] = []

        # 1. Deterministic Validation (Requirement 13)
        validated_category = (category or "other").strip()
        validated_confidence = max(0.0, min(1.0, float(image_confidence)))
        validated_fraud_risk = max(0.0, min(1.0, float(fraud_risk)))
        validated_severity = severity.upper() if severity and severity.upper() in VALID_SEVERITIES else "MEDIUM"
        validated_priority = priority.upper() if priority and priority.upper() in VALID_PRIORITIES else "P3"
        validated_emergency = bool(is_emergency)

        # 2. Decision Logic
        verification_status = "APPROVED"

        # Fraud / Spam check
        if is_fraud or validated_fraud_risk >= 0.45:
            verification_status = "REVIEW_REQUIRED"
            reasons.append(f"Elevated fraud/spam risk (score: {validated_fraud_risk:.2f})")

        # Duplicate check
        if is_duplicate:
            verification_status = "REVIEW_REQUIRED"
            reasons.append("Potential duplicate grievance detected")

        # Image evidence check (if photo was uploaded)
        if has_photo and not evidence_valid:
            verification_status = "REVIEW_REQUIRED"
            reasons.append("Uploaded image does not provide valid evidence for category")

        logger.info(
            f"[VerificationEngine] Verification Status: {verification_status} "
            f"for category '{validated_category}', Priority={validated_priority}, FraudRisk={validated_fraud_risk:.2f}"
        )

        return {
            "category": validated_category,
            "image_confidence": validated_confidence,
            "evidence_valid": evidence_valid,
            "duplicate": is_duplicate,
            "fraud_risk": validated_fraud_risk,
            "severity": validated_severity,
            "is_emergency": validated_emergency,
            "priority": validated_priority,
            "verification_status": verification_status,
            "reasons": reasons
        }


verification_engine = VerificationEngine()
