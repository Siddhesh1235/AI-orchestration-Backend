"""
Schemas for Fraud & Authenticity Detection results.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class FraudEvaluationResult(BaseModel):
    is_fraud: bool
    fraud_score: float = Field(ge=0.0, le=1.0, description="Normalized fraud risk score between 0.0 and 1.0")
    status: str = "APPROVED"  # "APPROVED" | "REVIEW_REQUIRED" | "REJECTED"
    reason: Optional[str] = None
    category: str = "AUTHENTIC"
    risk_signals: List[str] = Field(default_factory=list)
