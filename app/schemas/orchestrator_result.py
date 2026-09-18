"""
Schemas for AI Orchestration and Verification Engine outputs.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    category: str
    image_confidence: float
    evidence_valid: bool
    duplicate: bool
    fraud_risk: float
    severity: str
    is_emergency: bool
    priority: str
    verification_status: str  # "APPROVED" | "REVIEW_REQUIRED" | "REJECTED"
    reasons: List[str] = Field(default_factory=list)


class OrchestratorResult(BaseModel):
    ticket_id: str
    detected_category: str
    category_name_mr: Optional[str] = None
    confidence: float
    is_simulated: bool = False
    priority: str = "P3"
    severity: str = "MEDIUM"
    is_emergency: bool = False
    emergency_level: str = "NONE"
    verification_status: str = "APPROVED"
    fraud_score: float = 0.0
    evidence_valid: bool = True
    ward: Optional[int] = None
    assigned_department: str
    department_name_mr: Optional[str] = None
    sla_hours: int = 24
    sla_deadline: Optional[Any] = None
    escalation_level: str = "LEVEL_1_WORKER"
    assigned_worker_name: Optional[str] = None
    assigned_worker_contact: Optional[str] = None
    status: str
    photo_path: Optional[str] = None
    is_duplicate: bool = False
    repeat_count: int = 1
    reopen_count: int = 0
    is_fraud: bool = False
    fraud_reason: Optional[str] = None
    moderation_status: str = "PASSED"
    moderation_reason: Optional[str] = None
    message: str
