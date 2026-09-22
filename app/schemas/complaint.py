"""
Pydantic Schemas for Complaint Request/Response Payloads.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class ComplaintRegisterResponse(BaseModel):
    ticket_id: str
    detected_category: str
    category_name_mr: Optional[str] = None
    confidence: float
    is_simulated: bool = False
    priority: str = "MEDIUM"
    severity: Optional[str] = "MEDIUM"
    is_emergency: Optional[bool] = False
    emergency_level: Optional[str] = "NONE"
    verification_status: Optional[str] = "APPROVED"
    fraud_score: Optional[float] = 0.0
    evidence_valid: Optional[bool] = True
    ward: Optional[int] = None
    assigned_department: str
    department_name_mr: Optional[str] = None
    sla_hours: int
    sla_deadline: Optional[datetime] = None
    escalation_level: str = "LEVEL_1_WORKER"
    assigned_worker_name: Optional[str] = None
    assigned_worker_contact: Optional[str] = None
    supervisor_name: Optional[str] = None
    hod_name: Optional[str] = None
    status: str
    photo_path: Optional[str] = None
    video_path: Optional[str] = None
    video_details: Optional[dict] = None
    is_duplicate: bool = False
    repeat_count: int = 1
    reopen_count: int = 0
    reopen_reason: Optional[str] = None
    is_fraud: bool = False
    fraud_reason: Optional[str] = None
    moderation_status: Optional[str] = "PASSED"
    moderation_reason: Optional[str] = None
    message: str


class TimelineStep(BaseModel):
    step: str
    title_mr: str
    title_en: str
    timestamp: Optional[datetime] = None
    completed: bool = False
    current: bool = False


class ComplaintStatusResponse(BaseModel):
    ticket_id: str
    status: str
    priority: str = "MEDIUM"
    severity: Optional[str] = "MEDIUM"
    is_emergency: Optional[bool] = False
    emergency_level: Optional[str] = "NONE"
    verification_status: Optional[str] = "APPROVED"
    fraud_score: Optional[float] = 0.0
    evidence_valid: Optional[bool] = True
    detected_category: str
    category_name_mr: Optional[str] = None
    assigned_department: str
    department_name_mr: Optional[str] = None
    ward_number: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    description: str
    photo_path: Optional[str] = None
    video_path: Optional[str] = None
    created_at: datetime
    sla_hours: int
    sla_deadline: Optional[datetime] = None

    # Fraud & Duplicate Prevention
    is_duplicate: bool = False
    repeat_count: int = 1
    reopen_count: int = 0
    reopen_reason: Optional[str] = None
    is_fraud: bool = False
    fraud_reason: Optional[str] = None
    moderation_status: Optional[str] = "PASSED"
    moderation_reason: Optional[str] = None

    # 3-Level Escalation Hierarchy
    escalation_level: str = "LEVEL_1_WORKER"
    assigned_worker_name: Optional[str] = None
    assigned_worker_contact: Optional[str] = None
    supervisor_name: Optional[str] = None
    supervisor_contact: Optional[str] = None
    hod_name: Optional[str] = None
    hod_contact: Optional[str] = None
    escalated_at: Optional[datetime] = None
    escalation_reason: Optional[str] = None
    escalation_chain: Optional[dict] = None
    
    # Resolution Information
    resolved_by: Optional[str] = None
    officer_contact: Optional[str] = None
    officer_remarks: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_photo_path: Optional[str] = None

    # Feedback Information
    rating: Optional[int] = None
    feedback_comments: Optional[str] = None
    confirmed_resolved: bool = False

    # Interactive Visual Timeline
    timeline: List[TimelineStep] = []


class ComplaintUpdateStatusRequest(BaseModel):
    status: str = Field(..., description="IN_PROGRESS or RESOLVED")
    officer_name: Optional[str] = None
    officer_contact: Optional[str] = None
    remarks: Optional[str] = None


class ComplaintEscalateRequest(BaseModel):
    reason: Optional[str] = Field(None, description="Reason for escalating complaint")


class ComplaintEscalateResponse(BaseModel):
    ticket_id: str
    escalated: bool
    from_level: str
    to_level: str
    current_officer: Optional[str] = None
    current_contact: Optional[str] = None
    reason: Optional[str] = None
    message: str


class ComplaintFeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Star rating from 1 to 5")
    comments: Optional[str] = None
    confirmed_resolved: bool = True


class FeedbackResponse(BaseModel):
    ticket_id: str
    status: str
    rating: int
    comments: Optional[str] = None
    message: str
