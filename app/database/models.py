"""
SQLAlchemy ORM Models for PCMC Sarathi AI Grievance Redressal System.
Corresponds to Section 3.4.2 of PCMC Sarathi AI Specification.
"""

import enum
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Enum, Boolean
)
from app.database.session import Base


class ComplaintStatus(str, enum.Enum):
    REGISTERED = "REGISTERED"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CITIZEN_CONFIRMATION = "CITIZEN_CONFIRMATION"
    CLOSED = "CLOSED"
    REOPENED = "REOPENED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    CANCELLED = "CANCELLED"


class PriorityLevel(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class EscalationLevel(str, enum.Enum):
    LEVEL_1_WORKER = "LEVEL_1_WORKER"
    LEVEL_2_SUPERVISOR = "LEVEL_2_SUPERVISOR"
    LEVEL_3_HOD = "LEVEL_3_HOD"


class Complaint(Base):
    __tablename__ = "complaints"

    # Primary Identifiers
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ticket_id = Column(String(64), unique=True, index=True, nullable=False)

    # Citizen & Complaint Details
    citizen_phone = Column(String(20), nullable=True, default="Anonymous")
    description = Column(Text, nullable=False)

    # Geospatial Details
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    ward_number = Column(Integer, nullable=True)

    # Media
    photo_path = Column(String(255), nullable=True)
    video_path = Column(String(255), nullable=True)

    # AI Detection & Classification
    detected_category = Column(String(64), nullable=False)
    confidence_score = Column(Float, nullable=False, default=0.0)
    is_simulated = Column(Boolean, default=False)

    # Priority, Severity & Emergency
    priority = Column(Enum(PriorityLevel), default=PriorityLevel.MEDIUM, nullable=False)
    severity = Column(String(32), default="MEDIUM", nullable=True)
    is_emergency = Column(Boolean, default=False, nullable=False)
    emergency_level = Column(String(32), default="NONE", nullable=True)

    # Verification & Fraud Risk
    verification_status = Column(String(32), default="APPROVED", nullable=True)
    fraud_score = Column(Float, default=0.0, nullable=True)
    evidence_valid = Column(Boolean, default=True, nullable=True)

    # Municipal Department & SLA
    assigned_department = Column(String(64), nullable=False)
    sla_hours = Column(Integer, nullable=False, default=24)
    sla_deadline = Column(DateTime, nullable=True)

    # 3-Level Escalation Hierarchy
    escalation_level = Column(Enum(EscalationLevel), default=EscalationLevel.LEVEL_1_WORKER, nullable=False)
    assigned_worker_name = Column(String(128), nullable=True)
    assigned_worker_contact = Column(String(32), nullable=True)
    supervisor_name = Column(String(128), nullable=True)
    supervisor_contact = Column(String(32), nullable=True)
    hod_name = Column(String(128), nullable=True)
    hod_contact = Column(String(32), nullable=True)
    escalated_at = Column(DateTime, nullable=True)
    escalation_reason = Column(Text, nullable=True)

    # Fraud & Duplicate Prevention
    is_fraud = Column(Boolean, default=False, nullable=False)
    fraud_reason = Column(String(255), nullable=True)
    repeat_count = Column(Integer, default=1, nullable=False)
    reopen_count = Column(Integer, default=0, nullable=False)
    is_duplicate = Column(Boolean, default=False, nullable=False)
    parent_ticket_id = Column(String(64), nullable=True)

    # Lifecycle Status
    status = Column(Enum(ComplaintStatus), default=ComplaintStatus.REGISTERED, nullable=False)

    # Resolution Details (By Municipal Officer)
    resolution_photo_path = Column(String(255), nullable=True)
    resolved_by = Column(String(128), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    officer_contact = Column(String(32), nullable=True)
    officer_remarks = Column(Text, nullable=True)

    # Citizen Feedback & Closure Confirmation
    rating = Column(Integer, nullable=True)  # 1 to 5 stars
    feedback_comments = Column(Text, nullable=True)
    confirmed_resolved = Column(Boolean, default=False)
    reopen_reason = Column(Text, nullable=True)

    # Audit Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Complaint ticket_id={self.ticket_id} status={self.status} priority={self.priority} escalation={self.escalation_level}>"


class ComplaintAuditHistory(Base):
    __tablename__ = "complaint_audit_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    complaint_id = Column(Integer, nullable=False, index=True)
    ticket_id = Column(String(64), nullable=False, index=True)
    old_status = Column(String(64), nullable=True)
    new_status = Column(String(64), nullable=False)
    changed_by = Column(String(128), nullable=True)
    reason = Column(Text, nullable=True)
    extra_metadata = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def __repr__(self):
        return f"<ComplaintAuditHistory ticket={self.ticket_id} {self.old_status}->{self.new_status}>"
