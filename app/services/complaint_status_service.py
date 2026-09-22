"""
Deterministic Complaint Status Tracking and Audit Service.
Enforces strict valid status transition lifecycle rules, records immutable audit trails,
and coordinates non-blocking citizen notifications.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Set

from sqlalchemy.orm import Session
from app.database.models import Complaint, ComplaintStatus, ComplaintAuditHistory
from app.services.notification_service import notification_service

logger = logging.getLogger("pcms.status_service")


class InvalidStatusTransitionError(ValueError):
    """Raised when an illegal status lifecycle transition is attempted."""
    def __init__(self, old_status: Any, new_status: Any, message: Optional[str] = None):
        self.old_status = old_status
        self.new_status = new_status
        msg = message or f"Invalid status transition from '{old_status}' to '{new_status}'."
        super().__init__(msg)


VALID_TRANSITIONS: Dict[ComplaintStatus, Set[ComplaintStatus]] = {
    ComplaintStatus.REGISTERED: {
        ComplaintStatus.ASSIGNED,
        ComplaintStatus.IN_PROGRESS,
        ComplaintStatus.RESOLVED,
        ComplaintStatus.REVIEW_REQUIRED,
        ComplaintStatus.CANCELLED,
    },
    ComplaintStatus.ASSIGNED: {
        ComplaintStatus.IN_PROGRESS,
        ComplaintStatus.RESOLVED,
        ComplaintStatus.REVIEW_REQUIRED,
        ComplaintStatus.CANCELLED,
    },
    ComplaintStatus.IN_PROGRESS: {
        ComplaintStatus.RESOLVED,
        ComplaintStatus.CITIZEN_CONFIRMATION,
        ComplaintStatus.REVIEW_REQUIRED,
        ComplaintStatus.CANCELLED,
    },
    ComplaintStatus.RESOLVED: {
        ComplaintStatus.CITIZEN_CONFIRMATION,
        ComplaintStatus.CLOSED,
        ComplaintStatus.REOPENED,
    },
    ComplaintStatus.CITIZEN_CONFIRMATION: {
        ComplaintStatus.CLOSED,
        ComplaintStatus.REOPENED,
    },
    ComplaintStatus.REOPENED: {
        ComplaintStatus.ASSIGNED,
        ComplaintStatus.IN_PROGRESS,
        ComplaintStatus.RESOLVED,
        ComplaintStatus.CANCELLED,
    },
    ComplaintStatus.REVIEW_REQUIRED: {
        ComplaintStatus.ASSIGNED,
        ComplaintStatus.IN_PROGRESS,
        ComplaintStatus.CLOSED,
        ComplaintStatus.CANCELLED,
    },
    ComplaintStatus.CLOSED: set(),  # Terminal state: No direct transitions allowed
    ComplaintStatus.CANCELLED: set(),  # Terminal state: Cancelled complaints cannot be transitioned
}


class ComplaintStatusService:
    """
    Manages deterministic complaint status transitions and audit history.
    """

    def validate_transition(self, current_status: ComplaintStatus, new_status: ComplaintStatus) -> bool:
        """
        Validates if transition from current_status to new_status is permitted.
        """
        if current_status == new_status:
            return True

        allowed = VALID_TRANSITIONS.get(current_status, set())
        return new_status in allowed

    def record_initial_registration(
        self,
        db: Session,
        complaint: Complaint,
        changed_by: str = "SYSTEM: Citizen Registration",
        notes: Optional[str] = None
    ) -> ComplaintAuditHistory:
        """
        Logs the initial creation of a complaint into the immutable audit history.
        """
        audit_entry = ComplaintAuditHistory(
            complaint_id=complaint.id,
            ticket_id=complaint.ticket_id,
            old_status=None,
            new_status=ComplaintStatus.REGISTERED.value,
            changed_by=changed_by,
            reason=notes or f"Complaint registered via {complaint.citizen_phone or 'Citizen'}",
            created_at=datetime.now(timezone.utc)
        )
        db.add(audit_entry)
        db.commit()
        db.refresh(audit_entry)
        return audit_entry

    def transition_status(
        self,
        db: Session,
        complaint: Complaint,
        new_status: ComplaintStatus,
        changed_by: str,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        notify: bool = True
    ) -> Complaint:
        """
        Transitions complaint to new_status with validation, audit record creation,
        and notification dispatch.
        """
        current_status = complaint.status

        # If already at destination status and not a repeat reopen, nothing to do
        if current_status == new_status:
            logger.info(f"[StatusService] Ticket {complaint.ticket_id} is already in status {new_status}")
            return complaint

        if not self.validate_transition(current_status, new_status):
            msg = (
                f"Invalid status transition for ticket {complaint.ticket_id}: "
                f"cannot transition from '{current_status.value if hasattr(current_status, 'value') else current_status}' "
                f"to '{new_status.value if hasattr(new_status, 'value') else new_status}'. "
                f"Allowed destinations: {[s.value for s in VALID_TRANSITIONS.get(current_status, set())]}."
            )
            logger.warning(f"[StatusService] {msg}")
            raise InvalidStatusTransitionError(current_status, new_status, message=msg)

        # Update complaint properties based on target status
        complaint.status = new_status
        complaint.updated_at = datetime.now(timezone.utc)

        if new_status == ComplaintStatus.REOPENED:
            complaint.reopen_count = (complaint.reopen_count or 0) + 1
            complaint.reopen_reason = reason
            complaint.confirmed_resolved = False

        elif new_status == ComplaintStatus.CLOSED:
            complaint.confirmed_resolved = True

        # Create immutable audit record
        meta_str = json.dumps(metadata) if metadata else None
        audit_entry = ComplaintAuditHistory(
            complaint_id=complaint.id,
            ticket_id=complaint.ticket_id,
            old_status=current_status.value if hasattr(current_status, "value") else str(current_status),
            new_status=new_status.value if hasattr(new_status, "value") else str(new_status),
            changed_by=changed_by,
            reason=reason,
            extra_metadata=meta_str,
            created_at=datetime.now(timezone.utc)
        )
        db.add(audit_entry)
        db.commit()
        db.refresh(complaint)

        # Non-blocking notification
        if notify:
            extra_context = {
                "remarks": reason,
                "reopen_reason": reason,
                **(metadata or {})
            }
            notification_service.notify_status_change(complaint, extra_context=extra_context)

        return complaint

    def get_audit_history(self, db: Session, ticket_id: str) -> List[ComplaintAuditHistory]:
        """
        Retrieves complete chronological audit trail for a ticket.
        """
        return (
            db.query(ComplaintAuditHistory)
            .filter(ComplaintAuditHistory.ticket_id == ticket_id)
            .order_by(ComplaintAuditHistory.created_at.asc(), ComplaintAuditHistory.id.asc())
            .all()
        )


complaint_status_service = ComplaintStatusService()
