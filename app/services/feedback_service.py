"""
Complaint Closure, Citizen Feedback, and Reopening Service.
Handles officer resolution submissions, citizen verification confirmations,
1-5 star ratings, and complaint reopenings.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session
from app.database.models import Complaint, ComplaintStatus
from app.services.complaint_status_service import complaint_status_service

logger = logging.getLogger("pcms.feedback_service")


class FeedbackService:
    """
    Coordinates complaint resolution, citizen confirmation, feedback, and reopening.
    """

    def resolve_complaint(
        self,
        db: Session,
        ticket_id: str,
        officer_name: Optional[str] = None,
        officer_contact: Optional[str] = None,
        remarks: Optional[str] = None,
        resolution_photo_path: Optional[str] = None
    ) -> Complaint:
        """
        Marks a complaint as RESOLVED by the field officer, updating resolution metadata
        and transitioning lifecycle state.
        """
        complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id).first()
        if not complaint:
            raise ValueError(f"Complaint with ticket_id '{ticket_id}' not found.")

        complaint.resolved_by = officer_name or complaint.resolved_by or complaint.assigned_worker_name or "Municipal Officer"
        complaint.officer_contact = officer_contact or complaint.officer_contact or complaint.assigned_worker_contact
        complaint.officer_remarks = remarks or complaint.officer_remarks or "कामावर कार्यवाही पूर्ण करण्यात आली."
        complaint.resolution_photo_path = resolution_photo_path or complaint.resolution_photo_path
        complaint.resolved_at = datetime.now(timezone.utc)

        updated = complaint_status_service.transition_status(
            db=db,
            complaint=complaint,
            new_status=ComplaintStatus.RESOLVED,
            changed_by=f"OFFICER: {complaint.resolved_by}",
            reason=remarks or "Municipal officer marked complaint as resolved",
            metadata={
                "resolution_photo": resolution_photo_path,
                "officer_name": complaint.resolved_by,
                "officer_contact": complaint.officer_contact
            },
            notify=True
        )
        return updated

    def confirm_resolution(
        self,
        db: Session,
        ticket_id: str,
        confirmed: bool,
        reason: Optional[str] = None,
        rating: Optional[int] = None,
        comments: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Citizen closure confirmation workflow:
        - If confirmed == True: Closes ticket, saves rating and feedback.
        - If confirmed == False: Reopens ticket, logs reason, keeps history.
        """
        complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id).first()
        if not complaint:
            raise ValueError(f"Complaint with ticket_id '{ticket_id}' not found.")

        if confirmed:
            if rating is not None:
                complaint.rating = rating
            if comments:
                complaint.feedback_comments = comments
            complaint.confirmed_resolved = True

            complaint_status_service.transition_status(
                db=db,
                complaint=complaint,
                new_status=ComplaintStatus.CLOSED,
                changed_by="CITIZEN",
                reason="Citizen confirmed problem resolved",
                metadata={"rating": rating, "comments": comments},
                notify=True
            )

            return {
                "ticket_id": ticket_id,
                "status": ComplaintStatus.CLOSED.value,
                "confirmed": True,
                "rating": complaint.rating,
                "message": "तक्रार यशस्वीरीत्या बंद करण्यात आली आहे. आपल्या सहकार्याबद्दल धन्यवाद!"
            }
        else:
            reopen_reason = reason or "Citizen reported problem still exists"
            complaint_status_service.transition_status(
                db=db,
                complaint=complaint,
                new_status=ComplaintStatus.REOPENED,
                changed_by="CITIZEN",
                reason=reopen_reason,
                metadata={"citizen_reason": reopen_reason},
                notify=True
            )

            return {
                "ticket_id": ticket_id,
                "status": ComplaintStatus.REOPENED.value,
                "confirmed": False,
                "reopen_count": complaint.reopen_count,
                "reopen_reason": reopen_reason,
                "message": "तक्रार पुन्हा उघडण्यात आली आहे. संबंधित क्षेत्रीय अधिकाऱ्यांना तात्काळ पुढील कार्यवाहीसाठी सूचित केले आहे."
            }

    def submit_feedback(
        self,
        db: Session,
        ticket_id: str,
        rating: int,
        comments: Optional[str] = None,
        confirmed_resolved: bool = True
    ) -> Complaint:
        """
        Records 1-5 star citizen rating and closes ticket if not already closed.
        """
        complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id).first()
        if not complaint:
            raise ValueError(f"Complaint with ticket_id '{ticket_id}' not found.")

        complaint.rating = rating
        complaint.feedback_comments = comments
        complaint.confirmed_resolved = confirmed_resolved

        if complaint.status != ComplaintStatus.CLOSED:
            complaint_status_service.transition_status(
                db=db,
                complaint=complaint,
                new_status=ComplaintStatus.CLOSED,
                changed_by="CITIZEN",
                reason=f"Citizen feedback submitted with rating {rating}/5",
                metadata={"rating": rating, "comments": comments},
                notify=True
            )
        else:
            db.commit()
            db.refresh(complaint)

        return complaint

    def reopen_complaint(
        self,
        db: Session,
        ticket_id: str,
        reason: str,
        changed_by: str = "CITIZEN"
    ) -> Complaint:
        """
        Explicitly reopens an existing ticket without duplicating records.
        """
        complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id).first()
        if not complaint:
            raise ValueError(f"Complaint with ticket_id '{ticket_id}' not found.")

        return complaint_status_service.transition_status(
            db=db,
            complaint=complaint,
            new_status=ComplaintStatus.REOPENED,
            changed_by=changed_by,
            reason=reason,
            metadata={"reopen_reason": reason},
            notify=True
        )

    def get_complaint_history(self, db: Session, ticket_id: str) -> Dict[str, Any]:
        """
        Fetches the current complaint state alongside its full immutable audit history.
        """
        complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id).first()
        if not complaint:
            raise ValueError(f"Complaint with ticket_id '{ticket_id}' not found.")

        history_items = complaint_status_service.get_audit_history(db, ticket_id)
        return {
            "ticket_id": ticket_id,
            "current_status": complaint.status.value if hasattr(complaint.status, "value") else str(complaint.status),
            "reopen_count": complaint.reopen_count or 0,
            "reopen_reason": getattr(complaint, "reopen_reason", None),
            "history": history_items
        }


feedback_service = FeedbackService()
