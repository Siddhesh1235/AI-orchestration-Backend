"""
Duplicate & Repeat Complaint Service for PCMC Sarathi AI.
Fulfills User Requirement:
If an issue is already registered or was previously resolved and re-reported,
do NOT generate duplicate work orders; simply increment the repeat_count in the DB.
"""

import math
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database.models import Complaint, ComplaintStatus

logger = logging.getLogger("pcms.duplicate_service")

# Approximate 100-meter threshold in GPS coordinate degrees (~0.0009 deg)
PROXIMITY_THRESHOLD_DEG = 0.0010


class DuplicateService:
    def find_matching_complaint(
        self,
        db: Session,
        category: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        citizen_phone: Optional[str] = None,
        ward_number: Optional[int] = None
    ) -> Optional[Complaint]:
        """
        Detects if this issue is already registered based on:
        1. Spatial Proximity: Same category + within 100 meters within last 7 days.
        2. Citizen Matching: Same phone + same category + same ward within last 7 days.
        """
        now = datetime.now(timezone.utc)
        recent_cutoff = now - timedelta(days=7)

        # Query active or recently resolved complaints (not closed) of the same category
        query = db.query(Complaint).filter(
            Complaint.detected_category == category,
            Complaint.created_at >= recent_cutoff,
            Complaint.is_fraud == False,
            Complaint.status != ComplaintStatus.CLOSED
        )

        candidates = query.order_by(desc(Complaint.created_at)).all()

        for cand in candidates:
            # Case 1: Same citizen reporting identical category in same ward
            if citizen_phone and cand.citizen_phone == citizen_phone:
                if ward_number and cand.ward_number == ward_number:
                    logger.info(f"[DuplicateService] Matched existing ticket {cand.ticket_id} by citizen phone & ward.")
                    return cand

            # Case 2: Spatial GPS Proximity (within ~100m)
            if (
                latitude is not None and longitude is not None and
                cand.latitude is not None and cand.longitude is not None
            ):
                dist = math.sqrt((latitude - cand.latitude) ** 2 + (longitude - cand.longitude) ** 2)
                if dist <= PROXIMITY_THRESHOLD_DEG:
                    logger.info(f"[DuplicateService] Matched existing ticket {cand.ticket_id} by GPS proximity (dist={dist:.5f}).")
                    return cand

        return None

    def handle_repeat_complaint(
        self,
        db: Session,
        existing: Complaint,
        citizen_phone: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Increments repeat_count in the database instead of creating a duplicate ticket.
        If the issue was RESOLVED, increments reopen_count and resets status to IN_PROGRESS.
        """
        existing.repeat_count = (existing.repeat_count or 1) + 1
        now = datetime.now(timezone.utc)

        was_resolved = existing.status == ComplaintStatus.RESOLVED

        if was_resolved:
            existing.reopen_count = (existing.reopen_count or 0) + 1
            existing.status = ComplaintStatus.IN_PROGRESS
            existing.officer_remarks = f"नागरिकाने समस्या पूर्ववत असल्याचे नोंदवले (तक्रारदार संख्या: {existing.repeat_count})"
            logger.info(f"[DuplicateService] Re-opened resolved ticket {existing.ticket_id} (Count: {existing.repeat_count})")
        else:
            logger.info(f"[DuplicateService] Incremented repeat count on ticket {existing.ticket_id} -> {existing.repeat_count}")

        existing.updated_at = now
        db.commit()
        db.refresh(existing)

        return {
            "ticket_id": existing.ticket_id,
            "is_duplicate": True,
            "repeat_count": existing.repeat_count,
            "reopen_count": existing.reopen_count,
            "status": existing.status.value,
            "ward": existing.ward_number,
            "assigned_department": existing.assigned_department,
            "assigned_worker_name": existing.assigned_worker_name,
            "assigned_worker_contact": existing.assigned_worker_contact,
            "priority": existing.priority.value if hasattr(existing.priority, "value") else str(existing.priority),
            "sla_hours": existing.sla_hours,
            "sla_deadline": existing.sla_deadline,
            "message": (
                f"सदर समस्या आधीच नोंदवली गेली आहे (तिकीट क्र. {existing.ticket_id}). "
                f"आपला रिपोर्ट नोंदवून तक्रारदार संख्या (Count: {existing.repeat_count}) वाढवण्यात आली आहे."
            )
        }


duplicate_service = DuplicateService()
