"""
API Endpoints for PCMC Grievance Redressal (Section 3.4.2).
Provides:
1. POST /api/v1/complaints/register (Multipart Form: Text, GPS, Photo)
2. GET  /api/v1/complaints/{ticket_id}/status (Real-time tracking & history)
3. PATCH /api/v1/complaints/{ticket_id}/status (Field officer update & resolution photo)
4. POST /api/v1/complaints/{ticket_id}/feedback (Citizen closure confirmation & 5-star rating)
"""

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, Form, File, UploadFile, HTTPException, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.database.models import Complaint, ComplaintStatus, PriorityLevel, EscalationLevel
from app.schemas.complaint import (
    ComplaintRegisterResponse,
    ComplaintStatusResponse,
    ComplaintFeedbackRequest,
    FeedbackResponse,
    TimelineStep,
    ComplaintEscalateRequest,
    ComplaintEscalateResponse
)
from app.orchestrator.orchestrator import orchestrator
from app.services.notification_service import notification_service
from app.services.ward_service import ward_service
from app.services.conversational_service import conversational_service
from app.config.settings import settings

router = APIRouter(prefix="/complaints", tags=["Grievance Redressal"])


@router.post("/chat")
async def chat_with_bot(
    message: Optional[str] = Form(None, description="User chat text"),
    category: Optional[str] = Form(None, description="Selected category key"),
    latitude: Optional[float] = Form(None, description="GPS Latitude"),
    longitude: Optional[float] = Form(None, description="GPS Longitude"),
    citizen_phone: Optional[str] = Form("9876543210", description="Citizen Phone"),
    photo: Optional[UploadFile] = File(None, description="Evidence image"),
    confirm_register: bool = Form(False, description="Confirm grievance registration"),
    action: Optional[str] = Form(None, description="Action code: chat, select_category, register"),
    db: Session = Depends(get_db)
):
    """
    Interactive Multilingual (Marathi & English) Chatbot Endpoint:
    - Answers greetings without creating complaints.
    - Matches user language (English -> English, Marathi -> Marathi).
    - Acknowledges category clicks (e.g. Streetlight) without auto-submitting.
    - Confirms registration before saving ticket.
    - Returns single assigned worker attribution.
    """
    photo_filename = None
    photo_bytes = None

    if photo and photo.filename:
        photo_filename = photo.filename
        photo_bytes = await photo.read()

    result = conversational_service.handle_chat(
        db=db,
        message=message,
        category=category,
        latitude=latitude,
        longitude=longitude,
        citizen_phone=citizen_phone,
        photo_filename=photo_filename,
        photo_bytes=photo_bytes,
        confirm_register=confirm_register,
        action=action
    )

    return result


@router.post("/register", response_model=ComplaintRegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_complaint(
    description: Optional[str] = Form(None, description="तक्रारीचे वर्णन (Marathi/English) - ऐच्छिक"),
    latitude: Optional[float] = Form(None, description="GPS Latitude"),
    longitude: Optional[float] = Form(None, description="GPS Longitude"),
    citizen_phone: Optional[str] = Form("9876543210", description="नागरिकाचा मोबाईल क्रमांक"),
    photo: Optional[UploadFile] = File(None, description="समस्येचा फोटो"),
    priority: Optional[str] = Form(None, description="तक्रार प्राधान्य (LOW / MEDIUM / HIGH)"),
    db: Session = Depends(get_db)
):
    """
    3.4.2.1: Register Complaint via Text, Location, and Photo/Video.
    Description is optional; if omitted, a default grievance summary is used.
    """
    photo_filename = None
    photo_bytes = None

    if photo and photo.filename:
        photo_filename = photo.filename
        photo_bytes = await photo.read()

    if description and conversational_service.is_greeting(description) and not photo_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="केवळ अभिवादनावरून तक्रार नोंदवता येत नाही. कृपया समस्येचे वर्णन द्या. (Greetings cannot be registered as complaints. Please describe your civic grievance.)"
        )

    clean_description = (description or "").strip() or "नागरी समस्या तक्रार (Civic Grievance)"

    result = orchestrator.process_registration(
        db=db,
        description=clean_description,
        latitude=latitude,
        longitude=longitude,
        citizen_phone=citizen_phone,
        photo_filename=photo_filename,
        photo_bytes=photo_bytes,
        priority=priority
    )

    return result


@router.get("/{ticket_id}/status", response_model=ComplaintStatusResponse)
def get_complaint_status(ticket_id: str, db: Session = Depends(get_db)):
    """
    3.4.2.2: Real-time Status Updates, 3-Level Escalation Matrix, and Visual Timeline for Citizen.
    """
    complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id.strip()).first()
    if not complaint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"तक्रार क्र. '{ticket_id}' सापडली नाही. कृपया तिकीट नंबर तपासून घ्या."
        )

    # Build interactive visual timeline
    stages = [
        ("REGISTERED", "नोंदणी झाली", "Registered"),
        ("ASSIGNED", "अधिकारी नेमले", "Assigned"),
        ("IN_PROGRESS", "कार्यवाही सुरू", "In Progress"),
        ("RESOLVED", "समस्या सोडवली", "Resolved"),
        ("CLOSED", "तक्रार बंद", "Closed")
    ]
    status_order = [s[0] for s in stages]
    current_idx = status_order.index(complaint.status.value) if complaint.status.value in status_order else 0

    timeline = []
    for idx, (stage_code, mr_title, en_title) in enumerate(stages):
        is_completed = idx <= current_idx
        is_current = idx == current_idx
        step_time = None
        if stage_code == "REGISTERED":
            step_time = complaint.created_at
        elif stage_code == "RESOLVED":
            step_time = complaint.resolved_at
        elif stage_code == "CLOSED" and complaint.status == ComplaintStatus.CLOSED:
            step_time = complaint.updated_at

        timeline.append(TimelineStep(
            step=stage_code,
            title_mr=mr_title,
            title_en=en_title,
            timestamp=step_time,
            completed=is_completed,
            current=is_current
        ))

    # Fetch 3-level escalation chain for current ward & department
    escalation_chain = None
    if complaint.ward_number:
        escalation_chain = ward_service.get_escalation_chain(
            ward_number=complaint.ward_number,
            department=complaint.assigned_department
        )

    return ComplaintStatusResponse(
        ticket_id=complaint.ticket_id,
        status=complaint.status.value,
        priority=complaint.priority.value if hasattr(complaint.priority, "value") else str(complaint.priority),
        detected_category=complaint.detected_category,
        assigned_department=complaint.assigned_department,
        ward_number=complaint.ward_number,
        latitude=complaint.latitude,
        longitude=complaint.longitude,
        description=complaint.description,
        photo_path=complaint.photo_path,
        created_at=complaint.created_at,
        sla_hours=complaint.sla_hours,
        sla_deadline=complaint.sla_deadline,
        escalation_level=complaint.escalation_level.value if hasattr(complaint.escalation_level, "value") else str(complaint.escalation_level),
        assigned_worker_name=complaint.assigned_worker_name,
        assigned_worker_contact=complaint.assigned_worker_contact,
        supervisor_name=complaint.supervisor_name,
        supervisor_contact=complaint.supervisor_contact,
        hod_name=complaint.hod_name,
        hod_contact=complaint.hod_contact,
        escalated_at=complaint.escalated_at,
        escalation_reason=complaint.escalation_reason,
        escalation_chain=escalation_chain,
        is_fraud=complaint.is_fraud or False,
        fraud_reason=complaint.fraud_reason,
        moderation_status="REJECTED" if complaint.is_fraud and "Moderation" in (complaint.fraud_reason or "") else "PASSED",
        moderation_reason=complaint.fraud_reason if "Moderation" in (complaint.fraud_reason or "") else None,
        repeat_count=complaint.repeat_count or 1,
        reopen_count=complaint.reopen_count or 0,
        is_duplicate=complaint.is_duplicate or False,
        resolved_by=complaint.resolved_by,
        officer_contact=complaint.officer_contact,
        officer_remarks=complaint.officer_remarks,
        resolved_at=complaint.resolved_at,
        resolution_photo_path=complaint.resolution_photo_path,
        rating=complaint.rating,
        feedback_comments=complaint.feedback_comments,
        confirmed_resolved=complaint.confirmed_resolved,
        timeline=timeline
    )


@router.post("/{ticket_id}/escalate", response_model=ComplaintEscalateResponse)
def escalate_complaint_endpoint(
    ticket_id: str,
    payload: Optional[ComplaintEscalateRequest] = None,
    db: Session = Depends(get_db)
):
    """
    3-Level Escalation Endpoint:
    Escalates complaint: Level 1 (Ward Worker) -> Level 2 (Supervisor) -> Level 3 (HOD).
    Triggered when worker or supervisor fails to take action within SLA threshold.
    """
    complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id.strip()).first()
    if not complaint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"तक्रार क्र. '{ticket_id}' सापडली नाही."
        )

    reason = payload.reason if payload else None
    result = ward_service.escalate_complaint(db, complaint, reason=reason)
    return result


@router.patch("/{ticket_id}/status", response_model=ComplaintStatusResponse)
async def update_complaint_status(
    ticket_id: str,
    new_status: str = Form(..., description="ASSIGNED, IN_PROGRESS, or RESOLVED"),
    officer_name: Optional[str] = Form("Er. Santosh Patil (Junior Engineer)", description="अधिकाऱ्याचे नाव"),
    officer_contact: Optional[str] = Form("020-67333333", description="संपर्क नंबर"),
    remarks: Optional[str] = Form("कार्यवाही पूर्ण करण्यात आली.", description="कामाचा शेरा"),
    resolution_photo: Optional[UploadFile] = File(None, description="काम पूर्ण झाल्याचा फोटो (पुरावा)"),
    db: Session = Depends(get_db)
):
    """
    Field Officer API: Updates grievance status to IN_PROGRESS or RESOLVED with optional proof photo.
    """
    complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id.strip()).first()
    if not complaint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")

    status_upper = new_status.strip().upper()
    if status_upper not in ComplaintStatus.__members__:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid status: {new_status}")

    complaint.status = ComplaintStatus[status_upper]
    complaint.resolved_by = officer_name
    complaint.officer_contact = officer_contact
    complaint.officer_remarks = remarks

    # Handle resolution photo upload
    if resolution_photo and resolution_photo.filename:
        res_filename = f"RESOLVED_{ticket_id}_{resolution_photo.filename}"
        res_path = Path(settings.UPLOAD_DIR) / res_filename
        content = await resolution_photo.read()
        with open(res_path, "wb") as f:
            f.write(content)
        complaint.resolution_photo_path = str(res_path)

    if status_upper == "RESOLVED":
        complaint.resolved_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(complaint)

    # Dispatch WhatsApp update notification
    notification_service.notify_status_change(complaint, extra_context={
        "officer_name": officer_name,
        "officer_contact": officer_contact,
        "remarks": remarks
    })

    return get_complaint_status(ticket_id=ticket_id, db=db)


@router.post("/{ticket_id}/feedback", response_model=FeedbackResponse)
def submit_complaint_feedback(
    ticket_id: str,
    feedback: ComplaintFeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    3.4.2.2: Closure Confirmation & Citizen Feedback.
    Records 1-5 star rating, closes ticket, and sends thank-you notification.
    """
    complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id.strip()).first()
    if not complaint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")

    complaint.rating = feedback.rating
    complaint.feedback_comments = feedback.comments
    complaint.confirmed_resolved = feedback.confirmed_resolved

    if feedback.confirmed_resolved:
        complaint.status = ComplaintStatus.CLOSED

    db.commit()
    db.refresh(complaint)

    # Dispatch Closed notification
    notification_service.notify_status_change(complaint)

    return FeedbackResponse(
        ticket_id=complaint.ticket_id,
        status=complaint.status.value,
        rating=complaint.rating,
        comments=complaint.feedback_comments,
        message="आपल्या मौल्यवान अभिप्रायाबद्दल मनापासून धन्यवाद! तक्रार बंद (Closed) करण्यात आली आहे."
    )


# =========================================================================
# WARD-WISE SPECIFIC ENDPOINTS (प्रभागनिहाय सेवा)
# =========================================================================

@router.get("/wards/list")
def list_pcmc_wards():
    """Returns list of all 32 PCMC wards with offices and ward officers."""
    from app.services.ward_service import ward_service
    return ward_service.get_all_wards()


@router.get("/wards/{ward_number}")
def get_ward_info(ward_number: int):
    """Returns profile and assigned field engineers for a specific ward."""
    from app.services.ward_service import ward_service
    return ward_service.get_ward_details(ward_number)


@router.get("/ward/{ward_number}")
def get_complaints_by_ward(
    ward_number: int,
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Ward Officer Portal: Returns all grievances registered in a specific ward.
    """
    query = db.query(Complaint).filter(Complaint.ward_number == ward_number)
    if status_filter:
        query = query.filter(Complaint.status == status_filter.upper())
    complaints = query.order_by(Complaint.created_at.desc()).all()

    return [
        {
            "ticket_id": c.ticket_id,
            "status": c.status.value,
            "detected_category": c.detected_category,
            "assigned_department": c.assigned_department,
            "ward_number": c.ward_number,
            "description": c.description,
            "assigned_officer": c.resolved_by,
            "officer_contact": c.officer_contact,
            "sla_hours": c.sla_hours,
            "sla_deadline": c.sla_deadline,
            "photo_path": c.photo_path,
            "created_at": c.created_at
        }
        for c in complaints
    ]


@router.get("/ward/{ward_number}/stats")
def get_ward_statistics(ward_number: int, db: Session = Depends(get_db)):
    """
    Ward Scorecard: Returns metrics (total, pending, in-progress, resolved, closed).
    """
    from app.services.ward_service import ward_service
    return ward_service.get_ward_stats(db, ward_number)


@router.post("/check-escalations")
def check_and_auto_escalate(db: Session = Depends(get_db)):
    """
    Automated SLA Breached Escalation Checker:
    Scans unresolved complaints (REGISTERED, ASSIGNED) that have not been acted upon.
    Auto-escalates:
      - LEVEL_1_WORKER -> LEVEL_2_SUPERVISOR (if past worker SLA threshold)
      - LEVEL_2_SUPERVISOR -> LEVEL_3_HOD (if past supervisor SLA threshold)
    """
    from datetime import datetime, timezone
    from app.agents.severity_agent import severity_agent

    active_complaints = db.query(Complaint).filter(
        Complaint.status.in_([ComplaintStatus.REGISTERED, ComplaintStatus.ASSIGNED]),
        Complaint.escalation_level != EscalationLevel.LEVEL_3_HOD
    ).all()

    now = datetime.now(timezone.utc)
    escalated_list = []

    for comp in active_complaints:
        sched = severity_agent.get_escalation_schedule(comp.priority, comp.sla_hours)
        created_time = comp.created_at
        if created_time.tzinfo is None:
            created_time = created_time.replace(tzinfo=timezone.utc)

        elapsed_hours = (now - created_time).total_seconds() / 3600.0

        should_escalate = False
        reason = ""

        if comp.escalation_level == EscalationLevel.LEVEL_1_WORKER and elapsed_hours >= sched["worker_threshold_hours"]:
            should_escalate = True
            reason = f"क्षेत्रीय कामगाराने विहित मुदतीत ({sched['worker_threshold_hours']} तास) कार्यवाही न केल्याने आपोआप वर्ग झाले."
        elif comp.escalation_level == EscalationLevel.LEVEL_2_SUPERVISOR and elapsed_hours >= (sched["worker_threshold_hours"] + sched["supervisor_threshold_hours"]):
            should_escalate = True
            reason = f"पर्यवेक्षक स्तरावर विहित मुदतीत कार्यवाही न झाल्याने आपोआप HOD कडे वर्ग झाले."

        if should_escalate:
            res = ward_service.escalate_complaint(db, comp, reason=reason)
            escalated_list.append(res)

    return {
        "checked_count": len(active_complaints),
        "escalated_count": len(escalated_list),
        "escalations": escalated_list
    }
