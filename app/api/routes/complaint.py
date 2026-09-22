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
from sqlalchemy import func
from app.config.category_registry import get_all_categories

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
from app.schemas.feedback import (
    ComplaintResolveRequest,
    CitizenConfirmResolutionRequest,
    ComplaintHistoryResponse,
    ComplaintHistoryItem
)
from app.orchestrator.orchestrator import orchestrator
from app.services.notification_service import notification_service
from app.services.ward_service import ward_service
from app.services.complaint_status_service import complaint_status_service, InvalidStatusTransitionError
from app.services.feedback_service import feedback_service
from app.services.conversational_service import conversational_service
from app.services.voice.stt_service import local_stt_service
from app.services.voice.tts_service import local_tts_service
from app.services.bhashini_stt_service import bhashini_stt_service
from app.services.bhashini_tts_service import bhashini_tts_service
from app.services.video_understanding_service import video_understanding_service
from app.config.settings import settings
import uuid

router = APIRouter(prefix="/complaints", tags=["Grievance Redressal"])

ALLOWED_VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/3gpp",
    "video/3gp",
    "video/webm",
    "video/quicktime",
    "video/x-msvideo"
}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".3gp", ".3gpp", ".webm", ".mov", ".avi"}


async def _save_and_validate_video(video: UploadFile) -> str:
    """Validates video MIME type, size limit, and persists with UUID filename."""
    if not video or not video.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="कोणतीही व्हिडिओ फाईल प्राप्त झाली नाही. कृपया वैध व्हिडिओ अपलोड करा. (No video file received.)"
        )

    ext = Path(video.filename).suffix.lower()
    content_type = (video.content_type or "").lower()

    if content_type not in ALLOWED_VIDEO_MIME_TYPES and ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"अवैध व्हिडिओ फॉरमॅट: '{content_type or ext}'. केवळ MP4, WebM, 3GP व्हिडिओ फॉरमॅट स्वीकारले जातात. (Supported: video/mp4, video/3gpp, video/webm)"
        )

    video_bytes = await video.read()
    max_bytes = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024
    if len(video_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"व्हिडिओचा आकार {settings.MAX_VIDEO_SIZE_MB}MB पेक्षा जास्त आहे. (Video file exceeds {settings.MAX_VIDEO_SIZE_MB}MB limit.)"
        )

    if len(video_bytes) < 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="व्हिडिओ फाईल रिकामी किंवा खराब आहे. (Video file is empty or corrupted.)"
        )

    safe_name = "".join(c for c in Path(video.filename).name if c.isalnum() or c in (".", "_", "-"))
    if not safe_name:
        safe_name = f"video{ext or '.mp4'}"
    unique_filename = f"VIDEO_{uuid.uuid4().hex}_{safe_name}"
    target_path = Path(settings.UPLOAD_DIR) / unique_filename

    with open(target_path, "wb") as f:
        f.write(video_bytes)

    return str(target_path)


@router.get("/categories")
async def get_complaint_categories(db: Session = Depends(get_db)):
    """
    Returns all 12 dataset categories with localized names, department, SLA,
    and real-time complaint count from the database.
    """
    counts = dict(
        db.query(Complaint.detected_category, func.count(Complaint.id))
        .group_by(Complaint.detected_category)
        .all()
    )
    all_cats = get_all_categories()
    result = []
    for cat in all_cats:
        cat_key = cat["key"]
        cnt = counts.get(cat_key, 0)
        if cnt == 0 and cat.get("model_class"):
            cnt = counts.get(cat["model_class"], 0)
        result.append({
            "id": cat.get("id"),
            "key": cat_key,
            "model_class": cat.get("model_class"),
            "name_en": cat.get("name_en"),
            "name_mr": cat.get("name_mr"),
            "name_hi": cat.get("name_hi"),
            "department_code": cat.get("department_code"),
            "department_name": cat.get("department_name"),
            "department_name_mr": cat.get("department_name_mr"),
            "sla_hours": cat.get("sla_hours"),
            "icon": cat.get("icon"),
            "complaint_count": cnt,
            "required_fields": cat.get("required_fields", ["location", "description"])
        })
    return {
        "total_categories": len(result),
        "categories": result
    }


@router.post("/chat")
@router.post("/chat/complaint")
async def chat_with_bot(
    message: Optional[str] = Form(None, description="User chat text"),
    category: Optional[str] = Form(None, description="Selected category key"),
    latitude: Optional[float] = Form(None, description="GPS Latitude"),
    longitude: Optional[float] = Form(None, description="GPS Longitude"),
    citizen_phone: Optional[str] = Form("9876543210", description="Citizen Phone"),
    photo: Optional[UploadFile] = File(None, description="Evidence image"),
    video: Optional[UploadFile] = File(None, description="Evidence video (.mp4, .webm, .3gp)"),
    confirm_register: bool = Form(False, description="Confirm grievance registration"),
    action: Optional[str] = Form(None, description="Action code: chat, select_category, register"),
    ward_number: Optional[int] = Form(None, description="Selected PCMC Ward (1-32)"),
    voice: Optional[UploadFile] = File(None, description="Spoken voice audio (Bhashini STT)"),
    voice_base64: Optional[str] = Form(None, description="Spoken voice base64 (Bhashini STT)"),
    voice_reply: bool = Form(False, description="Whether to include spoken TTS audio version of the reply"),
    language: str = Form("mr", description="Language: mr, en, hi"),
    session_id: Optional[str] = Form(None, description="Client conversation session ID"),
    db: Session = Depends(get_db)
):
    """
    Interactive Multilingual (Marathi & English) Chatbot Endpoint:
    - Transcribes voice queries via Digital India Bhashini STT.
    - Synthesizes spoken voice responses via Digital India Bhashini TTS when voice_reply=True.
    - Processes video uploads (1-fps OpenCV frame sampling, best.pt classification, audio extraction).
    - Answers greetings without creating complaints.
    - Matches user language (English -> English, Marathi -> Marathi).
    - Acknowledges category clicks (e.g. Streetlight) without auto-submitting.
    - Confirms registration before saving ticket.
    - Returns single assigned worker attribution.
    """
    photo_filename = None
    photo_bytes = None

    video_result = None
    # 1. Video Ingestion Pipeline if video is attached in chat
    if video and video.filename:
        try:
            v_saved_path = await _save_and_validate_video(video)
            v_res = await video_understanding_service.process_video(v_saved_path, language=language)
            video_result = v_res
            if v_res.transcript:
                message = (message + " " + v_res.transcript).strip() if message else v_res.transcript
            if v_res.evidence_frame_path and os.path.exists(v_res.evidence_frame_path):
                photo_filename = Path(v_res.evidence_frame_path).name
                with open(v_res.evidence_frame_path, "rb") as f:
                    photo_bytes = f.read()
            if not category and v_res.category and v_res.category != "other":
                category = v_res.category
        except Exception as v_err:
            import logging
            logging.getLogger("pcms.complaint").warning(f"[ChatVideo] Error processing video: {v_err}")

    # 2. Local Whisper STT Voice Processing if voice audio is provided
    transcript = None
    if voice and voice.filename:
        v_bytes = await voice.read()
        ext = voice.filename.split(".")[-1].lower() if "." in voice.filename else "webm"
        stt_res = await local_stt_service.transcribe_audio(v_bytes, language=language, audio_format=ext)
        if stt_res.get("text"):
            transcript = stt_res["text"]
            message = (message + " " + transcript).strip() if message else transcript
    elif voice_base64:
        stt_res = await local_stt_service.transcribe_base64(voice_base64, language=language)
        if stt_res.get("text"):
            transcript = stt_res["text"]
            message = (message + " " + transcript).strip() if message else transcript

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
        video_result=video_result,
        confirm_register=confirm_register,
        action=action,
        ward_number=ward_number,
        session_id=session_id
    )

    if transcript:
        result["transcript"] = transcript

    # 3. Local MMS-TTS Voice Synthesis if voice_reply is requested
    if voice_reply and result and result.get("reply"):
        try:
            target_lang = result.get("language") or language or "mr"
            tts_res = await local_tts_service.synthesize_speech(
                text=result["reply"],
                language=target_lang
            )
            if tts_res.get("success") and tts_res.get("audio_base64"):
                result["audio_base64"] = tts_res["audio_base64"]
                result["audio_format"] = tts_res.get("audio_format", "wav")
                result["audio_available"] = tts_res.get("audio_available", True)
                result["voice_reply"] = True
            else:
                result["audio_base64"] = None
                result["audio_format"] = None
                result["audio_available"] = False
                result["voice_reply"] = False
        except Exception as tts_err:
            import logging
            logging.getLogger("pcms.complaint").warning(f"[VoiceReply] TTS synthesis error: {tts_err}")
            result["audio_base64"] = None
            result["audio_format"] = None
            result["audio_available"] = False
            result["voice_reply"] = False

    return result


@router.post("/register", response_model=ComplaintRegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_complaint(
    description: Optional[str] = Form(None, description="तक्रारीचे वर्णन (Marathi/English) - ऐच्छिक"),
    category: Optional[str] = Form(None, description="तक्रार वर्ग (Category Key)"),
    latitude: Optional[float] = Form(None, description="GPS Latitude"),
    longitude: Optional[float] = Form(None, description="GPS Longitude"),
    citizen_phone: Optional[str] = Form("9876543210", description="नागरिकाचा मोबाईल क्रमांक"),
    photo: Optional[UploadFile] = File(None, description="समस्येचा फोटो"),
    video: Optional[UploadFile] = File(None, description="समस्येचा व्हिडिओ (.mp4, .webm, .3gp)"),
    priority: Optional[str] = Form(None, description="तक्रार प्राधान्य (LOW / MEDIUM / HIGH)"),
    ward_number: Optional[int] = Form(None, description="निवडलेला PCMC प्रभाग (1-32)"),
    language: str = Form("mr", description="भाषा कोड (mr, hi, en)"),
    db: Session = Depends(get_db)
):
    """
    3.4.2.1: Register Complaint via Text, Location, and Photo/Video.
    Description is optional; if omitted, a default grievance summary is used.
    """
    # If video is provided, route through Video Ingestion Pipeline
    if video and video.filename and (not photo or not photo.filename):
        return await register_video_complaint(
            video=video,
            description=description,
            latitude=latitude,
            longitude=longitude,
            citizen_phone=citizen_phone,
            priority=priority,
            ward_number=ward_number,
            language=language,
            db=db
        )

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
        category=category,
        latitude=latitude,
        longitude=longitude,
        citizen_phone=citizen_phone,
        photo_filename=photo_filename,
        photo_bytes=photo_bytes,
        priority=priority,
        ward_number=ward_number
    )

    return result


@router.post("/register-video", response_model=ComplaintRegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_video_complaint(
    video: UploadFile = File(..., description="तक्रारीचा व्हिडिओ (.mp4, .webm, .3gp)"),
    description: Optional[str] = Form(None, description="तक्रारीचे वर्णन (ऐच्छिक - व्हिडिओतील ऑडिओ आपोआप ओळखला जातो)"),
    latitude: Optional[float] = Form(None, description="GPS Latitude"),
    longitude: Optional[float] = Form(None, description="GPS Longitude"),
    citizen_phone: Optional[str] = Form("9876543210", description="नागरिकाचा मोबाईल क्रमांक"),
    priority: Optional[str] = Form(None, description="तक्रार प्राधान्य (LOW / MEDIUM / HIGH)"),
    ward_number: Optional[int] = Form(None, description="निवडलेला PCMC प्रभाग (1-32)"),
    language: str = Form("mr", description="व्हॉईस/ऑडिओ भाषा (mr, hi, en)"),
    db: Session = Depends(get_db)
):
    """
    Video Grievance Ingestion Endpoint:
    1. Validates format (MP4, WebM, 3GP) and size cap (MAX_VIDEO_SIZE_MB).
    2. Extracts 1 frame per second (capped at MAX_FRAMES_PER_VIDEO).
    3. Filters blurry & non-civic frames via existing image_understanding_service.
    4. Classifies surviving frames via trained YOLO11 (best.pt).
    5. Aggregates via (frequency × confidence) voting and saves top evidence frame.
    6. Extracts & transcribes citizen speech from video via Bhashini ASR.
    7. Creates ticket through the standard municipal orchestrator pipeline.
    """
    video_path = await _save_and_validate_video(video)

    # Process video through video understanding pipeline
    video_res = await video_understanding_service.process_video(
        video_path=video_path,
        language=language
    )

    if video_res.duration_sec > settings.MAX_VIDEO_DURATION_SEC:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"व्हिडिओचा कालावधी {settings.MAX_VIDEO_DURATION_SEC} सेकंदांपेक्षा जास्त आहे ({video_res.duration_sec}s). कृपया लहान व्हिडिओ अपलोड करा. (Video duration exceeds {settings.MAX_VIDEO_DURATION_SEC}s limit.)"
        )

    # Synthesize clean complaint description (combining user text + speech transcript)
    clean_desc = (description or "").strip()
    if clean_desc and video_res.transcript:
        merged_desc = f"{clean_desc} (व्हिडिओ ऑडिओ: {video_res.transcript})"
    elif video_res.transcript:
        merged_desc = f"{video_res.transcript} (व्हिडिओ नोंदणी)"
    elif clean_desc:
        merged_desc = clean_desc
    else:
        merged_desc = f"{video_res.category} समस्या तक्रार (व्हिडिओ नोंदणी)"

    result = orchestrator.process_registration(
        db=db,
        description=merged_desc,
        category=video_res.category,
        latitude=latitude,
        longitude=longitude,
        citizen_phone=citizen_phone,
        priority=priority,
        ward_number=ward_number,
        video_path=video_path,
        existing_photo_path=video_res.evidence_frame_path
    )

    result["video_path"] = video_path
    result["video_details"] = video_res.model_dump()
    return result


@router.get("/{ticket_id}", response_model=ComplaintStatusResponse)
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
    status_val = complaint.status.value if hasattr(complaint.status, "value") else str(complaint.status)
    if status_val == "CITIZEN_CONFIRMATION":
        eff_status = "RESOLVED"
    elif status_val == "REOPENED":
        eff_status = "IN_PROGRESS"
    else:
        eff_status = status_val
    current_idx = status_order.index(eff_status) if eff_status in status_order else 0

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
        reopen_reason=getattr(complaint, "reopen_reason", None),
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
@router.post("/{ticket_id}/status", response_model=ComplaintStatusResponse)
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
    Enforces deterministic state transitions and immutable audit logging.
    """
    complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id.strip()).first()
    if not complaint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")

    status_upper = new_status.strip().upper()
    if status_upper not in ComplaintStatus.__members__:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid status: {new_status}")

    target_status = ComplaintStatus[status_upper]
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

    try:
        complaint_status_service.transition_status(
            db=db,
            complaint=complaint,
            new_status=target_status,
            changed_by=f"OFFICER: {officer_name or 'Field Officer'}",
            reason=remarks,
            metadata={
                "officer_name": officer_name,
                "officer_contact": officer_contact,
                "photo": complaint.resolution_photo_path
            },
            notify=True
        )
    except InvalidStatusTransitionError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

    return get_complaint_status(ticket_id=ticket_id, db=db)


@router.post("/{ticket_id}/resolve", response_model=ComplaintStatusResponse)
def resolve_complaint_endpoint(
    ticket_id: str,
    payload: Optional[ComplaintResolveRequest] = None,
    db: Session = Depends(get_db)
):
    """
    Field Officer endpoint: Marks grievance as RESOLVED with remarks and proof.
    """
    complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id.strip()).first()
    if not complaint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")

    officer_name = payload.officer_name if payload else None
    officer_contact = payload.officer_contact if payload else None
    remarks = payload.remarks if payload else None
    photo = payload.resolution_photo_path if payload else None

    try:
        feedback_service.resolve_complaint(
            db=db,
            ticket_id=ticket_id,
            officer_name=officer_name,
            officer_contact=officer_contact,
            remarks=remarks,
            resolution_photo_path=photo
        )
    except InvalidStatusTransitionError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

    return get_complaint_status(ticket_id=ticket_id, db=db)


@router.post("/{ticket_id}/confirm-resolution")
def confirm_resolution_endpoint(
    ticket_id: str,
    payload: CitizenConfirmResolutionRequest,
    db: Session = Depends(get_db)
):
    """
    Citizen Closure Confirmation:
    - If confirmed (Yes) -> transitions to CLOSED, optional 1-5 star rating.
    - If rejected (No) -> transitions to REOPENED with citizen reason.
    """
    complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id.strip()).first()
    if not complaint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")

    try:
        result = feedback_service.confirm_resolution(
            db=db,
            ticket_id=ticket_id,
            confirmed=payload.confirmed,
            reason=payload.reason,
            rating=payload.rating,
            comments=payload.comments
        )
        return result
    except InvalidStatusTransitionError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


@router.post("/{ticket_id}/reopen", response_model=ComplaintStatusResponse)
def reopen_complaint_endpoint(
    ticket_id: str,
    reason: Optional[str] = Form("Citizen indicated grievance still persists"),
    db: Session = Depends(get_db)
):
    """
    Citizen endpoint: Reopens an existing ticket without duplicating records.
    """
    complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id.strip()).first()
    if not complaint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")

    try:
        feedback_service.reopen_complaint(
            db=db,
            ticket_id=ticket_id,
            reason=reason or "Problem still persists",
            changed_by="CITIZEN"
        )
    except InvalidStatusTransitionError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

    return get_complaint_status(ticket_id=ticket_id, db=db)


@router.get("/{ticket_id}/history", response_model=ComplaintHistoryResponse)
def get_complaint_audit_history(
    ticket_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieves complete chronological audit history for a grievance ticket.
    """
    complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id.strip()).first()
    if not complaint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")

    history_data = feedback_service.get_complaint_history(db=db, ticket_id=ticket_id)
    return history_data


@router.post("/{ticket_id}/feedback", response_model=FeedbackResponse)
def submit_complaint_feedback(
    ticket_id: str,
    feedback: ComplaintFeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    3.4.2.2: Closure Confirmation & Citizen Feedback.
    Records 1-5 star rating, closes ticket, and records audit trail.
    """
    complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id.strip()).first()
    if not complaint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")

    try:
        complaint = feedback_service.submit_feedback(
            db=db,
            ticket_id=ticket_id,
            rating=feedback.rating,
            comments=feedback.comments,
            confirmed_resolved=feedback.confirmed_resolved
        )
    except InvalidStatusTransitionError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

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
