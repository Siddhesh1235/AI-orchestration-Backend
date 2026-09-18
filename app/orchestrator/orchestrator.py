"""
Central Orchestration Engine for WardMitra AI / PCMC Sarathi.
Executes the full 15-step municipal grievance pipeline:
Security + Validation -> Preprocessing -> Language -> Intent & Clarification ->
YOLO11 Classification -> Evidence Verification -> Duplicate Detection ->
Fraud Risk -> Severity -> Emergency -> Deterministic Priority (P1-P4) ->
Routing -> Verification Engine -> Human Review OR Auto Creation.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.agents.image_agent import image_agent
from app.agents.nlp_agent import nlp_agent
from app.agents.geo_agent import geo_agent
from app.agents.routing_agent import routing_agent
from app.agents.severity_agent import severity_agent
from app.agents.fraud_agent import fraud_agent
from app.agents.moderation_agent import moderation_agent
from app.orchestrator.decision_engine import verification_engine
from app.services.fraud_service import duplicate_service
from app.services.notification_service import notification_service
from app.services.ward_service import ward_service
from app.database.models import Complaint, ComplaintStatus, EscalationLevel, PriorityLevel

logger = logging.getLogger("pcms.orchestrator")


class GrievanceOrchestrator:
    def process_registration(
        self,
        db: Session,
        description: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        citizen_phone: Optional[str] = None,
        photo_filename: Optional[str] = None,
        photo_bytes: Optional[bytes] = None,
        priority: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end Grievance Registration Pipeline with strict Verification Gate.
        """
        saved_photo_path = None
        is_simulated = False

        # Step 1: Save uploaded photo if present
        if photo_filename and photo_bytes:
            unique_filename = f"{routing_agent.generate_ticket_id()}_{photo_filename}"
            target_path = Path(settings.UPLOAD_DIR) / unique_filename
            with open(target_path, "wb") as f:
                f.write(photo_bytes)
            saved_photo_path = str(target_path)
            logger.info(f"[Orchestrator] Saved grievance evidence image to: {saved_photo_path}")

        # Step 2: Content Moderation & Profanity Check
        moderation_eval = moderation_agent.moderate_complaint(
            description=description,
            photo_path=saved_photo_path
        )
        if not moderation_eval["is_safe"]:
            mod_reasons = " | ".join(moderation_eval["reasons"])
            logger.warning(f"[Orchestrator] Content Moderation violation blocked: {mod_reasons}")
            ticket_id = routing_agent.generate_ticket_id()

            # Record blocked grievance for security audit trail without officer dispatch
            rejected_complaint = Complaint(
                ticket_id=ticket_id,
                citizen_phone=citizen_phone or "Anonymous",
                description=description,
                latitude=latitude,
                longitude=longitude,
                ward_number=None,
                photo_path=saved_photo_path,
                detected_category="rejected_inappropriate_content",
                confidence_score=0.0,
                is_simulated=False,
                priority=PriorityLevel.LOW,
                severity="LOW",
                is_emergency=False,
                emergency_level="NONE",
                verification_status="REJECTED",
                fraud_score=1.0,
                evidence_valid=False,
                assigned_department="DISCIPLINARY_REVIEW",
                sla_hours=0,
                escalation_level=EscalationLevel.LEVEL_1_WORKER,
                is_fraud=True,
                fraud_reason=f"Content Moderation Violation: {mod_reasons}",
                status=ComplaintStatus.CLOSED
            )
            db.add(rejected_complaint)
            db.commit()

            return {
                "ticket_id": ticket_id,
                "detected_category": "Inappropriate / Abusive Content",
                "category_name_mr": "आक्षेपार्ह / अयोग्य मजकूर",
                "confidence": 1.0,
                "is_simulated": False,
                "priority": "LOW",
                "priority_code": "P4",
                "severity": "LOW",
                "is_emergency": False,
                "emergency_level": "NONE",
                "verification_status": "REJECTED",
                "fraud_score": 1.0,
                "evidence_valid": False,
                "ward": None,
                "assigned_department": "Rejected",
                "department_name_mr": "नाकारण्यात आले",
                "sla_hours": 0,
                "sla_deadline": None,
                "escalation_level": "REJECTED",
                "assigned_worker_name": "N/A",
                "assigned_worker_contact": "N/A",
                "supervisor_name": "N/A",
                "hod_name": "N/A",
                "status": "REJECTED",
                "photo_path": saved_photo_path,
                "is_duplicate": False,
                "repeat_count": 0,
                "reopen_count": 0,
                "is_fraud": True,
                "fraud_reason": mod_reasons,
                "moderation_status": "REJECTED",
                "moderation_reason": mod_reasons,
                "message": f"🚫 तक्रार नाकारली: {mod_reasons}"
            }

        # Step 3: NLP Analysis (Language & Intent & Category)
        nlp_result = nlp_agent.extract_intent_and_category(description)
        detected_category = nlp_result.get("category") or "pothole"
        confidence = nlp_result.get("confidence", 0.70) or 0.70
        evidence_valid = True
        img_result = None

        # Step 4: YOLO11 Image Classification & Evidence Verification
        if saved_photo_path and os.path.exists(saved_photo_path):
            img_result = image_agent.classify_image(saved_photo_path)
            is_simulated = img_result.get("is_pretrained", False)

            # Verify evidence relevance to reported problem
            ev_check = image_agent.verify_evidence(
                complaint_category=detected_category,
                image_prediction=img_result,
                image_path=saved_photo_path
            )
            evidence_valid = ev_check["evidence_valid"]

            if evidence_valid and img_result.get("confidence", 0) > confidence:
                confidence = img_result["confidence"]

        # Step 5: Geospatial Ward Mapping
        geo_result = geo_agent.map_coordinates_to_ward(latitude, longitude)
        ward_number = geo_result["ward_number"]

        # Step 6: Duplicate / Repeat Complaint Check
        is_duplicate = False
        existing_match = duplicate_service.find_matching_complaint(
            db=db,
            category=detected_category,
            latitude=latitude,
            longitude=longitude,
            citizen_phone=citizen_phone,
            ward_number=ward_number
        )

        if existing_match:
            is_duplicate = True
            dup_result = duplicate_service.handle_repeat_complaint(
                db=db,
                existing=existing_match,
                citizen_phone=citizen_phone
            )
            notification_service.notify_status_change(existing_match)

            return {
                "ticket_id": existing_match.ticket_id,
                "detected_category": existing_match.detected_category,
                "category_name_mr": existing_match.detected_category,
                "confidence": existing_match.confidence_score,
                "is_simulated": existing_match.is_simulated,
                "priority": existing_match.priority.value if hasattr(existing_match.priority, "value") else str(existing_match.priority),
                "priority_code": "P2" if existing_match.priority == PriorityLevel.HIGH else "P3",
                "severity": existing_match.severity or "MEDIUM",
                "is_emergency": existing_match.is_emergency if hasattr(existing_match, "is_emergency") else False,
                "emergency_level": existing_match.emergency_level if hasattr(existing_match, "emergency_level") else "NONE",
                "verification_status": "REVIEW_REQUIRED",
                "fraud_score": existing_match.fraud_score or 0.0,
                "evidence_valid": existing_match.evidence_valid if hasattr(existing_match, "evidence_valid") else True,
                "ward": existing_match.ward_number,
                "assigned_department": existing_match.assigned_department,
                "department_name_mr": existing_match.assigned_department,
                "sla_hours": existing_match.sla_hours,
                "sla_deadline": existing_match.sla_deadline,
                "escalation_level": existing_match.escalation_level.value if hasattr(existing_match.escalation_level, "value") else str(existing_match.escalation_level),
                "assigned_worker_name": existing_match.assigned_worker_name,
                "assigned_worker_contact": existing_match.assigned_worker_contact,
                "supervisor_name": existing_match.supervisor_name,
                "hod_name": existing_match.hod_name,
                "status": existing_match.status.value,
                "photo_path": existing_match.photo_path,
                "is_duplicate": True,
                "repeat_count": existing_match.repeat_count,
                "reopen_count": existing_match.reopen_count,
                "is_fraud": False,
                "fraud_reason": None,
                "moderation_status": "PASSED",
                "moderation_reason": None,
                "message": dup_result["message"]
            }

        # Step 7: Fraud & Authenticity Risk Scoring
        fraud_eval = fraud_agent.evaluate_authenticity(
            description=description,
            latitude=latitude,
            longitude=longitude,
            text_image_inconsistent=(not evidence_valid if saved_photo_path else False),
            is_irrelevant_evidence=(not evidence_valid if saved_photo_path else False),
            is_duplicate=is_duplicate
        )
        is_fraud = fraud_eval["is_fraud"]
        fraud_reason = fraud_eval["reason"] if is_fraud else None
        fraud_score = fraud_eval["fraud_score"]

        # Step 8: Severity, Emergency, and Priority Calculation
        priority_eval = severity_agent.evaluate_priority(
            description=description,
            category=detected_category,
            requested_priority=priority
        )
        determined_priority = priority_eval["priority"]
        priority_code = priority_eval["priority_code"]
        severity_val = priority_eval["severity"]
        is_emergency = priority_eval["is_emergency"]
        emergency_level = priority_eval["emergency_level"]

        # Step 9: Department & Dynamic SLA Routing
        routing_info = routing_agent.route_complaint(detected_category)
        ticket_id = routing_agent.generate_ticket_id()

        schedule = severity_agent.get_escalation_schedule(
            priority=determined_priority,
            base_sla_hours=routing_info["sla_hours"]
        )

        # Step 10: 3-Level Municipal Escalation Hierarchy Mapping
        escalation_chain = ward_service.get_escalation_chain(ward_number, routing_info["department"])
        worker_info = escalation_chain["level_1_worker"]
        supervisor_info = escalation_chain["level_2_supervisor"]
        hod_info = escalation_chain["level_3_hod"]

        # Step 11: Verification Engine Evaluation
        verification = verification_engine.verify(
            category=detected_category,
            image_confidence=img_result["confidence"] if img_result else confidence,
            evidence_valid=evidence_valid,
            is_duplicate=is_duplicate,
            fraud_risk=fraud_score,
            severity=severity_val,
            is_emergency=is_emergency,
            priority=priority_code,
            has_photo=bool(saved_photo_path),
            is_fraud=is_fraud,
            department=routing_info["department"]
        )
        verification_status = verification["verification_status"]

        # Final Status determination:
        # Only APPROVED complaints become REGISTERED.
        # Suspicious, high fraud, or invalid evidence complaints become REVIEW_REQUIRED.
        if verification_status == "APPROVED" and not is_fraud:
            final_status = ComplaintStatus.REGISTERED
            assigned_worker = worker_info["name"]
            assigned_contact = worker_info["contact"]
        else:
            final_status = ComplaintStatus.REVIEW_REQUIRED
            assigned_worker = "Municipal Verification Desk"
            assigned_contact = "020-67333333"

        # Step 12: Persist to Database
        complaint = Complaint(
            ticket_id=ticket_id,
            citizen_phone=citizen_phone or "9876543210",
            description=description,
            latitude=latitude,
            longitude=longitude,
            ward_number=ward_number,
            photo_path=saved_photo_path,
            detected_category=detected_category,
            confidence_score=confidence,
            is_simulated=is_simulated,
            priority=determined_priority,
            severity=severity_val,
            is_emergency=is_emergency,
            emergency_level=emergency_level,
            verification_status=verification_status,
            fraud_score=fraud_score,
            evidence_valid=evidence_valid,
            assigned_department=routing_info["department"],
            sla_hours=schedule["sla_hours"],
            sla_deadline=schedule["sla_deadline"],
            escalation_level=EscalationLevel.LEVEL_1_WORKER,
            assigned_worker_name=assigned_worker,
            assigned_worker_contact=assigned_contact,
            supervisor_name=supervisor_info["name"],
            supervisor_contact=supervisor_info["contact"],
            hod_name=hod_info["name"],
            hod_contact=hod_info["contact"],
            resolved_by=assigned_worker,
            officer_contact=assigned_contact,
            is_fraud=is_fraud,
            fraud_reason=fraud_reason,
            repeat_count=1,
            reopen_count=0,
            is_duplicate=False,
            status=final_status
        )
        db.add(complaint)
        db.commit()
        db.refresh(complaint)

        logger.info(
            f"[Orchestrator] Grievance {ticket_id} [{verification_status}] Priority={priority_code} "
            f"Severity={severity_val} Emergency={is_emergency} -> Status={final_status.value}"
        )

        # Step 13: Dispatch Citizen Notification (Only for authentic complaints)
        if not is_fraud:
            notification_service.notify_status_change(complaint)

        if final_status == ComplaintStatus.REGISTERED:
            msg = f"तक्रार यशस्वीरीत्या नोंदवली गेली. तिकीट क्र: {complaint.ticket_id} (प्राधान्य: {priority_code})"
        elif is_fraud:
            msg = f"⚠️ संशयास्पद/अवैध तक्रार नोंदवली: {fraud_reason}"
        else:
            msg = f"तक्रार पुनरावलोकनासाठी (Review Required) पाठवण्यात आली आहे. तिकीट क्र: {complaint.ticket_id}"

        return {
            "ticket_id": complaint.ticket_id,
            "detected_category": complaint.detected_category,
            "category_name_mr": routing_info["category_name_mr"],
            "confidence": complaint.confidence_score,
            "is_simulated": complaint.is_simulated,
            "priority": complaint.priority.value,
            "priority_code": priority_code,
            "severity": severity_val,
            "is_emergency": is_emergency,
            "emergency_level": emergency_level,
            "verification_status": verification_status,
            "fraud_score": fraud_score,
            "evidence_valid": evidence_valid,
            "ward": complaint.ward_number,
            "assigned_department": complaint.assigned_department,
            "department_name_mr": routing_info["department_name_mr"],
            "sla_hours": complaint.sla_hours,
            "sla_deadline": complaint.sla_deadline,
            "escalation_level": complaint.escalation_level.value,
            "assigned_worker_name": complaint.assigned_worker_name,
            "assigned_worker_contact": complaint.assigned_worker_contact,
            "supervisor_name": complaint.supervisor_name,
            "hod_name": complaint.hod_name,
            "status": complaint.status.value,
            "photo_path": complaint.photo_path,
            "is_duplicate": False,
            "repeat_count": 1,
            "reopen_count": 0,
            "is_fraud": complaint.is_fraud,
            "fraud_reason": complaint.fraud_reason,
            "moderation_status": "PASSED",
            "moderation_reason": None,
            "message": msg
        }


orchestrator = GrievanceOrchestrator()
