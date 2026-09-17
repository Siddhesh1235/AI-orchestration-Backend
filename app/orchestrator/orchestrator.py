"""
Central Orchestration Engine for PCMC Sarathi AI.
Fulfills BoQ Component B & F: Coordinates Multi-Agent intelligence,
resolves classification priority, maps ward, routes department, and stores grievance.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.config.settings import settings
from app.agents.image_agent import image_agent
from app.agents.nlp_agent import nlp_agent
from app.agents.geo_agent import geo_agent
from app.agents.routing_agent import routing_agent
from app.agents.severity_agent import severity_agent
from app.agents.fraud_agent import fraud_agent
from app.agents.moderation_agent import moderation_agent
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
        Executes end-to-end Grievance Registration Pipeline:
        1. Save photo to storage (if provided)
        2. Fraud / Spam / Gibberish evaluation
        3. Image Classifier inference (YOLO / best.pt or pretrained)
        4. Multilingual NLP analysis of text
        5. Spatial Geo-tagging to PCMC Ward 1-32
        6. Duplicate / Repeat Complaint Check:
           If already registered -> INCREMENT COUNT ONLY (no duplicate ticket)
        7. Priority & Severity Grading (LOW / MEDIUM / HIGH)
        8. Department routing & dynamic SLA calculation
        9. 3-Level Escalation Chain setup (Worker -> Supervisor -> HOD)
        10. Database transaction & Ticket Generation
        11. Trigger Citizen WhatsApp Notification
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

        # Step 2: Fraud & Spam Detection
        fraud_eval = fraud_agent.evaluate_authenticity(
            description=description,
            latitude=latitude,
            longitude=longitude
        )
        is_fraud = fraud_eval["is_fraud"]
        fraud_reason = fraud_eval["reason"] if is_fraud else None

        # Step 2.5: Content Moderation & Profanity Check
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

        # Step 3: Multi-Agent Intelligence Layer
        # Priority: Image Classifier takes precedence if photo is present
        if saved_photo_path and os.path.exists(saved_photo_path):
            img_result = image_agent.classify_image(saved_photo_path)
            final_category = img_result["category"]
            confidence = img_result["confidence"]
            is_simulated = img_result.get("is_pretrained", False)
            nlp_result = nlp_agent.extract_intent_and_category(description)
        else:
            # Fallback to Text NLP classification
            nlp_result = nlp_agent.extract_intent_and_category(description)
            final_category = nlp_result["category"]
            confidence = nlp_result["confidence"]
            is_simulated = False

        # Step 4: Geospatial Ward Mapping
        geo_result = geo_agent.map_coordinates_to_ward(latitude, longitude)
        ward_number = geo_result["ward_number"]

        # Step 5: Duplicate / Repeat Complaint Handling
        # If this issue is already active/reported at this location, simply increment repeat_count!
        if not is_fraud:
            existing_match = duplicate_service.find_matching_complaint(
                db=db,
                category=final_category,
                latitude=latitude,
                longitude=longitude,
                citizen_phone=citizen_phone,
                ward_number=ward_number
            )
            if existing_match:
                dup_result = duplicate_service.handle_repeat_complaint(
                    db=db,
                    existing=existing_match,
                    citizen_phone=citizen_phone
                )
                # Dispatch notification about count increase
                notification_service.notify_status_change(existing_match)

                return {
                    "ticket_id": existing_match.ticket_id,
                    "detected_category": existing_match.detected_category,
                    "category_name_mr": existing_match.detected_category,
                    "confidence": existing_match.confidence_score,
                    "is_simulated": existing_match.is_simulated,
                    "priority": existing_match.priority.value if hasattr(existing_match.priority, "value") else str(existing_match.priority),
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

        # Step 6: Priority & Severity Evaluation (Low / Medium / High)
        priority_eval = severity_agent.evaluate_priority(
            description=description,
            category=final_category,
            requested_priority=priority
        )
        determined_priority = priority_eval["priority"]

        # Step 7: Department & Dynamic SLA Routing
        routing_info = routing_agent.route_complaint(final_category)
        ticket_id = routing_agent.generate_ticket_id()

        schedule = severity_agent.get_escalation_schedule(
            priority=determined_priority,
            base_sla_hours=routing_info["sla_hours"]
        )

        # Step 8: 3-Level Municipal Escalation Chain Mapping
        escalation_chain = ward_service.get_escalation_chain(ward_number, routing_info["department"])
        worker_info = escalation_chain["level_1_worker"]
        supervisor_info = escalation_chain["level_2_supervisor"]
        hod_info = escalation_chain["level_3_hod"]

        # Step 9: Persist to Database (Initially Level 1 Worker)
        complaint = Complaint(
            ticket_id=ticket_id,
            citizen_phone=citizen_phone or "9876543210",
            description=description,
            latitude=latitude,
            longitude=longitude,
            ward_number=ward_number,
            photo_path=saved_photo_path,
            detected_category=final_category,
            confidence_score=confidence,
            is_simulated=is_simulated,
            priority=determined_priority,
            assigned_department=routing_info["department"],
            sla_hours=schedule["sla_hours"],
            sla_deadline=schedule["sla_deadline"],
            escalation_level=EscalationLevel.LEVEL_1_WORKER,
            assigned_worker_name=worker_info["name"],
            assigned_worker_contact=worker_info["contact"],
            supervisor_name=supervisor_info["name"],
            supervisor_contact=supervisor_info["contact"],
            hod_name=hod_info["name"],
            hod_contact=hod_info["contact"],
            resolved_by=worker_info["name"],
            officer_contact=worker_info["contact"],
            is_fraud=is_fraud,
            fraud_reason=fraud_reason,
            repeat_count=1,
            reopen_count=0,
            is_duplicate=False,
            status=ComplaintStatus.REGISTERED
        )
        db.add(complaint)
        db.commit()
        db.refresh(complaint)

        logger.info(
            f"[Orchestrator] Registered {ticket_id} ({determined_priority.value}) for Ward {ward_number} "
            f"-> Level 1 Worker: {worker_info['name']} | Fraud: {is_fraud}"
        )

        # Step 10: Dispatch Citizen Notification (Only if authentic)
        if not is_fraud:
            notification_service.notify_status_change(complaint)

        msg = (
            f"तक्रार यशस्वीरीत्या नोंदवली गेली. तिकीट क्र: {complaint.ticket_id} (प्राधान्य: {complaint.priority.value})"
            if not is_fraud else
            f"⚠️ संशयास्पद/अवैध तक्रार नोंदवली: {fraud_reason}"
        )

        return {
            "ticket_id": complaint.ticket_id,
            "detected_category": complaint.detected_category,
            "category_name_mr": routing_info["category_name_mr"],
            "confidence": complaint.confidence_score,
            "is_simulated": complaint.is_simulated,
            "priority": complaint.priority.value,
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
