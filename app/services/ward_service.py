"""
Ward Service for PCMC Sarathi AI.
Provides Ward Directory, Ward Officers Mapping, Field Engineers allocation,
and Ward-specific grievance telemetry.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config.settings import settings
from app.database.models import Complaint, ComplaintStatus

logger = logging.getLogger("pcms.ward_service")


class WardService:
    def __init__(self):
        self.ward_config_path = Path(settings.BASE_DIR) / "app" / "config" / "ward_config.yaml"
        self.wards_data: Dict[int, Any] = {}
        self._load_ward_config()

    def _load_ward_config(self):
        if self.ward_config_path.exists():
            try:
                with open(self.ward_config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    self.wards_data = data.get("wards", {})
            except Exception as e:
                logger.error(f"[WardService] Error loading ward_config.yaml: {e}")

    def get_all_wards(self) -> List[Dict[str, Any]]:
        """Returns list of all configured PCMC wards with office details."""
        result = []
        for w_num in range(1, 33):
            info = self.wards_data.get(w_num, {})
            result.append({
                "ward_number": w_num,
                "ward_name_mr": info.get("ward_name_mr", f"प्रभाग क्र. {w_num}"),
                "ward_name_en": info.get("ward_name_en", f"Ward No. {w_num}"),
                "zone": info.get("zone", f"Zone {chr(65 + ((w_num - 1) // 4))}"),
                "office_address": info.get("office_address", f"प्रभाग {w_num} कार्यालय, PCMC"),
                "ward_officer": info.get("ward_officer", "प्रभाग क्षेत्रीय अधिकारी"),
                "ward_officer_contact": info.get("ward_officer_contact", "020-27411000")
            })
        return result

    def get_ward_details(self, ward_number: int) -> Dict[str, Any]:
        """Returns detailed profile of a specific PCMC ward."""
        info = self.wards_data.get(ward_number, {})
        return {
            "ward_number": ward_number,
            "ward_name_mr": info.get("ward_name_mr", f"प्रभाग क्र. {ward_number}"),
            "ward_name_en": info.get("ward_name_en", f"Ward No. {ward_number}"),
            "zone": info.get("zone", f"Zone {chr(65 + ((ward_number - 1) // 4))}"),
            "office_address": info.get("office_address", f"प्रभाग {ward_number} कार्यालय, PCMC"),
            "ward_officer": info.get("ward_officer", "प्रभाग क्षेत्रीय अधिकारी"),
            "ward_officer_contact": info.get("ward_officer_contact", "020-27411000"),
            "field_engineers": info.get("field_engineers", {})
        }

    def get_field_engineer(self, ward_number: int, department: str) -> Dict[str, str]:
        """
        Retrieves the exact Field Engineer / Sanitary Inspector assigned to (Ward, Department).
        """
        ward_info = self.wards_data.get(ward_number, {})
        field_engineers = ward_info.get("field_engineers", {})
        
        # Check specific department engineer
        if department in field_engineers:
            return field_engineers[department]

        # Fallback to ward officer
        return {
            "name": ward_info.get("ward_officer", f"क्षेत्रीय अधिकारी (प्रभाग {ward_number})"),
            "contact": ward_info.get("ward_officer_contact", "020-67333333")
        }

    def get_ward_stats(self, db: Session, ward_number: int) -> Dict[str, Any]:
        """
        Calculates real-time performance scorecard for a specific ward.
        """
        ward_info = self.get_ward_details(ward_number)
        
        total = db.query(func.count(Complaint.id)).filter(Complaint.ward_number == ward_number).scalar() or 0
        registered = db.query(func.count(Complaint.id)).filter(Complaint.ward_number == ward_number, Complaint.status == ComplaintStatus.REGISTERED).scalar() or 0
        assigned = db.query(func.count(Complaint.id)).filter(Complaint.ward_number == ward_number, Complaint.status == ComplaintStatus.ASSIGNED).scalar() or 0
        in_progress = db.query(func.count(Complaint.id)).filter(Complaint.ward_number == ward_number, Complaint.status == ComplaintStatus.IN_PROGRESS).scalar() or 0
        resolved = db.query(func.count(Complaint.id)).filter(Complaint.ward_number == ward_number, Complaint.status == ComplaintStatus.RESOLVED).scalar() or 0
        closed = db.query(func.count(Complaint.id)).filter(Complaint.ward_number == ward_number, Complaint.status == ComplaintStatus.CLOSED).scalar() or 0

        # Department-wise breakdown
        dept_counts = (
            db.query(Complaint.assigned_department, func.count(Complaint.id))
            .filter(Complaint.ward_number == ward_number)
            .group_by(Complaint.assigned_department)
            .all()
        )

        return {
            "ward_number": ward_number,
            "ward_name": ward_info["ward_name_mr"],
            "zone": ward_info["zone"],
            "ward_officer": ward_info["ward_officer"],
            "office_address": ward_info["office_address"],
            "total_complaints": total,
            "pending": registered + assigned,
            "in_progress": in_progress,
            "resolved": resolved,
            "closed": closed,
            "department_breakdown": {dept: count for dept, count in dept_counts}
        }

    def get_department_hod(self, department: str) -> Dict[str, str]:
        """Retrieves Head of Department (HOD) details from model_config.yaml."""
        model_config_path = Path(settings.MODEL_CONFIG_PATH)
        if model_config_path.exists():
            try:
                with open(model_config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    dept_info = cfg.get("departments", {}).get(department, {})
                    return {
                        "name": dept_info.get("head", "Er. Chief Municipal Engineer (HOD)"),
                        "contact": dept_info.get("contact", "020-67333333")
                    }
            except Exception as e:
                logger.warning(f"[WardService] Could not read HOD from model config: {e}")

        return {
            "name": f"HOD ({department})",
            "contact": "020-67333300"
        }

    def get_escalation_chain(self, ward_number: int, department: str) -> Dict[str, Any]:
        """
        Returns full 3-Level Municipal Escalation Chain:
        Level 1: Ward Worker / Field Engineer
        Level 2: Ward Supervisor / Ward Officer
        Level 3: Department Head (HOD)
        """
        ward_info = self.get_ward_details(ward_number)
        worker_info = self.get_field_engineer(ward_number, department)
        hod_info = self.get_department_hod(department)

        return {
            "level_1_worker": {
                "role": "Ward Field Engineer / Sanitary Inspector",
                "role_mr": "क्षेत्रीय अभियंता / प्रभाग कामगार",
                "name": worker_info["name"],
                "contact": worker_info["contact"],
                "ward": ward_number
            },
            "level_2_supervisor": {
                "role": "Ward Officer / Assistant Commissioner",
                "role_mr": "प्रभाग अधिकारी / पर्यवेक्षक",
                "name": ward_info["ward_officer"],
                "contact": ward_info["ward_officer_contact"],
                "ward": ward_number
            },
            "level_3_hod": {
                "role": "Head of Department (HOD)",
                "role_mr": "विभागप्रमुख (HOD)",
                "name": hod_info["name"],
                "contact": hod_info["contact"],
                "department": department
            }
        }

    def escalate_complaint(self, db: Session, complaint: Complaint, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Escalates complaint along the 3-level municipal hierarchy:
        LEVEL_1_WORKER -> LEVEL_2_SUPERVISOR -> LEVEL_3_HOD
        """
        from app.database.models import EscalationLevel
        from datetime import datetime, timezone
        from app.services.notification_service import notification_service

        current_level = complaint.escalation_level
        now = datetime.now(timezone.utc)

        # Backfill chain if missing on legacy records
        if not complaint.supervisor_name or not complaint.hod_name:
            chain = self.get_escalation_chain(complaint.ward_number or 15, complaint.assigned_department)
            complaint.assigned_worker_name = complaint.assigned_worker_name or chain["level_1_worker"]["name"]
            complaint.assigned_worker_contact = complaint.assigned_worker_contact or chain["level_1_worker"]["contact"]
            complaint.supervisor_name = complaint.supervisor_name or chain["level_2_supervisor"]["name"]
            complaint.supervisor_contact = complaint.supervisor_contact or chain["level_2_supervisor"]["contact"]
            complaint.hod_name = complaint.hod_name or chain["level_3_hod"]["name"]
            complaint.hod_contact = complaint.hod_contact or chain["level_3_hod"]["contact"]

        if current_level == EscalationLevel.LEVEL_1_WORKER:
            new_level = EscalationLevel.LEVEL_2_SUPERVISOR
            default_reason = "क्षेत्रीय कामगाराने विहित वेळेत कार्यवाही न केल्याने पर्यवेक्षकांकडे वर्ग केले."
            complaint.escalation_level = new_level
            complaint.escalated_at = now
            complaint.escalation_reason = reason or default_reason
            complaint.resolved_by = complaint.supervisor_name or complaint.resolved_by
            complaint.officer_contact = complaint.supervisor_contact or complaint.officer_contact
            db.commit()
            db.refresh(complaint)

            # Notification
            notification_service.notify_escalation(complaint, new_level, complaint.escalation_reason)

            return {
                "ticket_id": complaint.ticket_id,
                "escalated": True,
                "from_level": "LEVEL_1_WORKER",
                "to_level": "LEVEL_2_SUPERVISOR",
                "current_officer": complaint.resolved_by,
                "current_contact": complaint.officer_contact,
                "reason": complaint.escalation_reason,
                "message": "तक्रार प्रभाग पर्यवेक्षकांकडे (Level 2) यशस्वीरीत्या वर्ग केली."
            }

        elif current_level == EscalationLevel.LEVEL_2_SUPERVISOR:
            new_level = EscalationLevel.LEVEL_3_HOD
            default_reason = "पर्यवेक्षकाने वेळेत कार्यवाही न केल्याने थेट विभागप्रमुखांकडे (HOD) वर्ग केले."
            complaint.escalation_level = new_level
            complaint.escalated_at = now
            complaint.escalation_reason = reason or default_reason
            complaint.resolved_by = complaint.hod_name or complaint.resolved_by
            complaint.officer_contact = complaint.hod_contact or complaint.officer_contact
            db.commit()
            db.refresh(complaint)

            # Notification
            notification_service.notify_escalation(complaint, new_level, complaint.escalation_reason)

            return {
                "ticket_id": complaint.ticket_id,
                "escalated": True,
                "from_level": "LEVEL_2_SUPERVISOR",
                "to_level": "LEVEL_3_HOD",
                "current_officer": complaint.resolved_by,
                "current_contact": complaint.officer_contact,
                "reason": complaint.escalation_reason,
                "message": "तक्रार महापालिका विभागप्रमुखांकडे (Level 3 HOD) यशस्वीरीत्या वर्ग केली."
            }

        else:
            return {
                "ticket_id": complaint.ticket_id,
                "escalated": False,
                "from_level": "LEVEL_3_HOD",
                "to_level": "LEVEL_3_HOD",
                "current_officer": complaint.resolved_by,
                "current_contact": complaint.officer_contact,
                "reason": "तक्रार आधीच सर्वोच्च पातळीवर (Level 3 HOD) प्रलंबित आहे.",
                "message": "तक्रार आधीच सर्वोच्च पातळीवर (HOD) आहे."
            }


ward_service = WardService()
