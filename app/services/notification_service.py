"""
Notification Service for PCMC Sarathi AI (WhatsApp-First Platform).
Provides localized Marathi notifications for each status transition in the grievance lifecycle.
Designed with pluggable WhatsApp Cloud API architecture (mocked for development/testing).
"""

import logging
from typing import Optional, Dict, Any

from app.database.models import Complaint, ComplaintStatus, EscalationLevel

logger = logging.getLogger("pcms.notification")

NOTIFICATION_TEMPLATES = {
    ComplaintStatus.REGISTERED: (
        "🤝 *वॉर्ड मित्र (Ward Mitra)*\n"
        "नमस्कार! आपली तक्रार नोंदवून घेण्यात आली आहे.\n\n"
        "📌 *तक्रार क्र. (Ticket ID):* {ticket_id}\n"
        "⚡ *प्राधान्य (Priority):* {priority}\n"
        "📂 *विषय/प्रकार:* {category}\n"
        "🏢 *संबंधित विभाग:* {department}\n"
        "📍 *प्रभाग/वॉर्ड:* प्रभाग क्र. {ward_number}\n"
        "👷‍♂️ *नेमलेले क्षेत्रीय कामगार (Level 1):* {officer_name} ({officer_contact})\n"
        "⏱️ *निकालाचा अपेक्षित वेळ (SLA):* {sla_hours} तास\n\n"
        "आपण खालील लिंकद्वारे कधीही तक्रारीची स्थिती तपासू शकता:\n"
        "👉 https://wardmitra.ai/track?id={ticket_id}"
    ),
    ComplaintStatus.ASSIGNED: (
        "📢 *वॉर्ड मित्र अपडेट - तक्रार वर्ग करण्यात आली*\n\n"
        "आपली तक्रार क्र. *{ticket_id}* संबंधित क्षेत्रीय अधिकाऱ्याकडे सोपवली आहे.\n"
        "👤 *नेमलेले अधिकारी:* {officer_name}\n"
        "📞 *संपर्क क्रमांक:* {officer_contact}\n"
        "सदर तक्रार सोडवण्यासाठी तात्काळ कार्यवाही सुरू करण्यात येत आहे."
    ),
    ComplaintStatus.IN_PROGRESS: (
        "⚡ *वॉर्ड मित्र अपडेट - काम प्रगतीपथावर*\n\n"
        "आपली तक्रार क्र. *{ticket_id}* वर क्षेत्रीय कर्मचाऱ्यांकडून प्रत्यक्षात काम सुरू आहे.\n"
        "📝 *शेरा:* {officer_remarks}\n"
        "लवकरच काम पूर्ण करून आपल्याला कळवण्यात येईल."
    ),
    ComplaintStatus.RESOLVED: (
        "✅ *वॉर्ड मित्र अपडेट - तक्रार निवारण पूर्ण!*\n\n"
        "आनंदाची बातमी! आपली तक्रार क्र. *{ticket_id}* यशस्वीरीत्या सोडवण्यात आली आहे.\n"
        "👤 *पूर्ण करणारे अधिकारी:* {officer_name}\n"
        "📝 *शेरा:* {officer_remarks}\n\n"
        "कृपया कामाची खात्री करून आपला अभिप्राय (Rating) नोंदवा:\n"
        "👉 https://wardmitra.ai/feedback?id={ticket_id}"
    ),
    ComplaintStatus.CLOSED: (
        "🎉 *वॉर्ड मित्र - तक्रार बंद (Closed)*\n\n"
        "तक्रार क्र. *{ticket_id}* वरील आपल्या मौल्यवान अभिप्रायाबद्दल धन्यवाद!\n"
        "⭐ *आपले रेटिंग:* {rating}/५\n"
        "आपल्या प्रभागाला सुंदर आणि स्वच्छ ठेवण्यासाठी सहकार्य केल्याबद्दल आभार! 🙏"
    )
}


class NotificationService:
    def send_whatsapp(self, phone: str, message: str) -> bool:
        """
        Sends WhatsApp message via Meta Cloud API / BSP.
        Currently operates in mock mode for development, outputting structured logs.
        """
        phone_sanitized = phone or "Citizen"
        logger.info("\n" + "=" * 60)
        logger.info(f"📱 [MOCK WHATSAPP BROADCAST]")
        logger.info(f"Recipient: {phone_sanitized}")
        logger.info(f"Message:\n{message}")
        logger.info("=" * 60 + "\n")
        return True

    def notify_status_change(self, complaint: Complaint, extra_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Formats localized Marathi template and triggers WhatsApp alert.
        """
        extra = extra_context or {}
        template = NOTIFICATION_TEMPLATES.get(complaint.status)
        if not template:
            logger.warning(f"[Notification] No template defined for status: {complaint.status}")
            return False

        message = template.format(
            ticket_id=complaint.ticket_id,
            priority=getattr(complaint.priority, "value", str(complaint.priority)),
            category=complaint.detected_category,
            department=complaint.assigned_department,
            ward_number=complaint.ward_number or "मध्यवर्ती",
            sla_hours=complaint.sla_hours,
            officer_name=extra.get("officer_name") or complaint.resolved_by or complaint.assigned_worker_name or "क्षेत्रीय अधिकारी",
            officer_contact=extra.get("officer_contact") or complaint.officer_contact or complaint.assigned_worker_contact or "020-67333333",
            officer_remarks=extra.get("remarks") or complaint.officer_remarks or "कामावर कार्यवाही करण्यात आली.",
            rating=complaint.rating or 5
        )

        return self.send_whatsapp(complaint.citizen_phone, message)

    def notify_escalation(self, complaint: Complaint, new_level: EscalationLevel, reason: str) -> bool:
        """
        Dispatches Marathi WhatsApp alert when a complaint is escalated:
        Level 1 Worker -> Level 2 Supervisor -> Level 3 HOD.
        """
        if new_level == EscalationLevel.LEVEL_2_SUPERVISOR:
            message = (
                "⚠️ *वॉर्ड मित्र अलर्ट - तक्रार प्रभाग पर्यवेक्षकांकडे (Supervisor) वर्ग केली!*\n\n"
                f"आपली तक्रार क्र. *{complaint.ticket_id}* वर क्षेत्रीय कर्मचाऱ्याकडून वेळेत "
                "कार्यवाही न झाल्याने, ती प्रभाग अधिकाऱ्यांकडे (Supervisor) वर्ग करण्यात आली आहे.\n\n"
                f"👔 *पर्यवेक्षक अधिकारी:* {complaint.supervisor_name}\n"
                f"📞 *संपर्क क्रमांक:* {complaint.supervisor_contact}\n"
                f"📝 *शेरा/कारण:* {reason}\n\n"
                f"👉 https://wardmitra.ai/track?id={complaint.ticket_id}"
            )
        elif new_level == EscalationLevel.LEVEL_3_HOD:
            message = (
                "🚨 *वॉर्ड मित्र गंभीर अलर्ट - तक्रार विभागप्रमुखांकडे (HOD) वर्ग केली!*\n\n"
                f"आपली तक्रार क्र. *{complaint.ticket_id}* वर पर्यवेक्षक पातळीवरही वेळेत "
                "कार्यवाही न झाल्याने, तक्रार थेट महापालिकेच्या विभागप्रमुखांकडे (HOD) वर्ग करण्यात आली आहे.\n\n"
                f"🏛️ *विभागप्रमुख (HOD):* {complaint.hod_name}\n"
                f"📞 *संपर्क क्रमांक:* {complaint.hod_contact}\n"
                f"📝 *शेरा/कारण:* {reason}\n\n"
                f"👉 https://wardmitra.ai/track?id={complaint.ticket_id}"
            )
        else:
            return False

        return self.send_whatsapp(complaint.citizen_phone, message)


notification_service = NotificationService()
