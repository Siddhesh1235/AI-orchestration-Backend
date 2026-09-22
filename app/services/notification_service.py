"""
Notification Service for PCMC Sarathi AI (WhatsApp-First Platform).
Provides localized Marathi notifications for each status transition in the grievance lifecycle.
Supports pluggable providers (Mock, Meta WhatsApp Cloud API, Fast2SMS, FCM Push)
with resilient fault tolerance and in-memory test inspection.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from app.config.settings import settings
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
        "आनंदाची बातमी! आपली तक्रार क्र. *{ticket_id}* क्षेत्रीय कर्मचाऱ्यांकडून पूर्ण झाल्याचे नोंदवले आहे.\n"
        "👤 *पूर्ण करणारे अधिकारी:* {officer_name}\n"
        "📝 *शेरा:* {officer_remarks}\n\n"
        "कृपया कामाची खात्री करून आपले समाधान नोंदवा:\n"
        "१. होय, समस्या सुटली आहे (Yes, Problem Resolved)\n"
        "२. नाही, समस्या अद्याप बाकी आहे (No, Problem Still Exists)\n\n"
        "👉 https://wardmitra.ai/confirm?id={ticket_id}"
    ),
    ComplaintStatus.CITIZEN_CONFIRMATION: (
        "✅ *वॉर्ड मित्र अपडेट - तक्रार निवारण पूर्ण!*\n\n"
        "आनंदाची बातमी! आपली तक्रार क्र. *{ticket_id}* क्षेत्रीय कर्मचाऱ्यांकडून पूर्ण झाल्याचे नोंदवले आहे.\n"
        "👤 *पूर्ण करणारे अधिकारी:* {officer_name}\n"
        "📝 *शेरा:* {officer_remarks}\n\n"
        "कृपया कामाची खात्री करून आपले समाधान नोंदवा:\n"
        "१. होय, समस्या सुटली आहे (Yes, Problem Resolved)\n"
        "२. नाही, समस्या अद्याप बाकी आहे (No, Problem Still Exists)\n\n"
        "👉 https://wardmitra.ai/confirm?id={ticket_id}"
    ),
    ComplaintStatus.CLOSED: (
        "🎉 *वॉर्ड मित्र - तक्रार यशस्वीरीत्या बंद (Closed)*\n\n"
        "तक्रार क्र. *{ticket_id}* वरील आपल्या मौल्यवान अभिप्रायाबद्दल धन्यवाद!\n"
        "⭐ *आपले रेटिंग:* {rating}/५\n"
        "आपल्या प्रभागाला सुंदर आणि स्वच्छ ठेवण्यासाठी सहकार्य केल्याबद्दल आभार! 🙏"
    ),
    ComplaintStatus.REOPENED: (
        "🔄 *वॉर्ड मित्र अपडेट - तक्रार पुन्हा उघडण्यात आली (Reopened)*\n\n"
        "आपली तक्रार क्र. *{ticket_id}* पुन्हा उघडण्यात आली आहे, कारण समस्येचे पूर्ण निवारण झालेले नाही.\n"
        "📝 *नागरिकाचे कारण:* {reopen_reason}\n"
        "👤 *नेमलेले अधिकारी:* {officer_name} ({officer_contact})\n"
        "आमची टीम तात्काळ यावर पुन्हा वेगाने कारवाई करत आहे."
    ),
    ComplaintStatus.CANCELLED: (
        "🛑 *वॉर्ड मित्र अपडेट - तक्रार रद्द करण्यात आली (Cancelled)*\n\n"
        "आपली तक्रार क्र. *{ticket_id}* आपल्या विनंतीनुसार यशस्वीरीत्या रद्द करण्यात आली आहे.\n"
        "📂 *विषय:* {category}\n"
        "📝 *कारण:* नागरिकांच्या विनंतीनुसार तक्रार मागे घेण्यात आली.\n\n"
        "भविष्यात कोणत्याही नागरी सेवेसाठी मदत लागल्यास वॉर्डमित्र सदैव उपलब्ध आहे. धन्यवाद! 🙏"
    )
}

UPVOTE_TEMPLATE = (
    "ℹ️ *वॉर्ड मित्र अपडेट - तक्रार आधीच नोंदवली आहे*\n\n"
    "आपण नोंदवलेली समस्या आधीच प्रभाग क्र. {ward_number} मध्ये तक्रार क्र. *{ticket_id}* म्हणून दाखल आहे.\n"
    "आपली तक्रार विद्यमान तक्रारीशी जोडण्यात आली असून (एकूण नागरिक: {repeat_count}) तिचा प्राधान्यक्रम वाढवला गेला आहे.\n\n"
    "👉 तक्रार ट्रॅक करा: https://wardmitra.ai/track?id={ticket_id}"
)


class BaseNotificationProvider:
    def send(self, recipient: str, message: str, **kwargs) -> bool:
        raise NotImplementedError


class MockWhatsAppProvider(BaseNotificationProvider):
    def __init__(self):
        self.sent_messages: List[Dict[str, Any]] = []

    def send(self, recipient: str, message: str, **kwargs) -> bool:
        phone_sanitized = recipient or "Citizen"
        record = {
            "channel": "WHATSAPP",
            "recipient": phone_sanitized,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        self.sent_messages.append(record)
        logger.info("\n" + "=" * 60)
        logger.info(f"📱 [MOCK WHATSAPP BROADCAST]")
        logger.info(f"Recipient: {phone_sanitized}")
        logger.info(f"Message:\n{message}")
        logger.info("=" * 60 + "\n")
        return True


class MockSMSProvider(BaseNotificationProvider):
    def __init__(self):
        self.sent_messages: List[Dict[str, Any]] = []

    def send(self, recipient: str, message: str, **kwargs) -> bool:
        phone_sanitized = recipient or "Citizen"
        record = {
            "channel": "SMS",
            "recipient": phone_sanitized,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        self.sent_messages.append(record)
        logger.info(f"✉️ [MOCK SMS] To: {phone_sanitized} | Msg: {message[:60]}...")
        return True


class MockPushProvider(BaseNotificationProvider):
    def __init__(self):
        self.sent_messages: List[Dict[str, Any]] = []

    def send(self, recipient: str, message: str, **kwargs) -> bool:
        record = {
            "channel": "PUSH",
            "recipient": recipient,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        self.sent_messages.append(record)
        logger.info(f"🔔 [MOCK PUSH] To: {recipient} | Msg: {message[:60]}...")
        return True


class NotificationService:
    def __init__(self):
        self.whatsapp_provider: BaseNotificationProvider = MockWhatsAppProvider()
        self.sms_provider: BaseNotificationProvider = MockSMSProvider()
        self.push_provider: BaseNotificationProvider = MockPushProvider()
        self.all_sent_messages: List[Dict[str, Any]] = []

    def send_whatsapp(self, phone: str, message: str, **kwargs) -> bool:
        """
        Dispatches WhatsApp message with strict fault tolerance.
        Failure must NEVER raise an exception to the caller.
        """
        if not getattr(settings, "NOTIFICATION_ENABLED", True) or not getattr(settings, "WHATSAPP_ENABLED", True):
            return False

        try:
            success = self.whatsapp_provider.send(phone, message, **kwargs)
            if success:
                self.all_sent_messages.append({
                    "channel": "WHATSAPP",
                    "recipient": phone,
                    "message": message,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            return success
        except Exception as exc:
            logger.error(f"[NotificationService] WhatsApp dispatch error (non-fatal): {exc}", exc_info=True)
            return False

    def send_sms(self, phone: str, message: str, **kwargs) -> bool:
        """
        Dispatches SMS notification with fault tolerance.
        """
        if not getattr(settings, "NOTIFICATION_ENABLED", True) or not getattr(settings, "SMS_ENABLED", True):
            return False

        try:
            success = self.sms_provider.send(phone, message, **kwargs)
            if success:
                self.all_sent_messages.append({
                    "channel": "SMS",
                    "recipient": phone,
                    "message": message,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            return success
        except Exception as exc:
            logger.error(f"[NotificationService] SMS dispatch error (non-fatal): {exc}", exc_info=True)
            return False

    def send_push(self, recipient: str, message: str, **kwargs) -> bool:
        """
        Dispatches Push notification with fault tolerance.
        """
        if not getattr(settings, "NOTIFICATION_ENABLED", True) or not getattr(settings, "PUSH_ENABLED", True):
            return False

        try:
            success = self.push_provider.send(recipient, message, **kwargs)
            if success:
                self.all_sent_messages.append({
                    "channel": "PUSH",
                    "recipient": recipient,
                    "message": message,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            return success
        except Exception as exc:
            logger.error(f"[NotificationService] Push dispatch error (non-fatal): {exc}", exc_info=True)
            return False

    def notify_status_change(self, complaint: Complaint, extra_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Formats localized Marathi template and triggers WhatsApp alert.
        Preserves backward-compatible signature.
        """
        try:
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
                reopen_reason=getattr(complaint, "reopen_reason", None) or extra.get("reopen_reason") or "समस्येचे समाधान झाले नाही.",
                rating=complaint.rating or 5
            )

            return self.send_whatsapp(complaint.citizen_phone, message, ticket_id=complaint.ticket_id, status=str(complaint.status))
        except Exception as exc:
            logger.error(f"[Notification] notify_status_change error (non-fatal): {exc}", exc_info=True)
            return False

    def notify_escalation(self, complaint: Complaint, new_level: EscalationLevel, reason: str) -> bool:
        """
        Dispatches Marathi WhatsApp alert when a complaint is escalated:
        Level 1 Worker -> Level 2 Supervisor -> Level 3 HOD.
        """
        try:
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

            return self.send_whatsapp(complaint.citizen_phone, message, ticket_id=complaint.ticket_id, escalation_level=str(new_level))
        except Exception as exc:
            logger.error(f"[Notification] notify_escalation error (non-fatal): {exc}", exc_info=True)
            return False

    def notify_duplicate_upvote(self, existing_complaint: Complaint, citizen_phone: Optional[str] = None) -> bool:
        """
        Notifies a citizen that their reported issue was linked to an existing active complaint.
        """
        try:
            phone = citizen_phone or existing_complaint.citizen_phone
            message = UPVOTE_TEMPLATE.format(
                ticket_id=existing_complaint.ticket_id,
                ward_number=existing_complaint.ward_number or "मध्यवर्ती",
                repeat_count=existing_complaint.repeat_count
            )
            return self.send_whatsapp(phone, message, ticket_id=existing_complaint.ticket_id, event="duplicate_upvote")
        except Exception as exc:
            logger.error(f"[Notification] notify_duplicate_upvote error (non-fatal): {exc}", exc_info=True)
            return False

    def get_sent_notifications(self, channel: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Returns list of sent notifications for test inspection and debugging.
        """
        if channel:
            c = channel.upper()
            return [m for m in self.all_sent_messages if m.get("channel") == c]
        return list(self.all_sent_messages)

    def clear_sent_notifications(self):
        """
        Resets test notification history.
        """
        self.all_sent_messages.clear()
        if hasattr(self.whatsapp_provider, "sent_messages"):
            self.whatsapp_provider.sent_messages.clear()
        if hasattr(self.sms_provider, "sent_messages"):
            self.sms_provider.sent_messages.clear()
        if hasattr(self.push_provider, "sent_messages"):
            self.push_provider.sent_messages.clear()


notification_service = NotificationService()
