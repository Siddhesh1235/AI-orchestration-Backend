"""
Multilingual (Marathi & English) Conversational Service for PCMC Sarathi / WardMitra AI.
Powered by Local Ollama (llama3.2).
Handles:
1. Natural conversation & smart replies in Marathi or English
2. Greetings without premature complaint registration
3. Category acknowledgment without immediate complaint submission
4. Confirmation flow before raising grievances
5. Clear, single assigned worker attribution
"""

import re
import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.orchestrator.orchestrator import orchestrator
from app.agents.nlp_agent import nlp_agent
from app.agents.routing_agent import routing_agent
from app.database.models import Complaint
from app.clients.llm_client import llm_client

logger = logging.getLogger("pcms.conversational")

CATEGORY_DISPLAY_NAMES = {
    "streetlight": {"en": "Streetlight", "mr": "पथदिवा (Streetlight)"},
    "pothole": {"en": "Pothole & Road Damage", "mr": "खड्डा व रस्ता दुरुस्ती"},
    "garbage": {"en": "Garbage & Cleanliness", "mr": "कचरा व स्वच्छता"},
    "drainage": {"en": "Drainage & Sewerage", "mr": "ड्रेनेज व सांडपाणी"},
    "pipeline_water_leakage": {"en": "Water Pipeline Leakage", "mr": "पाणी गळती व पुरवठा"},
    "traffic_jams": {"en": "Traffic Congestion", "mr": "वाहतूक कोंडी"},
    "trees": {"en": "Fallen Trees & Garden", "mr": "झाड पडणे व उद्यान"},
    "noise_pollution": {"en": "Noise Pollution", "mr": "ध्वनी प्रदूषण"},
    "encroachment": {"en": "Footpath Encroachment", "mr": "अतिक्रमण"},
    "electricity": {"en": "Electric Hazard / DP", "mr": "विद्युत धोका / डीपी"},
    "unauthorized_banner_flex": {"en": "Illegal Banners & Flex", "mr": "अनधिकृत फ्लेक्स व बॅनर"}
}


class ConversationalService:
    def detect_language(self, text: str) -> str:
        """
        Detects whether input text is primarily Marathi ('mr') or English ('en').
        """
        if not text:
            return "mr"

        # Check for Devanagari characters
        devanagari_count = len(re.findall(r'[\u0900-\u097F]', text))
        if devanagari_count > 0:
            return "mr"

        # Transliterated Marathi keywords check
        marathi_roman_keywords = [
            "aahe", "ahe", "nahi", "mala", "amcha", "amchya", "rasta", "khadda",
            "kachra", "pani", "samasya", "durust", "takraar", "takrar", "kara",
            "kiti", "kuthe", "madhe", "jawal", "kela", "zala", "namaskar",
            "kasa", "kashi", "kay", "aani", "pan", "ho", "aata", "vattit"
        ]
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        for w in words:
            if w in marathi_roman_keywords:
                return "mr"

        return "en"

    def is_greeting(self, text: str) -> bool:
        """Checks if user text is merely a greeting or casual opener."""
        cleaned = text.strip().lower()
        # Common English/Roman greeting variations (hi, hii, hiii, hello, heyy, hlo, etc.)
        if re.match(r"^(h+[i|y]+|h+e+l+o+|h+e+y+|h+l+o+|g+m+|g+n+|good\s*morning|good\s*afternoon|good\s*evening)[\s!\.]*$", cleaned):
            return True
        # Marathi Devanagari greetings
        if re.match(r"^(नमस्कार|रामराम|शुभ\s*सकाळ|शुभ\s*दुपार|शुभ\s*संध्याकाळ|प्रणाम|जय\s*महाराष्ट्र)[\s!\.]*$", cleaned):
            return True
        # Transliterated greetings & openers
        if re.match(r"^(namaskar|namaste|radhe\s*radhe|kasa\s*ahes|kashi\s*ahes|how\s*are\s*you|how\s*r\s*u)[\s!\.]*$", cleaned):
            return True
        return False

    def is_registration_intent(self, text: str, action: Optional[str] = None, confirm_register: bool = False) -> bool:
        """Determines if the user explicitly confirms or asks to register a complaint."""
        if confirm_register or action == "register":
            return True

        t_lower = text.lower()
        registration_triggers = [
            "register complaint", "raise complaint", "file complaint", "lodge complaint",
            "तक्रार नोंदवा", "तक्रार दाखल करा", "नोंदणी करा", "तक्रार सबमिट करा", "तक्रार करा",
            "takrar nondva", "takrar dakhala", "takrar kara", "register ticket", "book complaint"
        ]
        return any(trig in t_lower for trig in registration_triggers)

    def is_tracking_intent(self, text: str) -> Optional[str]:
        """Checks if user wants to track a ticket and extracts ticket ID if available."""
        t_upper = text.upper()
        ticket_match = re.search(r'\bWM-\d{8}-\d{4}\b', t_upper)
        if ticket_match:
            return ticket_match.group(0)

        t_lower = text.lower()
        if any(w in t_lower for w in ["track", "status", "तपासा", "स्थिती", "तिकीट"]):
            return "UNKNOWN"
        return None

    def handle_chat(
        self,
        db: Session,
        message: Optional[str] = None,
        category: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        citizen_phone: Optional[str] = "Anonymous",
        photo_filename: Optional[str] = None,
        photo_bytes: Optional[bytes] = None,
        confirm_register: bool = False,
        action: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Core conversational controller powered by Ollama.
        """
        text = (message or "").strip()
        lang = self.detect_language(text)

        # 1. Handle Greetings (hi, hii, hello, gm, नमस्कार etc.)
        if text and self.is_greeting(text):
            if lang == "mr":
                reply = (
                    "नमस्कार! 👋 मी PCMC सारथी / वॉर्डमित्र AI सहाय्यक आहे.\n"
                    "मी पिंपरी चिंचवड मधील रस्ते, पथदिवे, कचरा, ड्रेनेज, पाणी पुरवठा अशा नागरी समस्या सोडवण्यात मदत करतो. "
                    "मी आज आपली काय मदत करू शकतो?"
                )
            else:
                reply = (
                    "Hello! 👋 I am your WardMitra / PCMC Sarathi AI assistant.\n"
                    "I can help you report and track civic issues such as potholes, streetlights, garbage, or water leaks in your ward. "
                    "How can I assist you today?"
                )
            return {
                "reply": reply,
                "intent": "GREETING",
                "language": lang,
                "category": None,
                "category_name": None,
                "ticket_data": None,
                "action_prompt": "select_or_describe"
            }

        # 2. Handle Tracking Request
        ticket_id_found = self.is_tracking_intent(text)
        if ticket_id_found and ticket_id_found != "UNKNOWN":
            complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id_found).first()
            if complaint:
                worker_str = complaint.assigned_worker_name or "Ward Field Engineer"
                worker_contact = f" ({complaint.assigned_worker_contact})" if complaint.assigned_worker_contact else ""
                if lang == "mr":
                    reply = (
                        f"🔍 **तक्रार स्थिती: {complaint.ticket_id}**\n"
                        f"• स्थिती: {complaint.status.value}\n"
                        f"• वर्ग: {complaint.detected_category}\n"
                        f"• प्रभाग: Ward {complaint.ward_number or '19A'}\n"
                        f"• नियुक्त कामगार: {worker_str}{worker_contact}\n"
                        f"• मुदत (SLA): {complaint.sla_hours} तास"
                    )
                else:
                    reply = (
                        f"🔍 **Complaint Status: {complaint.ticket_id}**\n"
                        f"• Status: {complaint.status.value}\n"
                        f"• Category: {complaint.detected_category}\n"
                        f"• Ward: Ward {complaint.ward_number or '19A'}\n"
                        f"• Assigned Worker: {worker_str}{worker_contact}\n"
                        f"• SLA Duration: {complaint.sla_hours} Hours"
                    )
                return {
                    "reply": reply,
                    "intent": "TRACK_COMPLAINT",
                    "language": lang,
                    "category": complaint.detected_category,
                    "category_name": complaint.detected_category,
                    "ticket_data": {
                        "ticket_id": complaint.ticket_id,
                        "status": complaint.status.value,
                        "ward": complaint.ward_number,
                        "assigned_worker_name": worker_str,
                        "assigned_worker_contact": complaint.assigned_worker_contact or "020-67333333"
                    },
                    "action_prompt": None
                }
            else:
                msg_mr = f"⚠️ तक्रार क्र. '{ticket_id_found}' सापडली नाही. कृपया योग्य तिकीट नंबर तपासा."
                msg_en = f"⚠️ Ticket '{ticket_id_found}' was not found. Please double-check your ticket number."
                return {
                    "reply": msg_mr if lang == "mr" else msg_en,
                    "intent": "TRACK_COMPLAINT",
                    "language": lang,
                    "category": None,
                    "category_name": None,
                    "ticket_data": None,
                    "action_prompt": None
                }

        # 3. Detect Civic Category if any
        detected_category = category
        if not detected_category and text:
            text_clean = text.lower()
            for cat_key in CATEGORY_DISPLAY_NAMES.keys():
                if cat_key in text_clean:
                    detected_category = cat_key
                    break
            if not detected_category:
                nlp_res = nlp_agent.extract_intent_and_category(text)
                if nlp_res.get("confidence", 0) >= 0.80:
                    detected_category = nlp_res.get("category")

        cat_info = CATEGORY_DISPLAY_NAMES.get(detected_category, {"en": "Civic Grievance", "mr": "नागरी समस्या"})
        cat_name_en = cat_info["en"]
        cat_name_mr = cat_info["mr"]
        cat_name_current = cat_name_mr if lang == "mr" else cat_name_en

        # 4. Handle Direct Category Selection Chip Click (action == "select_category")
        if action == "select_category" and detected_category:
            if lang == "mr":
                reply = (
                    f"मी समजलो! आपली समस्या **{cat_name_mr}** संदर्भातील आहे. 💡\n\n"
                    f"कृपया समस्येचे अचूक ठिकाण सांगा (किंवा फोटो जोडा), आणि खालील **'तक्रार नोंदवा'** बटणावर टॅप करा जेणेकरून आम्ही संबंधित प्रभागातील कामगाराकडे ही तक्रार नोंदवू शकू."
                )
            else:
                reply = (
                    f"Understood! Your issue is related to **{cat_name_en}**. 💡\n\n"
                    f"Please share the location or describe what needs fixing (or attach a photo), and tap **'Register Complaint'** so we can assign it directly to the ward worker."
                )

            return {
                "reply": reply,
                "intent": "CATEGORY_SELECTED",
                "language": lang,
                "category": detected_category,
                "category_name": cat_name_current,
                "ticket_data": None,
                "action_prompt": "confirm_or_detail"
            }

        # 5. User Confirmed Registration or Uploaded Evidence Image
        user_wants_registration = self.is_registration_intent(text, action, confirm_register) or bool(photo_bytes)
        if user_wants_registration:
            final_cat = detected_category or "pothole"
            final_desc = text if text else f"Grievance regarding {cat_name_en}"

            reg_result = orchestrator.process_registration(
                db=db,
                description=final_desc,
                latitude=latitude,
                longitude=longitude,
                citizen_phone=citizen_phone,
                photo_filename=photo_filename,
                photo_bytes=photo_bytes
            )

            if reg_result.get("moderation_status") == "REJECTED" or reg_result.get("is_fraud"):
                reason = reg_result.get("fraud_reason") or "Inappropriate content"
                if lang == "mr":
                    reply = f"🚫 आपली तक्रार नाकारली आहे: {reason}"
                else:
                    reply = f"🚫 Complaint rejected: {reason}"
                return {
                    "reply": reply,
                    "intent": "MODERATION_REJECTED",
                    "language": lang,
                    "category": reg_result.get("detected_category"),
                    "category_name": cat_name_current,
                    "ticket_data": reg_result,
                    "action_prompt": None
                }

            worker_name = reg_result.get("assigned_worker_name") or ("Er. संदीप माने" if lang == "mr" else "Er. Sandeep Mane")
            worker_contact = reg_result.get("assigned_worker_contact") or "020-67333333"

            if lang == "mr":
                reply = (
                    f"🎉 **आपली तक्रार यशस्वीरित्या नोंदवली गेली आहे!**\n\n"
                    f"• **तिकीट क्र:** {reg_result['ticket_id']}\n"
                    f"• **श्रेणी:** {cat_name_mr}\n"
                    f"• **प्रभाग:** Ward {reg_result.get('ward') or '19A'}\n"
                    f"• **नेमलेले कामगार:** {worker_name} ({worker_contact})\n"
                    f"• **निकालाची मुदत (SLA):** {reg_result.get('sla_hours', 24)} तास\n\n"
                    f"कामगार लवकरच समस्येचे निवारण करतील. आपण कधीही तिकीट नंबरवरून स्थिती तपासू शकता."
                )
            else:
                reply = (
                    f"🎉 **Your complaint has been successfully registered!**\n\n"
                    f"• **Ticket ID:** {reg_result['ticket_id']}\n"
                    f"• **Category:** {cat_name_en}\n"
                    f"• **Ward:** Ward {reg_result.get('ward') or '19A'}\n"
                    f"• **Assigned Worker:** {worker_name} ({worker_contact})\n"
                    f"• **Resolution Deadline (SLA):** {reg_result.get('sla_hours', 24)} Hours\n\n"
                    f"The assigned worker has been notified. You can track this ticket anytime."
                )

            return {
                "reply": reply,
                "intent": "COMPLAINT_REGISTERED",
                "language": lang,
                "category": reg_result.get("detected_category"),
                "category_name": cat_name_current,
                "ticket_data": {
                    "ticket_id": reg_result["ticket_id"],
                    "detected_category": reg_result.get("detected_category"),
                    "category_name": cat_name_current,
                    "ward": reg_result.get("ward") or 19,
                    "sla_hours": reg_result.get("sla_hours", 24),
                    "assigned_worker_name": worker_name,
                    "assigned_worker_contact": worker_contact,
                    "status": reg_result.get("status", "REGISTERED"),
                    "moderation_status": "PASSED"
                },
                "action_prompt": "track"
            }

        # 6. Natural Conversational Response Powered by Ollama
        llm_reply = None
        if text:
            sys_lang = "Marathi (मराठी)" if lang == "mr" else "English"
            system_prompt = (
                f"You are WardMitra / PCMC Sarathi, an intelligent and polite municipal AI assistant for Pimpri Chinchwad Municipal Corporation (PCMC). "
                f"You help citizens with civic information, municipal services, and resolving issues (streetlights, potholes, garbage, water supply, drainage, etc.). "
                f"Respond naturally and politely in {sys_lang}. "
                f"Keep your response concise, helpful and under 3 sentences. "
                f"If the citizen has a grievance or issue to report, let them know you can help them register it."
            )
            llm_reply = llm_client.generate(prompt=text, system_prompt=system_prompt)

        if llm_reply and len(llm_reply.strip()) > 5:
            reply = llm_reply.strip()
        else:
            if lang == "mr":
                reply = (
                    "मी आपल्या सेवेत हजर आहे. आपण रस्त्यावरील खड्डे, बंद पथदिवे, कचरा किंवा पाणी गळती यांसारख्या समस्या नोंदवू शकता. "
                    "कृपया समस्या सांगा किंवा वरील पर्यायांपैकी एकावर टॅप करा."
                )
            else:
                reply = (
                    "I am here to assist you. You can report civic issues such as potholes, dark streetlights, garbage heaps, or water pipe leakages. "
                    "Please describe the issue or select one of the category options above."
                )

        return {
            "reply": reply,
            "intent": "CONVERSATIONAL",
            "language": lang,
            "category": detected_category,
            "category_name": cat_name_current if detected_category else None,
            "ticket_data": None,
            "action_prompt": "confirm_or_detail" if detected_category else "select_or_describe"
        }


conversational_service = ConversationalService()
