"""
Multilingual (Marathi & English) Conversational Service for WardMitra AI / PCMC Sarathi AI.
Enforces intelligent multi-turn dialogue:
1. Clarification before complaint creation ("The light is off" -> home vs streetlight)
2. Category buttons do not auto-register -> asks for problem description
3. Strict language adherence (English <-> English, Marathi <-> Marathi, Unclear -> prompt to re-describe)
4. Image evidence verification & low-confidence photo rejection
5. Verification gate before complaint registration
"""

import re
import os
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.orchestrator.orchestrator import orchestrator
from app.agents.nlp_agent import (
    nlp_agent,
    INTENT_NEW_COMPLAINT,
    INTENT_CATEGORY_SELECTION,
    INTENT_CATEGORY_CONFIRMATION,
    INTENT_CATEGORY_CORRECTION,
    INTENT_LOCATION_PROVIDED,
    INTENT_DESCRIPTION_PROVIDED,
    INTENT_REGISTRATION_CONFIRMATION,
    INTENT_REGISTRATION_REJECTION,
    INTENT_CANCEL_COMPLAINT,
    INTENT_IMAGE_UPLOADED,
    INTENT_IMAGE_CLARIFICATION,
    INTENT_STATUS_QUERY,
    INTENT_FEEDBACK,
    INTENT_ABUSIVE_MESSAGE,
    INTENT_GREETING,
    INTENT_UNRELATED,
    INTENT_START_NEW_COMPLAINT
)
from app.agents.image_agent import image_agent
from app.agents.routing_agent import routing_agent
from app.database.models import Complaint, ComplaintStatus
from app.agents.geo_agent import geo_agent
from app.clients.llm_client import llm_client
from app.services.human_persona_service import human_persona_service
from app.services.image_understanding_service import image_understanding_service, ImageAnalysisResult
from app.services.conversation_state_manager import conversation_state_manager, ConversationState
from app.services.complaint_status_service import complaint_status_service
from app.services.feedback_service import feedback_service
from app.config.category_registry import CIVIC_12_CATEGORIES, normalize_category_key, get_category_info
from app.rules.category_rules import (
    is_ambiguous_light_complaint,
    get_light_clarification_prompt,
    get_streetlight_evidence_prompt,
    get_category_clarification_prompt
)
from app.config.settings import settings

logger = logging.getLogger("pcms.conversational")

CATEGORY_DISPLAY_NAMES: Dict[str, Dict[str, str]] = {}
for cat in CIVIC_12_CATEGORIES:
    CATEGORY_DISPLAY_NAMES[cat["key"]] = {
        "en": cat["name_en"],
        "mr": cat["name_mr"],
        "hi": cat["name_hi"]
    }
    for alias in cat["aliases"]:
        CATEGORY_DISPLAY_NAMES[alias] = {
            "en": cat["name_en"],
            "mr": cat["name_mr"],
            "hi": cat["name_hi"]
        }

# Additional common aliases
CATEGORY_DISPLAY_NAMES["damaged_streetlights"] = {"en": "Streetlight", "mr": "पथदिवा (Streetlight)", "hi": "स्ट्रीट लाइट"}
CATEGORY_DISPLAY_NAMES["potholes"] = {"en": "Pothole & Road Damage", "mr": "खड्डा व रस्ता दुरुस्ती", "hi": "सड़क के गड्ढे"}
CATEGORY_DISPLAY_NAMES["road_damage"] = {"en": "Road Damage", "mr": "रस्ता दुरुस्ती", "hi": "सड़क खराबी"}
CATEGORY_DISPLAY_NAMES["overflowing_garbage"] = {"en": "Garbage & Cleanliness", "mr": "कचरा व स्वच्छता", "hi": "कचरा और सफाई"}
CATEGORY_DISPLAY_NAMES["illegal_debris_dumping"] = {"en": "Illegal Debris Dumping", "mr": "अनधिकृत मलबा व कचरा", "hi": "अवैध मलबा डंपिंग"}
CATEGORY_DISPLAY_NAMES["drainage_failures"] = {"en": "Drainage & Sewerage", "mr": "ड्रेनेज व सांडपाणी", "hi": "ड्रेनेज और सीवर"}
CATEGORY_DISPLAY_NAMES["water_pipeline_leakages"] = {"en": "Water Pipeline Leakage", "mr": "पाणी गळती व पुरवठा", "hi": "पानी पाइपलाइन लीकेज"}
CATEGORY_DISPLAY_NAMES["pipelinedefects"] = {"en": "Water Pipeline Leakage", "mr": "पाणी गळती व पुरवठा", "hi": "पानी पाइपलाइन लीकेज"}
CATEGORY_DISPLAY_NAMES["water_supply"] = {"en": "Water Supply", "mr": "पाणीपुरवठा", "hi": "जलापूर्ति"}
CATEGORY_DISPLAY_NAMES["irregular_water_supply"] = {"en": "Irregular Water Supply", "mr": "अनियमित पाणीपुरवठा", "hi": "अनियमित जलापूर्ति"}
CATEGORY_DISPLAY_NAMES["road_incidents_traffic"] = {"en": "Traffic & Road Incidents", "mr": "वाहतूक कोंडी व रस्ता समस्या", "hi": "ट्रैफिक और सड़क घटनाएं"}
CATEGORY_DISPLAY_NAMES["banners_flex"] = {"en": "Illegal Banners & Flex", "mr": "अनधिकृत फ्लेक्स व बॅनर", "hi": "अवैध बैनर और फ्लेक्स"}



class ConversationalService:
    state_manager = conversation_state_manager

    def detect_language(self, text: str) -> str:
        """Delegates language detection to nlp_agent."""
        return nlp_agent.detect_language(text)

    def is_greeting(self, text: str) -> bool:
        """Checks if user text is a greeting or casual opener without a civic grievance."""
        cleaned = text.strip().lower()
        civic_keywords = [
            "pothole", "khadda", "drainage", "gutter", "garbage", "kachra",
            "streetlight", "light", "diwa", "pani", "water", "leak", "wire",
            "traffic", "encroachment", "banner", "खड्डा", "कचरा", "ड्रेनेज",
            "गटर", "पाणी", "लाईट", "दिवा", "तार", "वाहतूक", "अतिक्रमण", "झाड"
        ]
        if any(w in cleaned for w in civic_keywords):
            return False

        greeting_patterns = [
            r"^(h+[i|y]+|h+e+l+o+|h+e+y+|h+l+o+|g+m+|g+n+|good\s*(morning|afternoon|evening))\b",
            r"^(नमस्कार|रामराम|शुभ\s*सकाळ|शुभ\s*दुपार|शुभ\s*संध्याकाळ|प्रणाम|जय\s*महाराष्ट्र|जय\s*शिवराय|सुप्रभात)\b",
            r"^(namaskar|namaste|radhe\s*radhe|kasa\s*ahes|kashi\s*ahes|how\s*are\s*you|how\s*r\s*u)\b"
        ]
        return any(re.search(p, cleaned) for p in greeting_patterns)

    def extract_greeting_name(self, text: str) -> Optional[str]:
        """Extracts a user name if the greeting includes one (e.g. 'hello Siddheshwar')."""
        cleaned = text.strip()
        m = re.search(r'^(?:hello|hi+|hey+|namaskar|namaste|नमस्कार|रामराम)\s+([A-Za-z\u0900-\u097F]+)', cleaned, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            ignored = {"there", "all", "everyone", "bot", "ai", "sir", "madam", "ji", "bhau", "to", "wardmitra", "pcmc"}
            if candidate.lower() not in ignored:
                return candidate.capitalize()
        return None

    def extract_name_introduction(self, text: str) -> Optional[str]:
        """Detects if user is introducing their name (e.g. 'माझं नाव राहुल आहे', 'my name is Rahul')."""
        cleaned = text.strip()
        patterns = [
            r'(?:माझं|माझे)\s*नाव\s*([A-Za-z\u0900-\u097F]+)(?:\s*आहे)?',
            r'\bmy\s*name\s*is\s*([A-Za-z]+)\b',
            r'\bi\s*am\s*([A-Za-z]+)\b'
        ]
        for pat in patterns:
            m = re.search(pat, cleaned, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                if cand.lower() not in {"a", "an", "the", "fine", "good", "okay", "here", "citizen", "नागरिक", "वॉर्डमित्र", "sarathi"}:
                    return cand.capitalize()
        return None

    def is_smalltalk_or_gratitude(self, text: str) -> bool:
        """Checks if message is gratitude, acknowledgement, or goodbye."""
        cleaned = text.strip().lower()
        patterns = [
            r"\b(thank\s*you|thanks|thx|dhanyavad|धन्यवाद|आभार|थँक्यू|थँक्स|थॅन्क्स)\b",
            r"^(ok|okay|fine|alright|cool|छान|बरं|ठीक\s*आहे|bye|goodbye|टाटा|अलविदा)[\s!\.]*$"
        ]
        return any(re.search(p, cleaned) for p in patterns)

    def is_bot_inquiry_or_help(self, text: str) -> bool:
        """Checks if message is inquiring about bot identity or asking how to use the system."""
        cleaned = text.strip().lower()
        patterns = [
            r"\b(who\s*are\s*(?:you|u)|what\s*is\s*your\s*name|what\s*can\s*you\s*do|what\s*do\s*you\s*do|help\s*me|how\s*to\s*use)\b",
            r"(?:तू\s*कोण\s*आहेस|तू\s*कोण|तुम्ही\s*कोण\s*आहात|तुम्ही\s*कोण|तुझं\s*नाव\s*काय|तुमचं\s*नाव\s*काय|तू\s*काय\s*करतोस|काय\s*करू\s*शकता|मदत\s*करा|माहिती\s*द्या)",
            r"\b(tu\s*kon\s*ahes|tu\s*kon|tumhi\s*kon\s*ahat|tuz\s*nav\s*kay)\b"
        ]
        return any(re.search(p, cleaned) for p in patterns)

    def is_complaint_process_inquiry(self, text: str) -> bool:
        """
        Checks if citizen is asking how to register, how to file a complaint,
        what the process/steps are, or how the grievance system works.
        Designed for citizens from both urban (City) and rural (Gramin) areas.
        """
        cleaned = text.strip().lower()
        patterns = [
            r'\bhow\s+(?:to|can\s+i|do\s+i|should\s+i)\s+(?:register|file|lodge|raise|submit|make|put|report|complain)',
            r'\bhow\s+(?:to|can\s+i|do\s+i).*?\b(?:complaint|grievance|ticket)\b',
            r'\b(?:what\s+is\s+the\s+process|registration\s+process|complaint\s+process|steps\s+to\s+register|how\s+does\s+this\s+work)\b',
            r'\b(?:procedure|steps|guide|help)\s+.*?\b(?:register|file|complaint|grievance)\b',
            r'\b(?:guide\s+me|teach\s+me|show\s+me\s+how|help\s+me\s+register)\b',
            r'(?:तक्रार\s+कशी|कशी\s+तक्रार|कशी\s+नोंदवा|कशी\s+करायची|कशी\s+करावी|नोंदवण्याची\s+पद्धत|नोंदणी\s+कशी|प्रक्रिया|पायऱ्या|माहिती\s+द्या|कशी\s+नोंदवू)',
            r'(?:मला\s+(?:एक\s+)?(?:तक्रार|कम्प्लेंट)|तक्रार\s+(?:नोंदवायची|करायची|द्यायची|दाखल\s+करायची)|कम्प्लेंट\s+(?:कशी|करायची|नोंदवायची)|नवीन\s+तक्रार\s+करायची)',
            r'(?:मला\s+मार्गदर्शन\s*करा|मार्गदर्शन\s*करा|मला\s*शिकवा|शिकवा|समजावून\s*सांगा|कसं\s*करायचं)',
            r'\b(?:मार्गदर्शन|शिकवा)\b',
            r'\bi\s+(?:want|need|wish)\s+to\s+(?:register|file|lodge|raise|submit|make)\s+(?:a\s+)?(?:complaint|grievance|ticket)\b',
            r'\b(?:takrar\s+kashi|kashi\s+takrar|process\s+sanga|step\s*by\s*step|kashi\s+karaychi)\b'
        ]
        return any(re.search(p, cleaned) for p in patterns) or nlp_agent.is_complaint_process_inquiry(cleaned)

    def is_cancel_intent(self, text: str) -> bool:
        """Checks if user wants to cancel an active complaint."""
        return nlp_agent.is_cancel_complaint_intent(text)

    def is_registration_intent(self, text: str, action: Optional[str] = None, confirm_register: bool = False) -> bool:
        """
        Determines if user explicitly confirmed grievance registration.
        CRITICAL: Never matches on inquiries like 'मला तक्रार करायची आहे' or single words within other words (e.g. 'होता', 'करायची')!
        """
        if confirm_register or action in ["register", "register_new", "force_register"]:
            return True

        t_clean = (text or "").lower().strip()
        if not t_clean:
            return False

        # If user is inquiring how to file, that is NOT registration intent!
        if self.is_complaint_process_inquiry(t_clean):
            return False

        explicit_phrases = [
            "register complaint", "raise complaint", "file complaint", "lodge complaint",
            "तक्रार नोंदवा", "तक्रार दाखल करा", "नोंदणी करा", "तक्रार सबमिट करा", "नोंदणी पूर्ण करा",
            "takrar nondva", "takrar dakhala", "register ticket", "book complaint",
            "please fix", "action ghyava", "okay register", "yes please register"
        ]
        if any(phrase in t_clean for phrase in explicit_phrases):
            return True

        # Short affirmative words - whole word match only
        words = re.findall(r'[\w\u0900-\u097F]+', t_clean)
        affirmative_words = {"हो", "होय", "yes", "yep", "sure", "करा", "नोंदवा", "submit", "register", "urgent", "proceed"}
        if any(w in affirmative_words for w in words):
            # Only trigger if the message is a concise confirmation (<= 3 words), or explicitly contains 'नोंदवा'/'register'/'submit'
            if len(words) <= 3 or any(w in ["नोंदवा", "register", "submit"] for w in words):
                return True

        return False

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

    def is_affirmative_confirmation(self, text: str, action: Optional[str] = None) -> bool:
        """Detects if user confirms video detection (e.g. 'Yes, Register Complaint', 'होय', 'हाँ')."""
        if action == "confirm_video_yes":
            return True
        if not text:
            return False
        t_clean = text.strip().lower()
        words = set(re.findall(r'[\w\u0900-\u097F]+', t_clean))
        positives = {
            "yes", "yep", "sure", "ok", "okay", "confirm", "proceed", "register",
            "submit", "correct", "right", "yeah", "y", "done", "plz", "please",
            "हो", "होय", "करा", "नोंदवा", "नक्की", "चालू", "बरोबर", "करून", "द्या",
            "हाँ", "हा", "करो", "दर्ज", "सही", "ज़रूर", "जरूर", "बिल्कुल"
        }
        negatives = {"no", "not", "dont", "don't", "नाही", "नको", "नहीं", "गलत", "different", "वेगळी", "दुसरी"}
        if any(w in positives for w in words):
            if not any(w in negatives for w in words):
                return True
        return False

    def is_negative_confirmation(self, text: str, action: Optional[str] = None) -> bool:
        """Detects if user declines video detection (e.g. 'No, Different Complaint', 'नाही', 'नहीं')."""
        if action == "confirm_video_no":
            return True
        if not text:
            return False
        t_clean = text.strip().lower()
        words = set(re.findall(r'[\w\u0900-\u097F]+', t_clean))
        negatives = {
            "no", "nope", "not", "dont", "don't", "different", "cancel", "wrong", "reject",
            "नाही", "नको", "वेगळी", "दुसरी", "चुकीचे", "रद्द",
            "नहीं", "ना", "दूसरी", "गलत", "रद्द", "मत"
        }
        return bool(any(w in negatives for w in words))

    def _detect_light_visual_status(self, text: str) -> Optional[str]:
        """
        Extracts visual status ('on' or 'off') for streetlight complaints.
        Safely avoids false positives like 'streetlight on MG road'.
        """
        if not text:
            return None
        t_clean = text.lower().strip()

        # Exact short responses
        if t_clean in ["on", "chalu", "chal raha", "चालू"]:
            return "on"
        if t_clean in ["off", "band", "dead", "dark", "बंद"]:
            return "off"

        # Explicit OFF signals
        is_light_off = (
            bool(re.search(r'\b(not working|nahi chal|chalunahi|चालू नाही|बंद आहे|लाईट बंद|light band|band hai|band aahe|blackout)\b', t_clean)) or
            bool(re.search(r'\b(?:lights?|lamp)\s+(?:is|are|stays?|turned|switched)?\s*off\b', t_clean)) or
            bool(re.search(r'\b(?:is|are|stays?|turned|switched)\s+off\b', t_clean))
        )

        # Explicit ON signals (burning during daytime)
        is_light_on = (
            bool(re.search(r'\b(chalu hai|chal raha hai|daytime|day time|burning during day|चालू आहे|लाईट चालू)\b', t_clean)) or
            bool(re.search(r'\b(?:lights?|lamp)\s+(?:is|are|stays?|turned|switched)\s+on\b', t_clean)) or
            bool(re.search(r'\b(?:is|are|stays?|remains?)\s+on\b', t_clean))
        )

        if is_light_off and not is_light_on:
            return "off"
        elif is_light_on and not is_light_off:
            return "on"
        return None


    def generate_video_confirmation_prompt(self, video_result: Any, lang: str = "en") -> Dict[str, Any]:
        raw_cat = getattr(video_result, "detected_category", None) or getattr(video_result, "category", "streetlight")
        cat = normalize_category_key(raw_cat)
        issue = getattr(video_result, "detected_issue", None) or getattr(video_result, "issue", "unknown")
        v_stat = getattr(video_result, "visual_status", None)
        conf_level = getattr(video_result, "confidence_level", "high")

        cat_info = CATEGORY_DISPLAY_NAMES.get(cat, {"en": cat.replace('_', ' ').title(), "mr": cat, "hi": cat})
        cat_en = cat_info.get("en", cat.replace('_', ' ').title())
        cat_mr = cat_info.get("mr", cat)
        cat_hi = cat_info.get("hi", cat)

        if cat == "streetlight":
            if v_stat == "off":
                if conf_level == "high":
                    msg_en = "I can see a streetlight-related issue in the video, and the light appears to be OFF. Do you want to register a complaint for this issue?"
                    msg_mr = "व्हिडिओमध्ये रस्त्यावरील दिव्याची (Streetlight) समस्या दिसत आहे आणि दिवा बंद (OFF) असल्याचे दिसते. आपण या समस्येसाठी तक्रार नोंदवू इच्छिता का?"
                    msg_hi = "वीडियो में स्ट्रीटलाइट से संबंधित समस्या दिखाई दे रही है और लाइट बंद (OFF) प्रतीत होती है। क्या आप इस समस्या के लिए शिकायत दर्ज करना चाहते हैं?"
                else:
                    msg_en = "It looks like there might be a streetlight issue in the video, and the light appears to be OFF. Would you like to register a complaint for this issue?"
                    msg_mr = "व्हिडिओमध्ये रस्त्यावरील दिव्याची समस्या असावी असे दिसते आणि दिवा बंद (OFF) वाटतो. आपण या समस्येसाठी तक्रार नोंदवू इच्छिता का?"
                    msg_hi = "वीडियो में स्ट्रीटलाइट की समस्या हो सकती है और लाइट बंद प्रतीत होती है। क्या आप इस समस्या के लिए शिकायत दर्ज करना चाहते हैं?"
            elif v_stat == "on":
                if conf_level == "high":
                    msg_en = "I can see a streetlight in the video that appears to be burning during daytime. Do you want to register a complaint for this issue?"
                    msg_mr = "व्हिडिओमध्ये दिवसा पथदिवा चालू (ON) असल्याचे दिसत आहे. आपण या समस्येसाठी तक्रार नोंदवू इच्छिता का?"
                    msg_hi = "वीडियो में दिन के समय स्ट्रीटलाइट चालू (ON) दिखाई दे रही है। क्या आप इस समस्या के लिए शिकायत दर्ज करना चाहते हैं?"
                else:
                    msg_en = "It looks like there might be a streetlight burning during daytime in the video. Would you like to register a complaint for this issue?"
                    msg_mr = "व्हिडिओमध्ये दिवसा पथदिवा चालू असावा असे दिसते. आपण या समस्येसाठी तक्रार नोंदवू इच्छिता का?"
                    msg_hi = "वीडियो में दिन के समय स्ट्रीटलाइट चालू हो सकती है। क्या आप इस समस्या के लिए शिकायत दर्ज करना चाहते हैं?"
            else:
                if conf_level == "high":
                    msg_en = "I can identify a streetlight issue in your video. Would you like to register a complaint for this streetlight?"
                    msg_mr = "आपल्या व्हिडिओमध्ये पथदिव्याची समस्या स्पष्ट दिसत आहे. आपण या पथदिव्यासाठी तक्रार नोंदवू इच्छिता का?"
                    msg_hi = "वीडियो में स्ट्रीटलाइट से संबंधित समस्या दिखाई दे रही है। क्या आप इस स्ट्रीटलाइट के लिए शिकायत दर्ज करना चाहते हैं?"
                else:
                    msg_en = "It looks like there might be a streetlight issue in the video. Would you like to register a complaint for this streetlight?"
                    msg_mr = "व्हिडिओमध्ये पथदिव्याची समस्या असावी असे दिसते. आपण या पथदिव्यासाठी तक्रार नोंदवू इच्छिता का?"
                    msg_hi = "वीडियो में स्ट्रीटलाइट की समस्या प्रतीत होती है। क्या आप इस स्ट्रीटलाइट के लिए शिकायत दर्ज करना चाहते हैं?"
        elif cat in ["pothole", "potholes", "road_damage"]:
            if conf_level == "high":
                msg_en = "I identified a pothole on the road from your video. Would you like to submit a complaint for road repair?"
                msg_mr = "आपल्या व्हिडिओमध्ये रस्त्यावरील खड्डा (Pothole) स्पष्टपणे दिसत आहे. आपण रस्ता दुरुस्तीसाठी तक्रार नोंदवू इच्छिता का?"
                msg_hi = "आपके वीडियो में सड़क पर गड्ढा (Pothole) पहचाना गया है। क्या आप सड़क मरम्मत के लिए शिकायत दर्ज करना चाहते हैं?"
            else:
                msg_en = "It looks like there might be a pothole or road damage in the video. Would you like to register a complaint for road repair?"
                msg_mr = "व्हिडिओमध्ये रस्त्यावर खड्डा किंवा रस्त्याचे नुकसान असावे असे दिसते. आपण रस्ता दुरुस्तीसाठी तक्रार नोंदवू इच्छिता का?"
                msg_hi = "वीडियो में सड़क पर गड्ढा या सड़क क्षति दिखाई दे रही है। क्या आप सड़क मरम्मत के लिए शिकायत दर्ज करना चाहते हैं?"
        elif cat in ["garbage", "overflowing_garbage", "illegal_debris_dumping"]:
            if conf_level == "high":
                msg_en = "I identified an overflowing garbage issue in your video. Would you like to submit a complaint for garbage cleaning?"
                msg_mr = "आपल्या व्हिडिओमध्ये कचऱ्याचा ढीग / अस्वच्छता स्पष्टपणे दिसत आहे. आपण कचरा उचलण्यासाठी तक्रार नोंदवू इच्छिता का?"
                msg_hi = "आपके वीडियो में कचरे का ढेर दिखाई दे रहा है। क्या आप कचरा सफाई के लिए शिकायत दर्ज करना चाहते हैं?"
            else:
                msg_en = "It looks like there might be an overflowing garbage issue in the video. Would you like to register a complaint for garbage cleaning?"
                msg_mr = "व्हिडिओमध्ये कचऱ्याचा ढीग किंवा अस्वच्छता असावी असे दिसते. आपण कचरा उचलण्यासाठी तक्रार नोंदवू इच्छिता का?"
                msg_hi = "वीडियो में कचरा या गंदगी की समस्या हो सकती है। क्या आप कचरा सफाई के लिए शिकायत दर्ज करना चाहते हैं?"
        elif cat in ["drainage", "drainage_failures"]:
            if conf_level == "high":
                msg_en = "I identified a drainage leakage / blockage issue in your video. Would you like to submit a complaint for drainage maintenance?"
                msg_mr = "आपल्या व्हिडिओमध्ये ड्रेनेज / सांडपाण्याची समस्या दिसत आहे. आपण ड्रेनेज दुरुस्तीसाठी तक्रार नोंदवू इच्छिता का?"
                msg_hi = "आपके वीडियो में ड्रेनेज या नाली की समस्या दिखाई दे रही है। क्या आप ड्रेनेज मरम्मत के लिए शिकायत दर्ज करना चाहते हैं?"
            else:
                msg_en = "It looks like there might be a drainage issue in the video. Would you like to register a complaint for drainage maintenance?"
                msg_mr = "व्हिडिओमध्ये ड्रेनेजची समस्या असावी असे दिसते. आपण ड्रेनेज दुरुस्तीसाठी तक्रार नोंदवू इच्छिता का?"
                msg_hi = "वीडियो में ड्रेनेज की समस्या प्रतीत होती है। क्या आप ड्रेनेज मरम्मत के लिए शिकायत दर्ज करना चाहते हैं?"
        elif cat in ["water_pipeline_leakages", "pipelinedefects"]:
            if conf_level == "high":
                msg_en = "I identified a water pipeline leakage issue in your video. Would you like to submit a complaint for water pipeline repair?"
                msg_mr = "आपल्या व्हिडिओमध्ये पाणी पाईपलाईन गळतीची समस्या दिसत आहे. आपण पाईपलाईन दुरुस्तीसाठी तक्रार नोंदवू इच्छिता का?"
                msg_hi = "आपके वीडियो में पानी पाइपलाइन लीकेज की समस्या दिखाई दे रही है। क्या आप पाइपलाइन मरम्मत के लिए शिकायत दर्ज करना चाहते हैं?"
            else:
                msg_en = "It looks like there might be a water pipeline leakage in the video. Would you like to register a complaint for water pipeline repair?"
                msg_mr = "व्हिडिओमध्ये पाणी पाईपलाईन गळती असावी असे दिसते. आपण पाईपलाईन दुरुस्तीसाठी तक्रार नोंदवू इच्छिता का?"
                msg_hi = "वीडियो में पानी पाइपलाइन लीकेज प्रतीत होता है। क्या आप पाइपलाइन मरम्मत के लिए शिकायत दर्ज करना चाहते हैं?"
        else:
            if conf_level == "high":
                msg_en = f"I identified a {cat_en} issue from your video. Would you like to submit a complaint for this issue?"
                msg_mr = f"आपल्या व्हिडिओमध्ये {cat_mr} संदर्भातील समस्या स्पष्ट दिसत आहे. आपण या समस्येसाठी तक्रार नोंदवू इच्छिता का?"
                msg_hi = f"आपके वीडियो में {cat_hi} की समस्या दिखाई दे रही है। क्या आप इस समस्या के लिए शिकायत दर्ज करना चाहते हैं?"
            else:
                msg_en = f"It looks like there might be a {cat_en} issue in the video. Would you like to register a complaint for this issue?"
                msg_mr = f"व्हिडिओमध्ये {cat_mr} ची समस्या असावी असे दिसते. आपण या समस्येसाठी तक्रार नोंदवू इच्छिता का?"
                msg_hi = f"वीडियो में {cat_hi} की समस्या प्रतीत होती है। क्या आप इस समस्या के लिए शिकायत दर्ज करना चाहते हैं?"

        text_by_lang = {"en": msg_en, "mr": msg_mr, "hi": msg_hi}
        reply = text_by_lang.get(lang, msg_en)

        buttons_by_lang = {
            "mr": [
                {"text": "होय, तक्रार नोंदवा", "action": "confirm_video_yes"},
                {"text": "नाही, दुसरी समस्या आहे", "action": "confirm_video_no"}
            ],
            "hi": [
                {"text": "हाँ, शिकायत दर्ज करें", "action": "confirm_video_yes"},
                {"text": "नहीं, दूसरी समस्या है", "action": "confirm_video_no"}
            ],
            "en": [
                {"text": "Yes, Register Complaint", "action": "confirm_video_yes"},
                {"text": "No, Different Complaint", "action": "confirm_video_no"}
            ]
        }
        buttons = buttons_by_lang.get(lang, buttons_by_lang["en"])
        return {"reply": reply, "buttons": buttons}

    def generate_video_multiple_issues_prompt(self, candidate_issues: List[str], lang: str = "en") -> Dict[str, Any]:
        cat_names_en = [CATEGORY_DISPLAY_NAMES.get(c, {}).get("en", c.replace('_', ' ').title()) for c in candidate_issues]
        cat_names_mr = [CATEGORY_DISPLAY_NAMES.get(c, {}).get("mr", c) for c in candidate_issues]
        cat_names_hi = [CATEGORY_DISPLAY_NAMES.get(c, {}).get("hi", c) for c in candidate_issues]

        if lang == "mr":
            bullet_list = "\n".join([f"• {name}" for name in cat_names_mr])
            reply = f"व्हिडिओमध्ये एकापेक्षा जास्त नागरी समस्या दिसत आहेत:\n{bullet_list}\n\nआपण यापैकी कोणत्या समस्येची तक्रार नोंदवू इच्छिता?"
        elif lang == "hi":
            bullet_list = "\n".join([f"• {name}" for name in cat_names_hi])
            reply = f"वीडियो में एक से अधिक नागरिक समस्याएं दिखाई दे रही हैं:\n{bullet_list}\n\nआप इनमें से किस समस्या की शिकायत दर्ज कराना चाहते हैं?"
        else:
            bullet_list = "\n".join([f"• {name}" for name in cat_names_en])
            reply = f"I noticed multiple possible civic issues in your video:\n{bullet_list}\n\nWhich one would you like to report?"

        buttons = []
        for c in candidate_issues:
            norm_c = normalize_category_key(c)
            c_label = CATEGORY_DISPLAY_NAMES.get(c, {}).get(lang, c.replace('_', ' ').title())
            buttons.append({"text": c_label, "action": f"select_video_candidate_{norm_c}"})

        return {"reply": reply, "buttons": buttons}

    def generate_video_low_confidence_prompt(self, lang: str = "en") -> str:
        if lang == "mr":
            return "व्हिडिओमध्ये नेमकी कोणती नागरी समस्या आहे हे स्पष्टपणे ओळखता आले नाही. कृपया आपण समस्येचे थोडक्यात वर्णन करू शकता का?"
        elif lang == "hi":
            return "वीडियो में नागरिक समस्या स्पष्ट रूप से पहचानी नहीं जा सकी। क्या आप कृपया समस्या का संक्षेप में विवरण दे सकते हैं?"
        return "I could not clearly identify the civic issue in the video. Could you please briefly describe the problem?"

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
        video_result: Optional[Any] = None,
        confirm_register: bool = False,
        action: Optional[str] = None,
        ward_number: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main conversation controller fulfilling WardMitra AI municipal requirements.
        """
        session_key = session_id or (citizen_phone if citizen_phone and citizen_phone != "Anonymous" else None) or f"anon_{uuid.uuid4().hex}"
        state = conversation_state_manager.get_or_create(session_key, citizen_phone)
        result = self._process_chat(
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
            state=state
        )
        if isinstance(result, dict):
            if "conversation_state" not in result:
                result["conversation_state"] = state.to_dict()
            if "session_id" not in result:
                result["session_id"] = state.session_id
        return result

    def _process_chat(
        self,
        db: Session,
        message: Optional[str],
        category: Optional[str],
        latitude: Optional[float],
        longitude: Optional[float],
        citizen_phone: Optional[str],
        photo_filename: Optional[str],
        photo_bytes: Optional[bytes],
        video_result: Optional[Any],
        confirm_register: bool,
        action: Optional[str],
        ward_number: Optional[int],
        state: ConversationState
    ) -> Dict[str, Any]:
        text = (message or "").strip()
        if not text and video_result and getattr(video_result, "detected_language", None):
            lang = video_result.detected_language
            state.language = lang
        elif text:
            detected_lang = self.detect_language(text)
            if detected_lang == "unclear":
                lang = "unclear"
            elif detected_lang:
                # Session Language Locking: If session has an established Indian language (mr or hi),
                # do not let a Latin-character address, proper noun, button text, or short answer switch language to "en"
                # unless user uses explicit English conversational question words / inquiry sentences
                if getattr(state, "language", None) in ["mr", "hi"] and detected_lang == "en":
                    english_sentence_markers = {
                        "what", "where", "how", "why", "who", "when", "tell", "explain",
                        "can you", "could you", "in english", "english please", "speak english"
                    }
                    t_lower = text.lower()
                    has_explicit_en_query = any(m in t_lower for m in english_sentence_markers)
                    if not has_explicit_en_query:
                        detected_lang = state.language
                state.language = detected_lang
                lang = detected_lang
            else:
                lang = state.language or "mr"
        else:
            lang = state.language or "mr"

        # If user asks how to register / procedure / step-by-step guidance, clear any category override
        if text and self.is_complaint_process_inquiry(text):
            category = None

        # Resolution Confirmation / Citizen Feedback / Reopening handling
        if action and (
            action.startswith("confirm_resolution_yes") or 
            action.startswith("confirm_resolution_no") or 
            action.startswith("rate_")
        ):
            if action.startswith("confirm_resolution_yes"):
                tid = action.replace("confirm_resolution_yes_", "").replace("confirm_resolution_yes", "").strip("_")
                target_complaint = db.query(Complaint).filter(Complaint.ticket_id == tid).first() if tid else None
                if not target_complaint and text:
                    found_tid = self.is_tracking_intent(text)
                    if found_tid and found_tid != "UNKNOWN":
                        target_complaint = db.query(Complaint).filter(Complaint.ticket_id == found_tid).first()
                if target_complaint:
                    feedback_service.confirm_resolution(db, target_complaint.ticket_id, confirmed=True)
                    if lang == "mr":
                        reply = (
                            f"🎉 **तक्रार क्र. {target_complaint.ticket_id} यशस्वीरीत्या बंद (Closed) करण्यात आली आहे!**\n\n"
                            "आपल्या प्रभागाला स्वच्छ आणि सुंदर ठेवण्यास सहकार्य केल्याबद्दल धन्यवाद. 🙏\n"
                            "महापालिकेच्या सेवेचे मूल्यांकन करण्यासाठी कृपया आपले १ ते ५ स्टार रेटिंग द्या:"
                        )
                    elif lang == "hi":
                        reply = (
                            f"🎉 **शिकायत क्र. {target_complaint.ticket_id} सफलतापूर्वक बंद (Closed) कर दी गई है!**\n\n"
                            "कृपया नगर निगम की सेवा के लिए अपना १ से ५ स्टार रेटिंग दें:"
                        )
                    else:
                        reply = (
                            f"🎉 **Grievance {target_complaint.ticket_id} has been successfully CLOSED!**\n\n"
                            "Thank you for helping keep PCMC clean and well-maintained. 🙏\n"
                            "Please rate your service experience from 1 to 5 stars:"
                        )
                    rating_buttons = [
                        {"text": "⭐ 1 Star", "action": f"rate_1_{target_complaint.ticket_id}"},
                        {"text": "⭐⭐ 2 Stars", "action": f"rate_2_{target_complaint.ticket_id}"},
                        {"text": "⭐⭐⭐ 3 Stars", "action": f"rate_3_{target_complaint.ticket_id}"},
                        {"text": "⭐⭐⭐⭐ 4 Stars", "action": f"rate_4_{target_complaint.ticket_id}"},
                        {"text": "⭐⭐⭐⭐⭐ 5 Stars", "action": f"rate_5_{target_complaint.ticket_id}"}
                    ]
                    return {
                        "reply": reply,
                        "intent": "RESOLUTION_CONFIRMED",
                        "language": lang,
                        "category": target_complaint.detected_category,
                        "category_name": target_complaint.detected_category,
                        "ticket_data": {"ticket_id": target_complaint.ticket_id, "status": "CLOSED"},
                        "action_prompt": "submit_rating",
                        "buttons": rating_buttons
                    }

            elif action.startswith("confirm_resolution_no"):
                tid = action.replace("confirm_resolution_no_", "").replace("confirm_resolution_no", "").strip("_")
                target_complaint = db.query(Complaint).filter(Complaint.ticket_id == tid).first() if tid else None
                if not target_complaint and text:
                    found_tid = self.is_tracking_intent(text)
                    if found_tid and found_tid != "UNKNOWN":
                        target_complaint = db.query(Complaint).filter(Complaint.ticket_id == found_tid).first()
                if target_complaint:
                    reopen_reason = text if (text and len(text.strip()) > 3 and not text.lower().startswith("नाही") and not text.lower().startswith("no")) else "Citizen reported problem still persists"
                    feedback_service.confirm_resolution(db, target_complaint.ticket_id, confirmed=False, reason=reopen_reason)
                    if lang == "mr":
                        reply = (
                            f"🔄 **तक्रार क्र. {target_complaint.ticket_id} पुन्हा उघडण्यात आली आहे (Reopened).**\n\n"
                            f"• पुनः उघडण्याची संख्या: {target_complaint.reopen_count}\n"
                            f"• कारण: {reopen_reason}\n\n"
                            "संबंधित क्षेत्रीय अधिकाऱ्यांना आणि पर्यवेक्षकांना तात्काळ पुन्हा कार्यवाही करण्याचे निर्देश दिले आहेत."
                        )
                    elif lang == "hi":
                        reply = (
                            f"🔄 **शिकायत क्र. {target_complaint.ticket_id} फिर से खोल दी गई है (Reopened)।**\n\n"
                            f"• पुनः खोलने की संख्या: {target_complaint.reopen_count}\n"
                            "संबंधित अधिकारियों को तत्काल पुनः कार्रवाई के लिए सूचित कर दिया गया है।"
                        )
                    else:
                        reply = (
                            f"🔄 **Grievance {target_complaint.ticket_id} has been REOPENED.**\n\n"
                            f"• Reopen count: {target_complaint.reopen_count}\n"
                            f"• Citizen reason: {reopen_reason}\n\n"
                            "The grievance has been re-assigned to the field staff and supervisor for immediate follow-up."
                        )
                    return {
                        "reply": reply,
                        "intent": "RESOLUTION_REJECTED_REOPENED",
                        "language": lang,
                        "category": target_complaint.detected_category,
                        "category_name": target_complaint.detected_category,
                        "ticket_data": {"ticket_id": target_complaint.ticket_id, "status": "REOPENED", "reopen_count": target_complaint.reopen_count},
                        "action_prompt": None
                    }

            elif action.startswith("rate_"):
                parts = action.split("_")
                rating_val = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
                tid = "_".join(parts[2:]) if len(parts) > 2 else None
                target_complaint = db.query(Complaint).filter(Complaint.ticket_id == tid).first() if tid else None
                if target_complaint:
                    feedback_service.submit_feedback(db, target_complaint.ticket_id, rating=rating_val, comments=text)
                    if lang == "mr":
                        reply = f"🙏 तक्रार क्र. {target_complaint.ticket_id} साठी {rating_val}/५ स्टार दिल्याबद्दल धन्यवाद! आपला अभिप्राय आमच्यासाठी अत्यंत मौल्यवान आहे."
                    elif lang == "hi":
                        reply = f"🙏 शिकायत क्र. {target_complaint.ticket_id} के लिए {rating_val}/५ स्टार देने के लिए धन्यवाद! आपकी प्रतिक्रिया हमारे लिए बहुत महत्वपूर्ण है।"
                    else:
                        reply = f"🙏 Thank you for rating grievance {target_complaint.ticket_id} with {rating_val}/5 stars! Your feedback helps us improve municipal services."
                    return {
                        "reply": reply,
                        "intent": "FEEDBACK_SUBMITTED",
                        "language": lang,
                        "category": target_complaint.detected_category,
                        "category_name": target_complaint.detected_category,
                        "ticket_data": {"ticket_id": target_complaint.ticket_id, "status": "CLOSED", "rating": rating_val},
                        "action_prompt": None
                    }

        # Step 2 (BUG 1, BUG 10): Intent Detection Layer
        intent_info = nlp_agent.classify_intent(
            text=text,
            state=state,
            action=action,
            has_photo=bool(photo_filename and photo_bytes),
            has_video=bool(video_result)
        )
        detected_intent = intent_info["intent"]
        logger.info(
            f"[ConversationalService] Session '{state.session_id}': intent='{detected_intent}', "
            f"category='{state.complaint_category}', locked={state.category_locked}, "
            f"confirmed={state.category_confirmed}, pending_field='{state.pending_field}'"
        )

        # Step 2.5: Tracking Requests (via Button Action, Ticket ID, or Status Query)
        target_track_tid = None
        is_track_requested = False

        if action and action.startswith("track_"):
            target_track_tid = action.replace("track_", "").strip()
            is_track_requested = True
        elif action == "track":
            is_track_requested = True
        elif text:
            tid_match = self.is_tracking_intent(text)
            if tid_match:
                is_track_requested = True
                if tid_match != "UNKNOWN":
                    target_track_tid = tid_match
            elif detected_intent == INTENT_STATUS_QUERY:
                is_track_requested = True
        elif detected_intent == INTENT_STATUS_QUERY:
            is_track_requested = True

        if is_track_requested:
            # Fallback 1: check target_track_tid from state if unknown
            if not target_track_tid or target_track_tid == "UNKNOWN":
                target_track_tid = getattr(state, "last_duplicate_ticket", None) or getattr(state, "active_ticket_id", None)
            
            # Fallback 2: check state history for recently mentioned ticket ID
            if not target_track_tid and getattr(state, "history", None):
                for turn in reversed(state.history):
                    m_hist = re.search(r'\bWM-\d{8}-\d{4}\b', str(turn.get("reply", "")) + " " + str(turn.get("message", "")))
                    if m_hist:
                        target_track_tid = m_hist.group(0)
                        break

            # Fallback 3: check DB for latest active complaint of citizen_phone
            if not target_track_tid and citizen_phone and citizen_phone != "Anonymous":
                latest_c = (
                    db.query(Complaint)
                    .filter(Complaint.citizen_phone == citizen_phone)
                    .order_by(Complaint.created_at.desc())
                    .first()
                )
                if latest_c:
                    target_track_tid = latest_c.ticket_id

            if target_track_tid and target_track_tid != "UNKNOWN":
                complaint = db.query(Complaint).filter(Complaint.ticket_id == target_track_tid).first()
                if complaint:
                    state.active_ticket_id = complaint.ticket_id
                    worker_str = complaint.assigned_worker_name or ("Er. संदीप माने" if lang == "mr" else "Er. Sandeep Mane")
                    worker_contact = f" ({complaint.assigned_worker_contact})" if complaint.assigned_worker_contact else ""
                    ward_display = f"प्रभाग {complaint.ward_number}" if complaint.ward_number else "निश्चित केले नाही"
                    ward_info = geo_agent.get_ward_by_number(complaint.ward_number) if complaint.ward_number else None
                    if ward_info:
                        ward_display = f"प्रभाग {ward_info['ward_number']}: {ward_info['ward_name']} ({ward_info['zone']})"

                    cat_disp = CATEGORY_DISPLAY_NAMES.get(complaint.detected_category, {}).get(lang, complaint.detected_category)
                    if lang == "mr":
                        reply = (
                            f"🔍 **तक्रार स्थिती: {complaint.ticket_id}**\n\n"
                            f"• **स्थिती:** {complaint.status.value}\n"
                            f"• **वर्ग / श्रेणी:** {cat_disp}\n"
                            f"• **प्रभाग:** {ward_display}\n"
                            f"• **नियुक्त क्षेत्रीय अभियंता:** {worker_str}{worker_contact}\n"
                            f"• **निकालाची मुदत (SLA):** {complaint.sla_hours} तास"
                        )
                    elif lang == "hi":
                        reply = (
                            f"🔍 **शिकायत स्थिति: {complaint.ticket_id}**\n\n"
                            f"• **स्थिति:** {complaint.status.value}\n"
                            f"• **श्रेणी:** {cat_disp}\n"
                            f"• **प्रभाग:** {ward_display}\n"
                            f"• **नियुक्त क्षेत्रीय इंजीनियर:** {worker_str}{worker_contact}\n"
                            f"• **निवारण समयसीमा (SLA):** {complaint.sla_hours} घंटे"
                        )
                    else:
                        reply = (
                            f"🔍 **Complaint Status: {complaint.ticket_id}**\n\n"
                            f"• **Status:** {complaint.status.value}\n"
                            f"• **Category:** {cat_disp}\n"
                            f"• **Ward:** {ward_display}\n"
                            f"• **Assigned Field Engineer:** {worker_str}{worker_contact}\n"
                            f"• **SLA Target:** {complaint.sla_hours} Hours"
                        )

                    buttons = None
                    action_prompt = None
                    intent = "TRACK_COMPLAINT"
                    if complaint.status.value in ["RESOLVED", "CITIZEN_CONFIRMATION"]:
                        intent = "CONFIRM_RESOLUTION_REQUIRED"
                        action_prompt = "confirm_resolution"
                        if lang == "mr":
                            reply += (
                                "\n\n✅ **क्षेत्रीय अधिकाऱ्यांकडून ही समस्या सोडवण्यात आल्याचे नोंदवले आहे.**\n"
                                "कृपया कामाची प्रत्यक्ष खात्री करा: समस्या खरोखर सुटली आहे का?"
                            )
                            buttons = [
                                {"text": "✅ होय, समस्या सुटली आहे (Yes, Resolved)", "action": f"confirm_resolution_yes_{complaint.ticket_id}"},
                                {"text": "❌ नाही, समस्या अद्याप आहे (No, Still Exists)", "action": f"confirm_resolution_no_{complaint.ticket_id}"}
                            ]
                        elif lang == "hi":
                            reply += (
                                "\n\n✅ **क्षेत्रीय अधिकारी द्वारा यह समस्या हल कर दी गई है।**\n"
                                "कृपया पुष्टि करें: क्या समस्या हल हो चुकी है?"
                            )
                            buttons = [
                                {"text": "✅ हाँ, समस्या हल हो गई है", "action": f"confirm_resolution_yes_{complaint.ticket_id}"},
                                {"text": "❌ नहीं, समस्या अभी भी है", "action": f"confirm_resolution_no_{complaint.ticket_id}"}
                            ]
                        else:
                            reply += (
                                "\n\n✅ **Field staff have marked this grievance as RESOLVED.**\n"
                                "Please verify and confirm: Has the problem been resolved?"
                            )
                            buttons = [
                                {"text": "✅ Yes, Problem Resolved", "action": f"confirm_resolution_yes_{complaint.ticket_id}"},
                                {"text": "❌ No, Problem Still Exists", "action": f"confirm_resolution_no_{complaint.ticket_id}"}
                            ]

                    result_dict = {
                        "reply": reply,
                        "intent": intent,
                        "language": lang,
                        "category": complaint.detected_category,
                        "category_name": cat_disp,
                        "ticket_data": {
                            "ticket_id": complaint.ticket_id,
                            "status": complaint.status.value,
                            "ward": complaint.ward_number,
                            "assigned_worker_name": worker_str,
                            "assigned_worker_contact": complaint.assigned_worker_contact or "020-67333333"
                        },
                        "action_prompt": action_prompt,
                        "conversation_state": state.to_dict()
                    }
                    if buttons:
                        result_dict["buttons"] = buttons
                    return result_dict
                else:
                    if lang == "mr":
                        msg = f"⚠️ तक्रार क्र. '{target_track_tid}' सिस्टीममध्ये सापडली नाही. कृपया योग्य तिकीट नंबर तपासा."
                    elif lang == "hi":
                        msg = f"⚠️ शिकायत क्र. '{target_track_tid}' सिस्टम में नहीं मिली। कृपया सही टिकट नंबर जांचें।"
                    else:
                        msg = f"⚠️ Ticket '{target_track_tid}' was not found. Please double-check your ticket number."
                    return {
                        "reply": msg,
                        "intent": "TRACK_COMPLAINT",
                        "language": lang,
                        "category": None,
                        "category_name": None,
                        "ticket_data": None,
                        "action_prompt": "specify_ticket_id",
                        "conversation_state": state.to_dict()
                    }
            else:
                # No ticket found to track - ask citizen politely for Ticket ID without starting intake
                if lang == "mr":
                    msg = "🔍 कृपया आपला तक्रार क्रमांक (उदा. **WM-20260922-0001**) सांगा, जेणेकरून मी लगेच प्रगती व स्थिती तपासेन."
                elif lang == "hi":
                    msg = "🔍 कृपया अपना शिकायत क्रमांक (उदा. **WM-20260922-0001**) बताएं, ताकि मैं तुरंत स्थिति की जांच कर सकूं।"
                else:
                    msg = "🔍 Please provide your Ticket ID (e.g. **WM-20260922-0001**) to check the latest resolution status."
                return {
                    "reply": msg,
                    "intent": "TRACK_COMPLAINT",
                    "language": lang,
                    "category": None,
                    "category_name": None,
                    "ticket_data": None,
                    "action_prompt": "specify_ticket_id",
                    "conversation_state": state.to_dict()
                }

        # Step 3 (BUG 6): Pre-Moderation & Casual Abuse Handling
        # Runs before complaint processing or continuing questions
        if detected_intent == INTENT_ABUSIVE_MESSAGE:
            state.clear_pending_field()
            if lang == "mr":
                reply = "मी नागरी तक्रार निवारणात मदत करू शकतो. कृपया आपण नोंदवू इच्छित असलेली नागरी समस्या सांगा."
            elif lang == "hi":
                reply = "मैं नागरिक शिकायतों के निवारण में सहायता कर सकता हूँ। कृपया वह नागरिक समस्या बताएं जिसे आप दर्ज करना चाहते हैं।"
            else:
                reply = "I can help with civic complaints. Please describe the civic issue you want to report."
            return {
                "reply": reply,
                "intent": "ABUSIVE_MESSAGE",
                "language": lang,
                "category": None,
                "category_name": None,
                "ticket_data": None,
                "action_prompt": "describe_civic_issue",
                "conversation_state": state.to_dict()
            }

        # 0. Unclear Language Handling (Requirement 3 & Test 5)
        if (
            lang == "unclear"
            and text
            and not self.is_greeting(text)
            and not self.is_cancel_intent(text)
            and not self.is_tracking_intent(text)
            and not self.is_complaint_process_inquiry(text)
            and not video_result
            and state.complaint_status != "awaiting_confirmation"
        ):
            reply = "Please describe your complaint in English or Marathi. / कृपया आपली तक्रार इंग्रजी किंवा मराठीत सांगा."
            return {
                "reply": reply,
                "intent": "LANGUAGE_CLARIFICATION_REQUIRED",
                "language": "unclear",
                "category": None,
                "category_name": None,
                "ticket_data": None,
                "action_prompt": "specify_language"
            }

        # Step 3.5 (BUG 9): Unrelated / General Non-Civic Message Handling
        if detected_intent == INTENT_UNRELATED and not photo_bytes and not video_result and not state.pending_field:
            if not self.is_greeting(text) and not self.is_smalltalk_or_gratitude(text) and not self.is_complaint_process_inquiry(text) and not self.is_bot_inquiry_or_help(text) and not is_ambiguous_light_complaint(text) and not human_persona_service.find_knowledge_answer(text, lang):
                if state.category_locked and state.complaint_category:
                    cat_c = state.confirmed_category or state.complaint_category
                    cat_disp = CATEGORY_DISPLAY_NAMES.get(cat_c, {}).get(lang, cat_c.replace('_', ' ').title())
                    if lang == "mr":
                        reply = f"मी पिंपरी चिंचवड नागरी तक्रार निवारण सहाय्यक आहे. आपण **{cat_disp}** समस्येबद्दल बोलत होता. कृपया या समस्येचे ठिकाण सांगा किंवा नवीन तक्रार सुरू करायची असल्यास सांगा."
                    elif lang == "hi":
                        reply = f"मैं नागरिक शिकायत निवारण सहायक हूँ। हम आपकी **{cat_disp}** समस्या पर चर्चा कर रहे थे। कृपया इसका स्थान बताएं या यदि आप रद्द करना चाहते हैं तो बताएं।"
                    else:
                        reply = f"I am the PCMC WardMitra civic assistant. We were discussing your **{cat_disp}** issue. Please provide the location, or let me know if you would like to report something else."
                    if state.location:
                        state.set_pending_field("registration_confirmation")
                        state.complaint_status = "ready_for_submission"
                        action_prompt = "confirm_register"
                    else:
                        state.set_pending_field("location")
                        state.complaint_status = "collecting_information"
                        action_prompt = "provide_location"

                    return {
                        "reply": reply,
                        "intent": "UNRELATED_MESSAGE",
                        "language": lang,
                        "category": cat_c,
                        "category_name": cat_disp,
                        "ticket_data": None,
                        "action_prompt": action_prompt,
                        "conversation_state": state.to_dict()
                    }
                else:
                    if lang == "mr":
                        reply = "मी पिंपरी चिंचवड परिसरातील खड्डे, कचरा, पथदिवे, ड्रेनेज, पाणीपुरवठा अशा नागरी तक्रारी नोंदवण्यात मदत करू शकतो. कृपया आपली नागरी समस्या सांगा."
                    elif lang == "hi":
                        reply = "मैं पिंपरी चिंचवड क्षेत्र में गड्ढे, कचरा, स्ट्रीट लाइट, ड्रेनेज, पानी जैसी नागरिक शिकायतों में सहायता कर सकता हूँ। कृपया अपनी नागरिक समस्या बताएं।"
                    else:
                        reply = "I am the PCMC WardMitra civic grievance assistant. I can help you report issues like potholes, garbage, streetlights, drainage, or water leaks. Please describe the civic issue you are facing."
                    return {
                        "reply": reply,
                        "intent": "UNRELATED_MESSAGE",
                        "language": lang,
                        "category": None,
                        "category_name": None,
                        "ticket_data": None,
                        "action_prompt": "describe_civic_issue",
                        "conversation_state": state.to_dict()
                    }

        # Step 3.8: Explicit Start New Complaint Request (e.g. "नवीन तक्रार नोंदवा", "दुसरी तक्रार", "New Complaint")
        if detected_intent == INTENT_START_NEW_COMPLAINT or action in ["start_new_complaint", "new_complaint"]:
            state.reset_after_registration()
            if lang == "mr":
                reply = (
                    "होय नक्कीच! आपली नवीन समस्या कोणती आहे?\n\n"
                    "उदा. 💡 पथदिवा, 🕳️ खड्डा, 🗑️ कचरा, 🚰 पाणी गळती किंवा 🌊 ड्रेनेज.\n"
                    "कृपया समस्येचे वर्णन सांगा किंवा खालीलपैकी एक बटण निवडा:"
                )
            elif lang == "hi":
                reply = (
                    "जी बिल्कुल! आपकी नई समस्या क्या है?\n\n"
                    "उदा. 💡 स्ट्रीट लाइट, 🕳️ गड्ढा, 🗑️ कचरा, 🚰 पानी लीकेज या 🌊 ड्रेनेज।\n"
                    "कृपया समस्या का विवरण बताएं या नीचे दिए गए बटन को चुनें:"
                )
            else:
                reply = (
                    "Sure! What is your new civic issue?\n\n"
                    "e.g. 💡 Streetlight, 🕳️ Pothole, 🗑️ Garbage, 🚰 Water Leakage, or 🌊 Drainage.\n"
                    "Please describe your issue or select one of the buttons below:"
                )
            buttons = [
                {"text": "💡 पथदिवा / Streetlight", "action": "select_category_streetlight"},
                {"text": "🕳️ खड्डा / Pothole", "action": "select_category_pothole"},
                {"text": "🗑️ कचरा / Garbage", "action": "select_category_garbage"},
                {"text": "🚰 पाणीपुरवठा / Water", "action": "select_category_pipeline_water_leakage"},
                {"text": "🌊 ड्रेनेज / Drainage", "action": "select_category_drainage"}
            ]
            return {
                "reply": reply,
                "intent": "START_NEW_COMPLAINT",
                "language": lang,
                "category": None,
                "category_name": None,
                "ticket_data": None,
                "buttons": buttons,
                "action_prompt": "choose_category_or_describe",
                "conversation_state": state.to_dict()
            }

        # Step 3.9: Civic Knowledge Base & Informational Inquiry Priority
        # If user asks an informational question/inquiry, provide guidance via Ward RAG / PCMC knowledge base first
        from app.services.rag_service import is_knowledge_inquiry
        if (
            (detected_intent == "CIVIC_KNOWLEDGE_INQUIRY" or (text and is_knowledge_inquiry(text)))
            and not photo_bytes
            and not video_result
            and not self.is_bot_inquiry_or_help(text)
            and not self.is_complaint_process_inquiry(text)
            and not self.is_greeting(text)
            and not self.is_smalltalk_or_gratitude(text)
        ):
            kb_answer = human_persona_service.find_knowledge_answer(text, lang)
            if kb_answer:
                return {
                    "reply": kb_answer,
                    "intent": "CIVIC_KNOWLEDGE_INQUIRY",
                    "language": lang,
                    "category": None,
                    "category_name": None,
                    "ticket_data": None,
                    "action_prompt": "ask_more_or_register",
                    "conversation_state": state.to_dict()
                }

        # Step 4 (BUG 5): Category Correction
        # If user explicitly corrects category (e.g. 'no, my complaint is about potholes')
        if detected_intent == INTENT_CATEGORY_CORRECTION or (text and nlp_agent.extract_category_correction(text)):
            corr_cat = intent_info.get("category") or nlp_agent.extract_category_correction(text)
            if corr_cat:
                state.correct_category(corr_cat, source="correction")
                cat_info = CATEGORY_DISPLAY_NAMES.get(corr_cat, {"en": corr_cat.replace('_', ' ').title(), "mr": corr_cat, "hi": corr_cat})
                cat_disp = cat_info.get(lang, cat_info["en"])
                
                if not state.location:
                    state.set_pending_field("location")
                    if lang == "mr":
                        reply = f"समजले! मी तक्रार **{cat_disp}** मध्ये बदलली आहे. कृपया या समस्येचे अंदाजे ठिकाण, पत्ता किंवा जवळची खूण सांगा."
                    elif lang == "hi":
                        reply = f"समझ गया! मैंने शिकायत **{cat_disp}** में अपडेट कर दी है। कृपया इसका स्थान, पता या नजदीकी लैंडमार्क बताएं।"
                    else:
                        reply = f"Understood! I have updated your complaint to **{cat_disp}**. Please provide the approximate location, address, or nearest landmark for this issue."
                    return {
                        "reply": reply,
                        "intent": "CATEGORY_CORRECTION",
                        "language": lang,
                        "category": corr_cat,
                        "category_name": cat_disp,
                        "ticket_data": None,
                        "action_prompt": "provide_location",
                        "conversation_state": state.to_dict()
                    }
                else:
                    state.set_pending_field("registration_confirmation")
                    state.complaint_status = "ready_for_submission"
                    if lang == "mr":
                        reply = f"मी आपली तक्रार **{cat_disp}** (ठिकाण: **{state.location}**) अशी अद्यतनित केली आहे. मी ही तक्रार दाखल करू का?"
                    elif lang == "hi":
                        reply = f"मैंने आपकी शिकायत **{cat_disp}** (स्थान: **{state.location}**) के रूप में अपडेट कर दी है। क्या मैं यह शिकायत दर्ज करूँ?"
                    else:
                        reply = f"I have updated your complaint to **{cat_disp}** at **{state.location}**. Would you like me to register this complaint now?"
                    return {
                        "reply": reply,
                        "intent": "CATEGORY_CORRECTION",
                        "language": lang,
                        "category": corr_cat,
                        "category_name": cat_disp,
                        "ticket_data": None,
                        "action_prompt": "confirm_register",
                        "conversation_state": state.to_dict()
                    }

        # Step 4.5 (BUG 7 & BUG 8): Awaiting Candidate Selection Turn (via Text or Action)
        if (state.complaint_status == "awaiting_selection" or state.pending_field == "category") and text:
            candidates = state.candidate_categories or (state.video.candidate_issues if hasattr(state, "video") else [])
            t_lower = text.lower().strip()
            from app.agents.nlp_agent import CORRECTION_CATEGORY_KEYWORDS
            for cand in candidates:
                cand_norm = normalize_category_key(cand)
                if cand_norm in t_lower or cand.lower() in t_lower or any(kw in t_lower for kw in CORRECTION_CATEGORY_KEYWORDS.get(cand_norm, [])):
                    chosen_cat = cand_norm
                    state.lock_category(chosen_cat, source="user_selection")
                    state.candidate_categories = []
                    state.clear_pending_field()
                    if state.video.uploaded and state.video.has_multiple_issues:
                        state.video.detected_category = chosen_cat
                        state.video.has_multiple_issues = False
                        state.complaint_status = "awaiting_confirmation"
                        conf_info = self.generate_video_confirmation_prompt(state.video, lang=lang)
                        cat_display = CATEGORY_DISPLAY_NAMES.get(chosen_cat, {}).get(lang, chosen_cat.title())
                        return {
                            "reply": conf_info["reply"],
                            "intent": "VIDEO_CONFIRMATION_REQUIRED",
                            "language": lang,
                            "category": chosen_cat,
                            "category_name": cat_display,
                            "ticket_data": None,
                            "action_prompt": "confirm_video_complaint",
                            "buttons": conf_info["buttons"],
                            "conversation_state": state.to_dict()
                        }
                    else:
                        state.complaint_status = "collecting_information"
                        state.set_pending_field("location")
                        cat_display = CATEGORY_DISPLAY_NAMES.get(chosen_cat, {}).get(lang, chosen_cat.title())
                        if lang == "mr":
                            reply = f"उत्तम! आपण **{cat_display}** समस्या निवडली आहे. कृपया या समस्येचे अंदाजे ठिकाण, पत्ता किंवा जवळची खूण सांगा."
                        elif lang == "hi":
                            reply = f"बहुत अच्छा! आपने **{cat_display}** समस्या चुनी है। कृपया इसका अनुमानित स्थान या नजदीकी लैंडमार्क बताएं।"
                        else:
                            reply = f"Great! You selected **{cat_display}**. Please provide the approximate location, street name, or nearest landmark."
                        return {
                            "reply": reply,
                            "intent": "CATEGORY_CONFIRMATION",
                            "language": lang,
                            "category": chosen_cat,
                            "category_name": cat_display,
                            "ticket_data": None,
                            "action_prompt": "provide_location",
                            "conversation_state": state.to_dict()
                        }

        # Step 4.51 (BUG 3): Pending Cancel Confirmation Turn
        if state.pending_field == "cancel_confirmation" and not photo_bytes and not video_result:
            t_clean = (text or "").lower().strip()
            t_words = set(re.findall(r'[\w\u0900-\u097F]+', t_clean))
            is_cancel_neg = any(w in {"no", "nope", "keep", "नाही", "नको", "nahi", "nako"} for w in t_words) or any(p in t_clean for p in ["don't", "dont", "do not"]) or action in ["confirm_cancel_no", "no"]
            is_cancel_aff = not is_cancel_neg and (
                detected_intent == "CANCEL_CONFIRMED" or
                action in ["confirm_cancel_yes", "yes"] or
                any(w in {"yes", "yeah", "y", "sure", "ok", "okay", "cancel", "रद्द", "करा", "हो", "होय"} for w in t_words) or
                nlp_agent.is_affirmative_yes(text)
            )
            if is_cancel_aff:
                target_ticket = state.pending_cancel_ticket
                complaint = None
                if target_ticket:
                    complaint = db.query(Complaint).filter(Complaint.ticket_id == target_ticket).first()
                if not complaint and citizen_phone and citizen_phone != "Anonymous":
                    complaint = (
                        db.query(Complaint)
                        .filter(Complaint.citizen_phone == citizen_phone)
                        .filter(Complaint.status != ComplaintStatus.CANCELLED)
                        .order_by(Complaint.created_at.desc())
                        .first()
                    )

                if complaint:
                    if complaint.status == ComplaintStatus.CANCELLED:
                        if lang == "mr":
                            reply = f"ℹ️ तक्रार क्र. **{complaint.ticket_id}** आधीच रद्द केलेली आहे."
                        elif lang == "hi":
                            reply = f"ℹ️ शिकायत क्र. **{complaint.ticket_id}** पहले ही रद्द कर दी गई है।"
                        else:
                            reply = f"ℹ️ Complaint **{complaint.ticket_id}** is already cancelled."
                    else:
                        complaint_status_service.transition_status(
                            db=db,
                            complaint=complaint,
                            new_status=ComplaintStatus.CANCELLED,
                            changed_by=f"CITIZEN:{citizen_phone or 'Anonymous'}",
                            reason="Cancelled by citizen request via conversational chat",
                            notify=True
                        )
                        if lang == "mr":
                            reply = (
                                f"🛑 **आपली तक्रार क्र. {complaint.ticket_id} यशस्वीरीत्या रद्द (Cancelled) करण्यात आली आहे.**\n\n"
                                f"• **विषय:** {complaint.detected_category}\n"
                                f"• **नोंद:** नागरिकांच्या विनंतीनुसार तक्रार रद्द करण्यात आली.\n\n"
                                "आपल्याला भविष्यात कोणत्याही मदतीची गरज भासल्यास वॉर्डमित्र सदैव आपल्या सेवेत आहे! 🙏"
                            )
                        elif lang == "hi":
                            reply = (
                                f"🛑 **आपकी शिकायत क्र. {complaint.ticket_id} सफलतापूर्वक रद्द (Cancelled) कर दी गई है।**\n\n"
                                f"• **श्रेणी:** {complaint.detected_category}\n"
                                "भविष्य में किसी भी नागरिक सेवा के लिए वार्डमित्र सदैव उपलब्ध है!"
                            )
                        else:
                            reply = (
                                f"🛑 **Your complaint {complaint.ticket_id} has been successfully cancelled.**\n\n"
                                f"• **Category:** {complaint.detected_category}\n"
                                f"• **Status:** CANCELLED\n\n"
                                "Please let me know if you need assistance with any other civic grievance! 🙏"
                            )
                    state.pending_cancel_ticket = None
                    state.clear_pending_field()
                    state.complaint_status = "idle"
                    return {
                        "reply": reply,
                        "intent": "COMPLAINT_CANCELLED",
                        "language": lang,
                        "category": complaint.detected_category if complaint else None,
                        "category_name": complaint.detected_category if complaint else None,
                        "ticket_data": {"ticket_id": complaint.ticket_id, "status": "CANCELLED"} if complaint else None,
                        "action_prompt": "describe_civic_issue",
                        "conversation_state": state.to_dict()
                    }
                else:
                    state.pending_cancel_ticket = None
                    state.clear_pending_field()
                    state.complaint_status = "idle"
                    if lang == "mr":
                        reply = "माफ करा, रद्द करण्यासाठी कोणतीही सक्रिय तक्रार आढळली नाही."
                    elif lang == "hi":
                        reply = "क्षमा करें, रद्द करने के लिए कोई सक्रिय शिकायत नहीं मिली।"
                    else:
                        reply = "Sorry, no active complaint was found to cancel."
                    return {
                        "reply": reply,
                        "intent": "COMPLAINT_NOT_FOUND",
                        "language": lang,
                        "category": None,
                        "category_name": None,
                        "ticket_data": None,
                        "action_prompt": "describe_civic_issue",
                        "conversation_state": state.to_dict()
                    }
            elif is_cancel_neg or (not is_cancel_aff and nlp_agent.is_negative_no(text)) or action in ["confirm_cancel_no", "no"]:
                t_id = state.pending_cancel_ticket or ""
                state.pending_cancel_ticket = None
                state.clear_pending_field()
                state.complaint_status = "idle"
                if lang == "mr":
                    reply = f"समजले! तक्रार {t_id} रद्द केलेली नाही. ती पूर्ववत सक्रिय राहील."
                elif lang == "hi":
                    reply = f"समझ गया! शिकायत {t_id} रद्द नहीं की गई है। यह सक्रिय रहेगी।"
                else:
                    reply = f"Understood. Complaint {t_id} has not been cancelled and remains active."
                return {
                    "reply": reply,
                    "intent": "CANCELLATION_ABORTED",
                    "language": lang,
                    "category": None,
                    "category_name": None,
                    "ticket_data": None,
                    "action_prompt": "describe_civic_issue",
                    "conversation_state": state.to_dict()
                }

        # Step 4.52 (BUG 2): Image Category Confirmation Turn
        if state.pending_field == "image_category_confirmation" and not photo_bytes and not video_result:
            if detected_intent == INTENT_CATEGORY_CONFIRMATION or nlp_agent.is_affirmative_yes(text) or action in ["confirm_image_category_yes", "yes"]:
                state.category_confirmed = True
                state.category_source = "image"
                cat_c = state.complaint_category or "pothole"
                state.lock_category(cat_c, source="image")
                cat_info = CATEGORY_DISPLAY_NAMES.get(cat_c, {"en": cat_c.replace('_', ' ').title(), "mr": cat_c, "hi": cat_c})
                cat_disp = cat_info.get(lang, cat_info["en"])
                turn_seed = len(getattr(state, "history", [])) or getattr(state, "turn_count", 0)

                if not state.location:
                    state.set_pending_field("location")
                    state.complaint_status = "collecting_information"
                    reply = human_persona_service.generate_location_prompt(
                        category_name=cat_disp,
                        lang=lang,
                        turn_seed=turn_seed
                    )
                    return {
                        "reply": reply,
                        "intent": "CATEGORY_CONFIRMED",
                        "language": lang,
                        "category": cat_c,
                        "category_name": cat_disp,
                        "ticket_data": None,
                        "action_prompt": "provide_location",
                        "conversation_state": state.to_dict()
                    }
                else:
                    state.set_pending_field("registration_confirmation")
                    state.complaint_status = "ready_for_submission"
                    reply = human_persona_service.generate_registration_confirmation_prompt(
                        category_name=cat_disp,
                        location=state.location,
                        lang=lang,
                        turn_seed=turn_seed
                    )
                    buttons = [
                        {"text": "✅ Yes, Register / हो, नोंदवा", "action": "register"},
                        {"text": "❌ No / नाही", "action": "cancel"}
                    ]
                    return {
                        "reply": reply,
                        "intent": "CONVERSATIONAL",
                        "language": lang,
                        "category": cat_c,
                        "category_name": cat_disp,
                        "ticket_data": None,
                        "action_prompt": "confirm_register",
                        "buttons": buttons,
                        "conversation_state": state.to_dict()
                    }
            elif detected_intent in [INTENT_CATEGORY_CORRECTION, INTENT_REGISTRATION_REJECTION] or nlp_agent.is_negative_no(text) or action in ["confirm_image_category_no", "no"]:
                state.category_confirmed = False
                state.category_locked = False
                state.complaint_category = None
                corr_cat = intent_info.get("category") or nlp_agent.extract_category_correction(text)
                if corr_cat:
                    state.correct_category(corr_cat, source="correction")
                    cat_info = CATEGORY_DISPLAY_NAMES.get(corr_cat, {"en": corr_cat.replace('_', ' ').title(), "mr": corr_cat, "hi": corr_cat})
                    cat_disp = cat_info.get(lang, cat_info["en"])
                    state.set_pending_field("location")
                    state.complaint_status = "collecting_information"
                    if lang == "mr":
                        reply = f"समजले! मी तक्रार **{cat_disp}** मध्ये बदलली आहे. कृपया या समस्येचे अंदाजे ठिकाण, पत्ता किंवा जवळची खूण सांगा."
                    elif lang == "hi":
                        reply = f"समझ गया! मैंने शिकायत को **{cat_disp}** में बदल दिया है। कृपया इस समस्या का अनुमानित स्थान, पता या नजदीकी लैंडमार्क बताएं।"
                    else:
                        reply = f"Understood! I have updated the complaint to **{cat_disp}**. Please provide the approximate location, address, or landmark for this issue."
                    return {
                        "reply": reply,
                        "intent": "CATEGORY_CORRECTION",
                        "language": lang,
                        "category": corr_cat,
                        "category_name": cat_disp,
                        "ticket_data": None,
                        "action_prompt": "specify_location",
                        "conversation_state": state.to_dict()
                    }
                else:
                    state.set_pending_field("category")
                    state.complaint_status = "awaiting_selection"
                    if lang == "mr":
                        reply = "समजले. मग आपली तक्रार नक्की कोणत्या विषयाबद्दल आहे? कृपया योग्य श्रेणी सांगा (उदा. खड्डे, पथदिवे, कचरा, पाणीपुरवठा)."
                    elif lang == "hi":
                        reply = "समझ गया। तो आपकी शिकायत किस विषय के बारे में है? कृपया श्रेणी बताएं (जैसे गड्ढे, स्ट्रीटलाइट, कचरा, जलापूर्ति)।"
                    else:
                        reply = "Understood. What is your complaint regarding? Please describe the issue or specify the category (e.g., potholes, streetlights, garbage, water supply)."
                    return {
                        "reply": reply,
                        "intent": "CATEGORY_REJECTED",
                        "language": lang,
                        "category": None,
                        "category_name": None,
                        "ticket_data": None,
                        "action_prompt": "select_or_describe",
                        "conversation_state": state.to_dict()
                    }

        if text and self.is_greeting(text):
            name_token = self.extract_greeting_name(text)
            if lang == "mr":
                salutation = f"नमस्कार {name_token}! 👋" if name_token else "नमस्कार! 👋"
                reply = (
                    f"{salutation} मी वॉर्डमित्र (WardMitra AI) सहाय्यक आहे.\n\n"
                    "मी प्रभागातील रस्ते, पथदिवे, कचरा, ड्रेनेज, पाणी पुरवठा अशा नागरी समस्या सोडवण्यात मदत करतो. "
                    "मी आज आपली काय मदत करू शकतो?"
                )
            else:
                salutation = f"Hello {name_token}! 👋" if name_token else "Hello! 👋"
                reply = (
                    f"{salutation} Welcome to WardMitra AI.\n\n"
                    "I can help you report and track civic issues such as potholes, streetlights, garbage, drainage, or water supply in your ward. "
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

        # 1.1 Smalltalk / Gratitude
        if text and self.is_smalltalk_or_gratitude(text) and not confirm_register:
            if lang == "mr":
                reply = "आपले स्वागत आहे! प्रभागातील कोणत्याही नागरी समस्येसाठी कधीही संपर्क करा. वॉर्डमित्र सदैव सेवेत आहे."
            else:
                reply = "You're welcome! Feel free to reach out anytime if you face any civic issues like streetlights, potholes, garbage, or water leaks in your ward. Have a great day!"
            return {
                "reply": reply,
                "intent": "SMALLTALK",
                "language": lang,
                "category": None,
                "category_name": None,
                "ticket_data": None,
                "action_prompt": None
            }

        # 1.2 Bot Identity & Capabilities Inquiry
        if text and self.is_bot_inquiry_or_help(text):
            if lang == "mr":
                reply = (
                    "🤖 मी **वॉर्डमित्र AI (WardMitra)** आहे.\n\n"
                    "मी वॉर्डमित्र प्रकल्पांतर्गत प्रभाग पातळीवरील नागरी तक्रार निवारण सहाय्यक आहे. मी खालील समस्या सोडवण्यात मदत करतो:\n"
                    "• 💡 **पथदिवे**: बंद किंवा तुटलेले पथदिवे\n"
                    "• 🕳️ **खड्डे**: रस्त्यावरील खड्डे व नुकसान\n"
                    "• 🗑️ **कचरा**: कचऱ्याचे ढीग व अस्वच्छता\n"
                    "• 🚰 **पाणीपुरवठा**: पाईपलाईन गळती व अनियमित पाणी\n"
                    "• 🚾 **ड्रेनेज**: तुंबलेली गटारे व मॅनहोल\n\n"
                    "आपण थेट समस्येचा संदेश टाईप करू शकता, खालील बटण निवडू शकता किंवा कॅमेरा आयकॉनने फोटो जोडू शकता!"
                )
            else:
                reply = (
                    "🤖 I am **WardMitra AI**.\n\n"
                    "I am the dedicated ward civic grievance assistant for the WardMitra project. I can assist you with:\n"
                    "• 💡 **Streetlights**: Non-functional or flickering lights\n"
                    "• 🕳️ **Potholes & Roads**: Road damage and potholes\n"
                    "• 🗑️ **Garbage**: Uncollected waste and overflowing bins\n"
                    "• 🚰 **Water Supply**: Pipeline leakages and water issues\n"
                    "• 🚾 **Drainage**: Blocked gutters and open manholes\n\n"
                )
            return {
                "reply": reply,
                "intent": "SERVICE_INQUIRY",
                "language": lang,
                "category": None,
                "category_name": None,
                "ticket_data": None,
                "action_prompt": "describe_problem",
                "conversation_state": state.to_dict()
            }

        if detected_intent == INTENT_REGISTRATION_CONFIRMATION or nlp_agent.is_affirmative_yes(text) or action in ["register", "register_new", "force_register"]:
            confirm_register = True
        elif (
            (state.pending_field == "registration_confirmation" or state.complaint_status == "ready_for_submission") and
            (
                detected_intent == INTENT_REGISTRATION_REJECTION or
                action in ["cancel", "cancel_register", "confirm_register_no", "no"] or
                (text and nlp_agent.is_negative_no(text))
            )
        ):
            state.reset()
            if lang == "mr":
                reply = (
                    "🛑 **तक्रार नोंदणी रद्द करण्यात आली आहे.**\n\n"
                    "आपल्या संमतीशिवाय कोणतीही तक्रार नोंदवली जात नाही. "
                    "आपल्याला भविष्यात कोणतीही नागरी समस्या असल्यास किंवा इतर मदत हवी असल्यास वॉर्डमित्र सदैव आपल्या सेवेत आहे! 🙏"
                )
            elif lang == "hi":
                reply = (
                    "🛑 **शिकायत दर्ज करना रद्द कर दिया गया है।**\n\n"
                    "आपकी सहमति के बिना कोई शिकायत दर्ज नहीं की जाती। "
                    "यदि आपको कोई अन्य समस्या हो या सहायता चाहिए, तो वार्डमित्र सदैव आपकी सेवा में है! 🙏"
                )
            else:
                reply = (
                    "🛑 **Complaint registration has been cancelled.**\n\n"
                    "No complaint is submitted without your confirmation. "
                    "If you need any other civic assistance in the future, WardMitra is always at your service! 🙏"
                )
            return {
                "reply": reply,
                "intent": "REGISTRATION_CANCELLED",
                "language": lang,
                "category": None,
                "category_name": None,
                "ticket_data": None,
                "action_prompt": "describe_civic_issue",
                "conversation_state": state.to_dict()
            }

        # A. Video Ingestion Turn Handling
        if video_result:
            if getattr(video_result, "detected_language", None) and (not lang or lang == "unclear"):
                lang = video_result.detected_language
                state.language = lang
            state.on_video_received(video_result)

            # Multiple candidate issues check
            if getattr(video_result, "has_multiple_issues", False) or len(getattr(video_result, "candidate_issues", [])) > 1:
                state.complaint_status = "awaiting_selection"
                multi_info = self.generate_video_multiple_issues_prompt(video_result.candidate_issues, lang=lang)
                return {
                    "reply": multi_info["reply"],
                    "intent": "MULTIPLE_ISSUES_DETECTED",
                    "language": lang,
                    "category": None,
                    "category_name": None,
                    "ticket_data": None,
                    "action_prompt": "select_candidate_issue",
                    "buttons": multi_info["buttons"],
                    "conversation_state": state.to_dict()
                }

            # Low confidence check (< 0.40 or level == "low")
            if getattr(video_result, "confidence_level", "high") == "low" or getattr(video_result, "confidence", 0.0) < 0.40:
                low_conf_msg = self.generate_video_low_confidence_prompt(lang=lang)
                state.complaint_status = "collecting_information"
                state.record_question_asked("description")
                return {
                    "reply": low_conf_msg,
                    "intent": "LOW_CONFIDENCE_VIDEO",
                    "language": lang,
                    "category": None,
                    "category_name": None,
                    "ticket_data": None,
                    "action_prompt": "describe_problem",
                    "conversation_state": state.to_dict()
                }

            # High or Medium confidence -> Show user confirmation question
            conf_info = self.generate_video_confirmation_prompt(video_result, lang=lang)
            det_cat = getattr(video_result, "category", "streetlight")
            cat_display = CATEGORY_DISPLAY_NAMES.get(det_cat, {}).get(lang, det_cat.title())
            return {
                "reply": conf_info["reply"],
                "intent": "VIDEO_CONFIRMATION_REQUIRED",
                "language": lang,
                "category": det_cat,
                "category_name": cat_display,
                "ticket_data": None,
                "action_prompt": "confirm_video_complaint",
                "buttons": conf_info["buttons"],
                "conversation_state": state.to_dict()
            }

        # B. Video Candidate Selection Turn
        if action and action.startswith("select_video_candidate_"):
            chosen_cat = normalize_category_key(action.replace("select_video_candidate_", ""))
            state.lock_category(chosen_cat)
            state.video.detected_category = chosen_cat
            if chosen_cat != "streetlight":
                state.visual_status = None
                state.video.visual_status = None
            state.video.has_multiple_issues = False
            state.complaint_status = "awaiting_confirmation"
            conf_info = self.generate_video_confirmation_prompt(state.video, lang=lang)
            cat_display = CATEGORY_DISPLAY_NAMES.get(chosen_cat, {}).get(lang, chosen_cat.title())
            return {
                "reply": conf_info["reply"],
                "intent": "VIDEO_CONFIRMATION_REQUIRED",
                "language": lang,
                "category": chosen_cat,
                "category_name": cat_display,
                "ticket_data": None,
                "action_prompt": "confirm_video_complaint",
                "buttons": conf_info["buttons"],
                "conversation_state": state.to_dict()
            }

        # C. Video Confirmation Turn (Explicit user confirmation required before complaint registration flow)
        if state.complaint_status == "awaiting_confirmation" or action in ["confirm_video_yes", "confirm_video_no"]:
            t_clean = text.lower().strip()
            if self.is_affirmative_confirmation(text, action):
                state.on_video_confirmed_yes()
                # Check if user simultaneously provided location or landmark
                has_loc = any(w in t_clean for w in [
                    "road", "street", "chowk", "nagar", "colony", "sector", "lane", "near", "opposite", "area", "ward",
                    "रस्ता", "चौक", "नगर", "कॉलनी", "सेक्टर", "गल्ली", "जवळ", "समोर", "परिसर", "प्रभाग"
                ])
                if has_loc and len(t_clean) > 5:
                    state.location = text
                    state.record_question_answered("location", text)
                    state.update_missing_fields()

                next_field = state.get_next_missing_field()
                if not next_field:
                    cat_c = state.complaint_category or "streetlight"
                    cat_disp = CATEGORY_DISPLAY_NAMES.get(cat_c, {}).get(lang, cat_c)
                    state.set_pending_field("registration_confirmation")
                    state.complaint_status = "ready_for_submission"
                    turn_seed = len(getattr(state, "history", [])) or getattr(state, "turn_count", 0)
                    reply = human_persona_service.generate_registration_confirmation_prompt(
                        category_name=cat_disp,
                        location=state.location or "आपल्या परिसरातील",
                        lang=lang,
                        turn_seed=turn_seed
                    )
                    buttons = [
                        {"text": "✅ Yes, Register / हो, नोंदवा", "action": "register"},
                        {"text": "❌ No / नाही", "action": "cancel"}
                    ]
                    return {
                        "reply": reply,
                        "intent": "CONVERSATIONAL",
                        "language": lang,
                        "category": cat_c,
                        "category_name": cat_disp,
                        "ticket_data": None,
                        "action_prompt": "confirm_register",
                        "buttons": buttons,
                        "conversation_state": state.to_dict()
                    }

                state.record_question_asked(next_field)
                cat_c = state.complaint_category or "streetlight"
                cat_disp = CATEGORY_DISPLAY_NAMES.get(cat_c, {}).get(lang, cat_c)
                if next_field == "location":
                    if lang == "mr":
                        reply = "छान! तक्रार निश्चित केली आहे. कृपया या समस्येचे अंदाजे ठिकाण, रस्त्याचे नाव किंवा जवळची खूण (लँडमार्क) सांगा."
                    elif lang == "hi":
                        reply = "बहुत अच्छा! शिकायत की पुष्टि हो गई है। कृपया इसका अनुमानित स्थान, सड़क का नाम या नजदीकी लैंडमार्क बताएं।"
                    else:
                        reply = "Great! I have confirmed the complaint. Please provide the approximate location, street name, or nearest landmark for this issue."
                elif next_field == "description":
                    if lang == "mr":
                        reply = "कृपया या समस्येचे थोडे अधिक वर्णन सांगा."
                    elif lang == "hi":
                        reply = "कृपया इस समस्या का थोड़ा और विवरण दें।"
                    else:
                        reply = "Please provide a brief description of the issue."
                else:
                    if lang == "mr":
                        reply = f"कृपया {next_field} संदर्भातील माहिती द्या."
                    elif lang == "hi":
                        reply = f"कृपया {next_field} के बारे में जानकारी दें।"
                    else:
                        reply = f"Please provide details regarding {next_field}."

                return {
                    "reply": reply,
                    "intent": "REQUEST_MISSING_INFO",
                    "language": lang,
                    "category": cat_c,
                    "category_name": cat_disp,
                    "ticket_data": None,
                    "action_prompt": f"provide_{next_field}",
                    "conversation_state": state.to_dict()
                }

            elif self.is_negative_confirmation(text, action):
                state.on_video_confirmed_no()
                if lang == "mr":
                    reply = "ठीक आहे. कृपया आपण कोणती तक्रार नोंदवू इच्छिता ते सांगा."
                elif lang == "hi":
                    reply = "ठीक है। कृपया बताएं कि आप कौन सी शिकायत दर्ज करना चाहते हैं।"
                else:
                    reply = "Okay. Please tell me which issue you would like to report."

                return {
                    "reply": reply,
                    "intent": "GENERAL_CHAT",
                    "language": lang,
                    "category": None,
                    "category_name": None,
                    "ticket_data": None,
                    "action_prompt": "describe_problem",
                    "conversation_state": state.to_dict()
                }
            elif text:
                state.on_video_confirmed_no()

        # 1. Greetings & Personal Introductions (Do NOT create complaints on greetings)
        intro_name = self.extract_name_introduction(text)
        if text and intro_name:
            if lang == "mr":
                reply = (
                    f"नमस्कार {intro_name}! 🙏 PCMC सारथी (वॉर्डमित्र) मध्ये आपले सहर्ष स्वागत आहे.\n\n"
                    "मी पिंपरी चिंचवड महानगरपालिकेचा अधिकृत नागरी साहाय्यक आहे. मी आज आपली कशी मदत करू शकतो? "
                    "आपण खड्डे, पथदिवे, कचरा, पाणीपुरवठा किंवा ड्रेनेज यांसारख्या नागरी समस्यांची तक्रार नोंदवू शकता किंवा महापालिकेच्या सेवांबद्दल विचारू शकता."
                )
            elif lang == "hi":
                reply = (
                    f"नमस्ते {intro_name}! 🙏 PCMC सारथी (वॉर्डमित्र) में आपका स्वागत है।\n\n"
                    "मैं पिंपरी चिंचवड नगर निगम का आधिकारिक नागरिक सहायक हूँ। मैं आज आपकी क्या सहायता कर सकता हूँ? "
                    "आप गड्ढे, स्ट्रीट लाइट, कचरा, जलापूर्ति या जल निकासी जैसी नागरिक समस्याओं की रिपोर्ट कर सकते हैं।"
                )
            else:
                reply = (
                    f"Hello {intro_name}! 🙏 Welcome to PCMC सारथी (WardMitra AI).\n\n"
                    "I am the official civic assistant for Pimpri Chinchwad Municipal Corporation. How can I assist you today? "
                    "You can report civic issues like potholes, streetlights, garbage, water supply, or drainage, or ask about municipal services."
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

        if text and self.is_greeting(text):
            name_token = self.extract_greeting_name(text)
            if lang == "mr":
                salutation = f"नमस्कार {name_token}! 🙏" if name_token else "नमस्कार! 🙏"
                reply = (
                    f"{salutation} मी PCMC सारथी / वॉर्डमित्र AI नागरी साहाय्यक आहे.\n\n"
                    "मी आपल्या परिसरातील नागरी समस्या जसे खड्डे, पथदिवे, कचरा, ड्रेनेज, पाणी अशा समस्यांची नोंद घेणे व सोडवण्यात मदत करतो. "
                    "मी आज आपली काय मदत करू शकतो?"
                )
            elif lang == "hi":
                salutation = f"नमस्ते {name_token}! 🙏" if name_token else "नमस्ते! 🙏"
                reply = (
                    f"{salutation} मैं PCMC सारथी / वॉर्डमित्र AI नागरिक सहायक हूँ।\n\n"
                    "मैं आपके क्षेत्र में गड्ढे, स्ट्रीट लाइट, कचरा, ड्रेनेज, पानी जैसी नागरिक समस्याओं की रिपोर्टिंग और समाधान में मदद करता हूँ। "
                    "मैं आज आपकी क्या सहायता कर सकता हूँ?"
                )
            else:
                salutation = f"Hello {name_token}! 🙏" if name_token else "Hello! 🙏"
                reply = (
                    f"{salutation} Welcome to WardMitra AI (PCMC Sarathi).\n\n"
                    "I can help you report and track civic issues such as potholes, streetlights, garbage, drainage, or water supply in your area. "
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

        # 1.1 Smalltalk / Gratitude
        if text and self.is_smalltalk_or_gratitude(text) and not confirm_register:
            if lang == "mr":
                reply = "आपले सहर्ष स्वागत आहे! भविष्यात कोणतीही नागरी समस्या असल्यास वॉर्डमित्र सदैव आपल्या सेवेत आहे. आपला दिवस आनंदात जावो."
            elif lang == "hi":
                reply = "आपका स्वागत है! भविष्य में किसी भी नागरिक समस्या के लिए वॉर्डमित्र सदैव आपकी सेवा में उपलब्ध है। आपका दिन शुभ हो!"
            else:
                reply = "You're welcome! Feel free to reach out anytime if you face any civic issues like streetlights, potholes, garbage, or water leaks in PCMC. Have a great day!"
            return {
                "reply": reply,
                "intent": "SMALLTALK",
                "language": lang,
                "category": None,
                "category_name": None,
                "ticket_data": None,
                "action_prompt": None
            }

        # 1.2 Bot Identity & Capabilities Inquiry
        if text and self.is_bot_inquiry_or_help(text):
            if lang == "mr":
                reply = (
                    "🤖 मी **वॉर्डमित्र AI (PCMC सारथी)** आहे.\n\n"
                    "मी पिंपरी चिंचवड महानगरपालिकेचा अधिकृत नागरी तक्रार निवारण सहाय्यक आहे. मी खालील समस्या सोडवण्यात मदत करतो:\n"
                    "• 💡 **पथदिवे**: बंद किंवा तुटलेले पथदिवे\n"
                    "• 🕳️ **खड्डे**: रस्त्यावरील खड्डे व नुकसान\n"
                    "• 🗑️ **कचरा**: कचऱ्याचे ढीग व अस्वच्छता\n"
                    "• 🚰 **पाणीपुरवठा**: पाईपलाईन गळती व अनियमित पाणी\n"
                    "• 🚾 **ड्रेनेज**: तुंबलेली गटारे व मॅनहोल\n\n"
                    "आपण थेट समस्येचा संदेश टाईप करू शकता, खालील बटण निवडू शकता किंवा कॅमेरा आयकॉनने फोटो जोडू शकता!"
                )
            else:
                reply = (
                    "🤖 I am **WardMitra AI (PCMC Sarathi)**.\n\n"
                    "I am the official municipal grievance assistant for Pimpri Chinchwad Municipal Corporation. I can assist you with:\n"
                    "• 💡 **Streetlights**: Non-functional or flickering lights\n"
                    "• 🕳️ **Potholes & Roads**: Road damage and potholes\n"
                    "• 🗑️ **Garbage**: Uncollected waste and overflowing bins\n"
                    "• 🚰 **Water Supply**: Pipeline leakages and water issues\n"
                    "• 🚾 **Drainage**: Blocked gutters and open manholes\n\n"
                    "You can simply describe your issue, tap one of the quick category buttons below, or upload a photo to register a complaint!"
                )
            return {
                "reply": reply,
                "intent": "SERVICE_INQUIRY",
                "language": lang,
                "category": None,
                "category_name": None,
                "ticket_data": None,
                "action_prompt": "select_or_describe"
            }

        # 1.3 Complaint Registration Process & Step-by-Step Guidance (City & Gramin friendly)
        if text and self.is_complaint_process_inquiry(text):
            if lang == "mr":
                reply = (
                    "🤖 **वॉर्डमित्र AI - तक्रार नोंदवण्याची सोपी पद्धत (Step-by-Step Guide):**\n\n"
                    "आपण शहरी (City) किंवा ग्रामीण (Gramin) भागातील नागरिक असाल, तरीही तक्रार नोंदवणे अत्यंत सोपे आहे! खालील **४ सोप्या पायऱ्या** फॉलो करा:\n\n"
                    "१️⃣ **पायरी १ - समस्या निवडा किंवा सांगा (Choose Issue):**\n"
                    "   • खालीलपैकी एका बटनावर टॅप करा: 💡 **पथदिवा**, 🕳️ **खड्डा**, 🗑️ **कचरा**, 🚰 **पाणी गळती**, 🌊 **ड्रेनेज**.\n"
                    "   • किंवा आपला प्रश्न साध्या शब्दांत टाईप करा / 🎙️ माईक बटण दाबून थेट बोला.\n\n"
                    "२️⃣ **पायरी २ - नोंदणी फॉर्म उघडा (Open Form):**\n"
                    "   • चॅटमधील **'🚀 तक्रार नोंदवा (Register Complaint)'** बटनावर क्लिक करा.\n"
                    "   • चॅटमधील सर्व माहिती नोंदणी फॉर्मवर आपोआप भरली जाईल.\n\n"
                    "३️⃣ **पायरी ३ - फोटो व लोकेशन (Photo & GPS Location):**\n"
                    "   • समस्येचा एक स्पष्ट फोटो (📷) कॅमेऱ्याने काढून जोडा.\n"
                    "   • मोबाईलचे **📍 GPS लोकेशन** चालू ठेवा, जेणेकरून महापालिकेचे अधिकारी थेट जागेवर पोहोचू शकतील.\n\n"
                    "४️⃣ **पायरी ४ - सबमिट व थेट ट्रॅकिंग (Submit & Live Track):**\n"
                    "   • **'🚀 तक्रार सबमिट करा'** बटनावर क्लिक करा.\n"
                    "   • आपल्याला त्वरित **WM-२०२६...** तिकीट नंबर आणि नियुक्त क्षेत्रीय अभियंत्यांचे नाव व फोन नंबर मिळेल.\n"
                    "   • आपण थेट स्क्रीनवर **४-टप्पे थेट प्रगती (Live Timeline)** पाहू शकता!\n\n"
                    "👉 *सुरुवात करण्यासाठी खालील समस्येच्या बटनावर टॅप करा किंवा मेसेज टाईप करा!*"
                )
            else:
                reply = (
                    "🤖 **WardMitra AI - Step-by-Step Grievance Registration Guide:**\n\n"
                    "Whether you are from urban city wards or rural/outlying areas, filing a complaint is very easy! Follow these **4 simple steps**:\n\n"
                    "1️⃣ **Step 1 - Choose or Describe Your Issue:**\n"
                    "   • Tap any quick category button below: 💡 **Streetlight**, 🕳️ **Pothole**, 🗑️ **Garbage**, 🚰 **Water Leak**, 🌊 **Drainage**.\n"
                    "   • Or simply type your message / use the 🎙️ mic button to speak.\n\n"
                    "2️⃣ **Step 2 - Open Complaint Registration Form:**\n"
                    "   • Click the blue **'🚀 Register Complaint Now'** button in the chat.\n"
                    "   • Your issue description and category will be pre-filled automatically.\n\n"
                    "3️⃣ **Step 3 - Attach Photo & Location (GPS):**\n"
                    "   • Take a clear photo (📷) of the civic problem using your camera.\n"
                    "   • Keep your mobile **📍 GPS location** on so municipal field staff can reach the exact spot.\n\n"
                    "4️⃣ **Step 4 - Submit & Live Track:**\n"
                    "   • Click **'🚀 Submit Complaint'**.\n"
                    "   • You will immediately receive a **Ticket ID (WM-...)**, assigned engineer details, and SLA resolution target.\n"
                    "   • You can track the 4-stage resolution progress directly on the live timeline!\n\n"
                    "👉 *To get started, tap any issue button below or type your problem!*"
                )
            return {
                "reply": reply,
                "intent": "COMPLAINT_PROCESS_GUIDE",
                "language": lang,
                "category": None,
                "category_name": None,
                "ticket_data": None,
                "action_prompt": "select_category_or_describe"
            }

        # 1.4 PCMC Civic Knowledge Base & Municipal Inquiries (Birth/Death Certificates, Tax, Water, Waste, Helplines, Hospitals, Wards)
        kb_answer = human_persona_service.find_knowledge_answer(text, lang)
        if kb_answer:
            return {
                "reply": kb_answer,
                "intent": "CIVIC_KNOWLEDGE_INQUIRY",
                "language": lang,
                "category": None,
                "category_name": None,
                "ticket_data": None,
                "action_prompt": "ask_more_or_register"
            }

        # 1.5 Cancellation Requests (BUG 3)
        if (
            detected_intent == INTENT_CANCEL_COMPLAINT or
            self.is_cancel_intent(text) or
            action == "cancel_complaint"
        ) and not photo_bytes and not video_result:
            tid_in_text = None
            if text:
                m = re.search(r'\bWM-\d{8}-\d{4}\b', text.upper())
                if m:
                    tid_in_text = m.group(0)

            target_complaint = None
            if tid_in_text:
                target_complaint = db.query(Complaint).filter(Complaint.ticket_id == tid_in_text).first()
                if not target_complaint:
                    if lang == "mr":
                        msg = f"⚠️ तक्रार क्र. '{tid_in_text}' प्रणालीत आढळली नाही. कृपया योग्य तिकीट नंबर तपासा."
                    elif lang == "hi":
                        msg = f"⚠️ शिकायत क्र. '{tid_in_text}' नहीं मिली। कृपया सही टिकट नंबर जांचें।"
                    else:
                        msg = f"⚠️ Ticket '{tid_in_text}' was not found. Please double-check your ticket number."
                    return {
                        "reply": msg,
                        "intent": "COMPLAINT_NOT_FOUND",
                        "language": lang,
                        "category": None,
                        "category_name": None,
                        "ticket_data": None,
                        "action_prompt": "specify_ticket_id",
                        "conversation_state": state.to_dict()
                    }
            else:
                # Look up citizen's most recent complaint by citizen_phone
                if citizen_phone and citizen_phone != "Anonymous":
                    target_complaint = (
                        db.query(Complaint)
                        .filter(Complaint.citizen_phone == citizen_phone)
                        .filter(Complaint.status != ComplaintStatus.CANCELLED)
                        .order_by(Complaint.created_at.desc())
                        .first()
                    )

            if target_complaint:
                if target_complaint.status == ComplaintStatus.CANCELLED:
                    if lang == "mr":
                        msg = f"ℹ️ तक्रार क्र. **{target_complaint.ticket_id}** आधीच रद्द झालेली आहे."
                    elif lang == "hi":
                        msg = f"ℹ️ शिकायत क्र. **{target_complaint.ticket_id}** पहले ही रद्द कर दी गई है।"
                    else:
                        msg = f"ℹ️ Complaint **{target_complaint.ticket_id}** is already cancelled."
                    return {
                        "reply": msg,
                        "intent": "COMPLAINT_ALREADY_CANCELLED",
                        "language": lang,
                        "category": target_complaint.detected_category,
                        "category_name": target_complaint.detected_category,
                        "ticket_data": {"ticket_id": target_complaint.ticket_id, "status": "CANCELLED"},
                        "action_prompt": "describe_civic_issue",
                        "conversation_state": state.to_dict()
                    }

                # Found complaint -> ask for confirmation
                state.pending_cancel_ticket = target_complaint.ticket_id
                state.set_pending_field("cancel_confirmation")
                cat_c = target_complaint.detected_category
                cat_info = CATEGORY_DISPLAY_NAMES.get(cat_c, {"en": cat_c.replace('_', ' ').title(), "mr": cat_c, "hi": cat_c})
                cat_disp = cat_info.get(lang, cat_info["en"])

                if lang == "mr":
                    reply = (
                        f"आपण तक्रार क्र. **{target_complaint.ticket_id}** ({cat_disp}) खरोखर रद्द करू इच्छिता का?\n\n"
                        "कृपया पुष्टी करण्यासाठी **'होय'** किंवा **'नाही'** सांगा."
                    )
                elif lang == "hi":
                    reply = (
                        f"क्या आप शिकायत क्र. **{target_complaint.ticket_id}** ({cat_disp}) को वास्तव में रद्द करना चाहते हैं?\n\n"
                        "कृपया पुष्टि के लिए **'हाँ'** या **'नहीं'** कहें।"
                    )
                else:
                    reply = (
                        f"Are you sure you want to cancel complaint **{target_complaint.ticket_id}** ({cat_disp})?\n\n"
                        "Please confirm by replying **'Yes'** or **'No'**."
                    )
                buttons = [
                    {"text": "🛑 Yes, Cancel / हो, रद्द करा", "action": "confirm_cancel_yes"},
                    {"text": "Keep Active / सुरू ठेवा", "action": "confirm_cancel_no"}
                ]
                return {
                    "reply": reply,
                    "intent": "CANCEL_CONFIRMATION_REQUIRED",
                    "language": lang,
                    "category": target_complaint.detected_category,
                    "category_name": cat_disp,
                    "ticket_data": {"ticket_id": target_complaint.ticket_id, "status": target_complaint.status.value},
                    "action_prompt": "confirm_cancel",
                    "buttons": buttons,
                    "conversation_state": state.to_dict()
                }
            else:
                if lang == "mr":
                    msg = "आपल्या फोन नंबरवर कोणतीही सक्रिय तक्रार आढळली नाही. आपण रद्द करू इच्छित असलेला तिकीट क्रमांक (उदा. WM-20260921-0001) सांगू शकाल का?"
                elif lang == "hi":
                    msg = "आपके फोन नंबर पर कोई सक्रिय शिकायत नहीं मिली। कृपया रद्द करने के लिए टिकट आईडी बताएं।"
                else:
                    msg = "No active complaint was found for your phone number. Please provide the specific Ticket ID (e.g. WM-20260921-0001) you wish to cancel."
                return {
                    "reply": msg,
                    "intent": "COMPLAINT_NOT_FOUND",
                    "language": lang,
                    "category": None,
                    "category_name": None,
                    "ticket_data": None,
                    "action_prompt": "specify_ticket_id",
                    "conversation_state": state.to_dict()
                }

        # 2. Tracking Requests
        ticket_id_found = self.is_tracking_intent(text)
        if ticket_id_found and ticket_id_found != "UNKNOWN":
            complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id_found).first()
            if complaint:
                worker_str = complaint.assigned_worker_name or "Ward Field Engineer"
                worker_contact = f" ({complaint.assigned_worker_contact})" if complaint.assigned_worker_contact else ""
                ward_display = f"प्रभाग {complaint.ward_number}" if complaint.ward_number else "निश्चित केले नाही"
                ward_info = geo_agent.get_ward_by_number(complaint.ward_number) if complaint.ward_number else None
                if ward_info:
                    ward_display = f"प्रभाग {ward_info['ward_number']}: {ward_info['ward_name']} ({ward_info['zone']})"

                if lang == "mr":
                    reply = (
                        f"🔍 **तक्रार स्थिती: {complaint.ticket_id}**\n"
                        f"• स्थिती: {complaint.status.value}\n"
                        f"• वर्ग: {complaint.detected_category}\n"
                        f"• प्रभाग: {ward_display}\n"
                        f"• नियुक्त कामगार: {worker_str}{worker_contact}\n"
                        f"• मुदत (SLA): {complaint.sla_hours} तास"
                    )
                else:
                    reply = (
                        f"🔍 **Complaint Status: {complaint.ticket_id}**\n"
                        f"• Status: {complaint.status.value}\n"
                        f"• Category: {complaint.detected_category}\n"
                        f"• Ward: {ward_display}\n"
                        f"• Assigned Worker: {worker_str}{worker_contact}\n"
                        f"• SLA Duration: {complaint.sla_hours} Hours"
                    )

                buttons = None
                action_prompt = None
                intent = "TRACK_COMPLAINT"
                if complaint.status.value in ["RESOLVED", "CITIZEN_CONFIRMATION"]:
                    intent = "CONFIRM_RESOLUTION_REQUIRED"
                    action_prompt = "confirm_resolution"
                    if lang == "mr":
                        reply += (
                            "\n\n✅ **क्षेत्रीय अधिकाऱ्यांकडून ही समस्या सोडवण्यात आल्याचे नोंदवले आहे.**\n"
                            "कृपया कामाची प्रत्यक्ष खात्री करा: समस्या खरोखर सुटली आहे का?"
                        )
                        buttons = [
                            {"text": "✅ होय, समस्या सुटली आहे (Yes, Resolved)", "action": f"confirm_resolution_yes_{complaint.ticket_id}"},
                            {"text": "❌ नाही, समस्या अद्याप आहे (No, Still Exists)", "action": f"confirm_resolution_no_{complaint.ticket_id}"}
                        ]
                    elif lang == "hi":
                        reply += (
                            "\n\n✅ **क्षेत्रीय अधिकारी द्वारा यह समस्या हल कर दी गई है।**\n"
                            "कृपया पुष्टि करें: क्या समस्या हल हो चुकी है?"
                        )
                        buttons = [
                            {"text": "✅ हाँ, समस्या हल हो गई है", "action": f"confirm_resolution_yes_{complaint.ticket_id}"},
                            {"text": "❌ नहीं, समस्या अभी भी है", "action": f"confirm_resolution_no_{complaint.ticket_id}"}
                        ]
                    else:
                        reply += (
                            "\n\n✅ **Field staff have marked this grievance as RESOLVED.**\n"
                            "Please verify and confirm: Has the problem been resolved?"
                        )
                        buttons = [
                            {"text": "✅ Yes, Problem Resolved", "action": f"confirm_resolution_yes_{complaint.ticket_id}"},
                            {"text": "❌ No, Problem Still Exists", "action": f"confirm_resolution_no_{complaint.ticket_id}"}
                        ]

                result_dict = {
                    "reply": reply,
                    "intent": intent,
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
                    "action_prompt": action_prompt
                }
                if buttons:
                    result_dict["buttons"] = buttons
                return result_dict
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

        # 3. Clarification for Ambiguous Light Complaints (Requirement 1 & Test 1)
        # e.g., "The light is off." / "लाईट बंद आहे."
        if text and is_ambiguous_light_complaint(text):
            clarification_q = get_light_clarification_prompt(lang)
            return {
                "reply": clarification_q,
                "intent": "CLARIFICATION_REQUIRED",
                "language": lang,
                "category": None,
                "category_name": None,
                "ticket_data": None,
                "action_prompt": "clarify_home_or_streetlight"
            }

        # 4. Clarification continuation: User clarifies "Streetlight" (Requirement 1 continuation)
        t_clean = text.lower().strip()
        if t_clean in ["streetlight", "street light", "पथदिवा", "रस्त्यावरील दिवा", "रस्त्यावरचा दिवा", "streetlights"]:
            state.lock_category("streetlight")
            state.record_question_answered("home_vs_streetlight", "streetlight")
            evidence_prompt = get_streetlight_evidence_prompt(lang)
            return {
                "reply": evidence_prompt,
                "intent": "EVIDENCE_REQUIRED",
                "language": lang,
                "category": "streetlight",
                "category_name": CATEGORY_DISPLAY_NAMES["streetlight"][lang],
                "ticket_data": None,
                "action_prompt": "upload_photo",
                "conversation_state": state.to_dict()
            }

        # Check if user is answering the current question from previous turn
        if state.current_question == "visual_status" or (state.complaint_category == "streetlight" and not state.visual_status and not state.is_question_answered("visual_status")):
            if t_clean in ["off", "band", "band hai", "band aahe", "nahi", "not on", "dead", "blackout", "बंद", "बंद आहे", "लाईट बंद आहे"]:
                state.visual_status = "off"
                state.issue_type = "light_not_working"
                state.record_question_answered("visual_status", text)
                state.set_pending_field("location")
                state.record_question_asked("location")
                cat_display = CATEGORY_DISPLAY_NAMES.get("streetlight", {}).get(lang, "Streetlight")
                if lang == "mr":
                    reply = "मी नोंदवले आहे की हा पथदिवा बंद आहे. कृपया या पथदिव्याचे अंदाजे ठिकाण, रस्त्याचे नाव किंवा जवळची खूण (लँडमार्क) सांगा."
                elif lang == "hi":
                    reply = "स्ट्रीट लाइट बंद होने की जानकारी दर्ज कर ली गई है। कृपया इसका अनुमानित स्थान, सड़क का नाम या नजदीकी लैंडमार्क बताएं।"
                else:
                    reply = "I have noted that the streetlight is OFF. What is the approximate location, street name, or nearest landmark for this streetlight?"
                return {
                    "reply": reply,
                    "intent": "INFORMATION_REQUIRED",
                    "language": lang,
                    "category": "streetlight",
                    "category_name": cat_display,
                    "ticket_data": None,
                    "action_prompt": "provide_location",
                    "conversation_state": state.to_dict()
                }
            elif t_clean in ["on", "chalu", "chalu hai", "chal raha hai", "suru", "चालू", "सुरु आहे"]:
                state.visual_status = "on"
                state.issue_type = "burning_during_day"
                state.record_question_answered("visual_status", text)
                state.set_pending_field("location")
                state.record_question_asked("location")
                cat_display = CATEGORY_DISPLAY_NAMES.get("streetlight", {}).get(lang, "Streetlight")
                if lang == "mr":
                    reply = "हा पथदिवा चालू असल्याचे नोंदवले आहे. कृपया या पथदिव्याचे अंदाजे ठिकाण किंवा जवळची खूण सांगा."
                elif lang == "hi":
                    reply = "स्ट्रीट लाइट चालू होने की जानकारी दर्ज कर ली गई है। कृपया इसका अनुमानित स्थान या नजदीकी लैंडमार्क बताएं।"
                else:
                    reply = "I have noted that the streetlight is ON. What is the approximate location or nearest landmark for this streetlight?"
                return {
                    "reply": reply,
                    "intent": "INFORMATION_REQUIRED",
                    "language": lang,
                    "category": "streetlight",
                    "category_name": cat_display,
                    "ticket_data": None,
                    "action_prompt": "provide_location",
                    "conversation_state": state.to_dict()
                }

        # Check if citizen is introducing a different civic category (prevent treating new complaint as location or description)
        detected_category_hint = intent_info.get("category") if intent_info else None
        if not detected_category_hint and text:
            detected_category_hint = nlp_agent._keyword_match(t_clean)
        
        current_active_cat = state.confirmed_category or state.complaint_category
        is_category_switch = bool(
            detected_category_hint and
            current_active_cat and
            detected_category_hint != current_active_cat
        ) or (detected_intent in [INTENT_NEW_COMPLAINT, INTENT_CATEGORY_SELECTION, INTENT_START_NEW_COMPLAINT] and bool(detected_category_hint and detected_category_hint != current_active_cat))

        # Description Capture
        if not is_category_switch and (state.pending_field == "description" or state.current_question == "description" or detected_intent == INTENT_DESCRIPTION_PROVIDED) and not photo_bytes and text:
            state.description = text.strip()
            state.record_question_answered("description", text.strip())
            state.clear_pending_field()
            state.set_pending_field("location")
            state.record_question_asked("location")
            cat_c = state.confirmed_category or state.complaint_category or "other"
            cat_disp = CATEGORY_DISPLAY_NAMES.get(cat_c, {}).get(lang, cat_c.replace('_', ' ').title())
            if lang == "mr":
                reply = f"मी समस्येची नोंद घेतली आहे. कृपया **{cat_disp}** समस्येचे अंदाजे ठिकाण, पत्ता किंवा जवळची खूण सांगा."
            elif lang == "hi":
                reply = f"विवरण दर्ज कर लिया गया है। कृपया **{cat_disp}** समस्या का अनुमानित स्थान या नजदीकी लैंडमार्क बताएं।"
            else:
                reply = f"I have noted your description. What is the approximate location, street name, or nearest landmark for this **{cat_disp}** issue?"
            return {
                "reply": reply,
                "intent": "INFORMATION_REQUIRED",
                "language": lang,
                "category": cat_c,
                "category_name": cat_disp,
                "ticket_data": None,
                "action_prompt": "provide_location",
                "conversation_state": state.to_dict()
            }

        # Sub-category Handling & Protection (Garbage & Civic Subtypes)
        WASTE_SUBCAT_MAP = {
            "household waste": "household_waste",
            "household": "household_waste",
            "घरगुती कचरा": "household_waste",
            "घरगुती": "household_waste",
            "commercial waste": "commercial_waste",
            "commercial": "commercial_waste",
            "व्यावसायिक कचरा": "commercial_waste",
            "व्यावसायिक": "commercial_waste",
            "construction debris": "construction_debris",
            "debris": "construction_debris",
            "बांधकाम मलबा": "construction_debris",
            "मलबा": "construction_debris",
            "राडारोडा": "construction_debris",
            "ओला कचरा": "wet_waste",
            "सुका कचरा": "dry_waste",
            "wet waste": "wet_waste",
            "dry waste": "dry_waste"
        }
        matched_waste_subcat = None
        if t_clean:
            for w_key, w_val in WASTE_SUBCAT_MAP.items():
                if w_key in t_clean:
                    matched_waste_subcat = w_val
                    break

        if matched_waste_subcat and not photo_bytes and not video_result:
            state.issue_type = matched_waste_subcat
            state.sub_category = matched_waste_subcat
            state.record_question_answered("issue_type", text)
            state.record_question_answered("sub_category", text)
            
            cat_c = state.confirmed_category or state.complaint_category or "garbage"
            cat_disp = CATEGORY_DISPLAY_NAMES.get(cat_c, {}).get(lang, cat_c.replace('_', ' ').title())

            if state.location:
                state.set_pending_field("registration_confirmation")
                state.complaint_status = "ready_for_submission"
                turn_seed = len(getattr(state, "history", [])) or getattr(state, "turn_count", 0)
                reply = human_persona_service.generate_registration_confirmation_prompt(
                    category_name=cat_disp,
                    location=state.location,
                    lang=lang,
                    turn_seed=turn_seed
                )
                buttons = [
                    {"text": "✅ Yes, Register / हो, नोंदवा", "action": "register"},
                    {"text": "❌ No / नाही", "action": "cancel"}
                ]
                return {
                    "reply": reply,
                    "intent": "CONVERSATIONAL",
                    "language": lang,
                    "category": cat_c,
                    "category_name": cat_disp,
                    "ticket_data": None,
                    "action_prompt": "confirm_register",
                    "buttons": buttons,
                    "conversation_state": state.to_dict()
                }
            else:
                state.set_pending_field("location")
                state.complaint_status = "collecting_information"
                if lang == "mr":
                    reply = f"समजले, कचऱ्याचा प्रकार **{matched_waste_subcat}** नोंदवला आहे. कृपया या कचऱ्याचे ठिकाण, रस्त्याचे नाव किंवा जवळची खूण सांगा."
                elif lang == "hi":
                    reply = f"समझ गया, कचरे का प्रकार **{matched_waste_subcat}** दर्ज कर लिया है। कृपया इसका स्थान या नजदीकी लैंडमार्क बताएं।"
                else:
                    reply = f"Recorded waste type as **{matched_waste_subcat}**. Please provide the location, street name, or nearest landmark."
                return {
                    "reply": reply,
                    "intent": "INFORMATION_REQUIRED",
                    "language": lang,
                    "category": cat_c,
                    "category_name": cat_disp,
                    "ticket_data": None,
                    "action_prompt": "provide_location",
                    "conversation_state": state.to_dict()
                }

        # Location Capture (BUG 1, BUG 3)
        # Check if the user is answering a location question or providing location while category is locked
        is_location_candidate = (
            not is_category_switch and
            not matched_waste_subcat and
            not nlp_agent.is_affirmative_yes(text) and
            not nlp_agent.is_negative_no(text) and
            (
                state.pending_field == "location" or
                state.current_question == "location" or
                detected_intent == INTENT_LOCATION_PROVIDED or
                (
                    state.category_locked and
                    not state.location and
                    state.pending_field != "description" and
                    state.current_question != "description" and
                    len(t_clean) >= 3 and
                    not nlp_agent.is_abusive_or_frustrated(text) and
                    not self.is_greeting(text) and
                    not photo_bytes and
                    not video_result and
                    not self.is_registration_intent(text, action, confirm_register)
                ) or
                (
                    state.pending_field == "registration_confirmation" and
                    not self.is_registration_intent(text, action, confirm_register) and
                    (
                        bool(geo_agent.detect_ward_from_text(text)) or
                        any(loc in t_clean for loc in ["road", "street", "chowk", "nagar", "colony", "sector", "lane", "station", "ward", "रस्ता", "चौक", "नगर", "कॉलनी", "सेक्टर", "प्रभाग"])
                    )
                )
            )
        )

        if is_location_candidate and not photo_bytes and not self.is_registration_intent(text, action, confirm_register):
            if len(t_clean) >= 3 and not nlp_agent.is_affirmative_yes(text) and not nlp_agent.is_negative_no(text):
                state.location = text.strip()
                state.record_question_answered("location", text.strip())
                state.clear_pending_field()
                state.update_missing_fields()
                
                cat_c = state.confirmed_category or state.complaint_category or "pothole"
                cat_disp = CATEGORY_DISPLAY_NAMES.get(cat_c, {}).get(lang, cat_c.replace('_', ' ').title())

                # If streetlight and visual status is still unknown, ask visual status first
                if cat_c == "streetlight" and not state.visual_status and not state.is_question_answered("visual_status"):
                    state.set_pending_field("visual_status")
                    state.record_question_asked("visual_status")
                    if lang == "mr":
                        q_reply = f"मी **{state.location}** येथील पथदिव्याचे ठिकाण नोंदवले आहे. हा पथदिवा चालू आहे की बंद आहे?"
                    elif lang == "hi":
                        q_reply = f"मैंने **{state.location}** पर स्ट्रीट लाइट का स्थान दर्ज कर लिया है। क्या लाइट चालू है या बंद है?"
                    else:
                        q_reply = f"I have recorded your location as **{state.location}**. Is the streetlight currently ON or OFF?"
                    return {
                        "reply": q_reply,
                        "intent": "INFORMATION_REQUIRED",
                        "language": lang,
                        "category": "streetlight",
                        "category_name": cat_disp,
                        "ticket_data": None,
                        "action_prompt": "clarify_visual_status",
                        "conversation_state": state.to_dict()
                    }

                # Otherwise, ready for registration confirmation
                state.complaint_status = "ready_for_submission"
                state.set_pending_field("registration_confirmation")
                turn_seed = len(getattr(state, "history", [])) or getattr(state, "turn_count", 0)
                reply = human_persona_service.generate_registration_confirmation_prompt(
                    category_name=cat_disp,
                    location=state.location,
                    lang=lang,
                    turn_seed=turn_seed
                )
                
                buttons = [
                    {"text": "✅ Yes, Register / हो, नोंदवा", "action": "register"},
                    {"text": "❌ No / नाही", "action": "cancel"}
                ]
                return {
                    "reply": reply,
                    "intent": "CONVERSATIONAL",
                    "language": lang,
                    "category": cat_c,
                    "category_name": cat_disp,
                    "ticket_data": None,
                    "action_prompt": "confirm_register",
                    "buttons": buttons,
                    "conversation_state": state.to_dict()
                }

        # 5. Category Button / Bare Category Selection (Requirement 2 & Test 2)
        bare_category_target = None
        if action == "select_category" and category:
            bare_category_target = category
        elif t_clean in ["drainage", "ड्रेनेज", "pothole", "खड्डा", "garbage", "कचरा", "water", "पाणी"]:
            bare_category_target = t_clean

        if bare_category_target and not photo_bytes:
            clarification_prompt = get_category_clarification_prompt(bare_category_target, lang)
            cat_display = CATEGORY_DISPLAY_NAMES.get(bare_category_target, {}).get(lang, bare_category_target)
            state.lock_category(bare_category_target)
            state.set_pending_field("description")
            state.record_question_asked("description")
            return {
                "reply": clarification_prompt,
                "intent": "CATEGORY_CLARIFICATION_REQUIRED",
                "language": lang,
                "category": bare_category_target,
                "category_name": cat_display,
                "ticket_data": None,
                "action_prompt": "describe_problem",
                "conversation_state": state.to_dict()
            }

        # Detect Civic Category from NLP (user text) or fallback to session category (BUG 1, BUG 2, BUG 3)
        text_category = None
        extracted_cat = None
        if text and detected_intent not in [INTENT_LOCATION_PROVIDED, INTENT_REGISTRATION_CONFIRMATION, INTENT_REGISTRATION_REJECTION, INTENT_ABUSIVE_MESSAGE, INTENT_GREETING, INTENT_UNRELATED]:
            # Check informal normalizer first (fast Gen-Z and multilingual detection)
            informal_res = nlp_agent.extract_intent_and_category(text)
            if informal_res.get("category"):
                extracted_cat = informal_res["category"]
                state.issue_type = informal_res.get("issue_type")
            else:
                extracted_cat = nlp_agent._keyword_match(t_clean)
                if not extracted_cat:
                    for cat_key in CATEGORY_DISPLAY_NAMES.keys():
                        if cat_key in t_clean:
                            extracted_cat = cat_key
                            break

        current_cat = state.confirmed_category or state.complaint_category
        is_completed_or_idle = getattr(state, "complaint_status", "") in ["completed", "idle"]
        has_new_different_category = bool(extracted_cat and current_cat and extracted_cat != current_cat)

        if extracted_cat and (is_completed_or_idle or has_new_different_category or not current_cat):
            # Citizen is starting a new or different complaint
            detected_category = extracted_cat
            text_category = extracted_cat
            state.reset_after_registration()
            state.lock_category(detected_category, source="text")
        elif state.category_confirmed and state.confirmed_category:
            detected_category = state.confirmed_category
            text_category = state.confirmed_category
        elif state.category_locked and state.complaint_category:
            detected_category = state.complaint_category
            text_category = state.complaint_category
        else:
            detected_category = category or extracted_cat or state.complaint_category
            if detected_category:
                state.lock_category(detected_category, source="text")
                if detected_category == "streetlight" and not state.visual_status:
                    light_status = self._detect_light_visual_status(t_clean)
                    if light_status == "off":
                        state.visual_status = "off"
                        state.record_question_answered("visual_status", "off")
                    elif light_status == "on":
                        state.visual_status = "on"
                        state.record_question_answered("visual_status", "on")
                    else:
                        state.set_pending_field("visual_status")
                        state.record_question_asked("visual_status")
                if not state.location and state.pending_field != "visual_status":
                    state.set_pending_field("location")
                    state.record_question_asked("location")

        cat_info = CATEGORY_DISPLAY_NAMES.get(detected_category, {"en": "Civic Grievance", "mr": "नागरी समस्या"})
        cat_name_current = cat_info.get(lang, cat_info["en"])

        # 6. Image Understanding, Evidence Verification & Context-Aware Follow-up
        saved_tmp_photo_path = None
        img_analysis = None
        all_info_known = False

        if photo_filename and photo_bytes:
            unique_tmp_filename = f"chat_verify_{routing_agent.generate_ticket_id()}_{photo_filename}"
            tmp_path = Path(settings.UPLOAD_DIR) / unique_tmp_filename
            with open(tmp_path, "wb") as f:
                f.write(photo_bytes)
            saved_tmp_photo_path = str(tmp_path)

            # Persist location if citizen provided location text alongside photo
            if text and len(text.strip()) >= 3 and not nlp_agent.is_affirmative_yes(text) and not nlp_agent.is_negative_no(text):
                cat_names = set(CATEGORY_DISPLAY_NAMES.keys()) | {"light", "pothole", "garbage", "drainage", "water", "कचरा", "खड्डा", "दिवा"}
                if text.strip().lower() not in cat_names:
                    state.location = text.strip()
                    state.record_question_answered("location", text.strip())
                    state.update_missing_fields()

            target_cat_for_eval = category or (text_category if text else None) or state.complaint_category
            if not target_cat_for_eval and "light" in t_clean:
                target_cat_for_eval = "streetlight"

            # Deep AI Vision Understanding Pipeline
            img_analysis = image_understanding_service.analyze_image(
                saved_tmp_photo_path,
                user_message=text,
                context={
                    "category": target_cat_for_eval,
                    "ward_number": ward_number,
                    "latitude": latitude,
                    "longitude": longitude
                }
            )

            # Section 3, 4 & 5: Check Image vs User Complaint Consistency
            consistency = image_understanding_service.check_image_complaint_consistency(
                target_cat_for_eval,
                img_analysis,
                lang=lang
            )

            if consistency.get("is_mismatch"):
                mtype = consistency.get("mismatch_type")
                reply_msg = consistency.get("clarification_question")

                # Category Mismatch (e.g. Streetlight text + Garbage image): Protect current complaint! (BUG 7)
                if mtype == "category_mismatch":
                    state.complaint_status = "awaiting_selection"
                    state.set_pending_field("category")
                    state.candidate_categories = [target_cat_for_eval, img_analysis.category]
                    state.text_category = target_cat_for_eval
                    state.image_category = img_analysis.category
                    
                    text_cat_disp = CATEGORY_DISPLAY_NAMES.get(target_cat_for_eval, {}).get(lang, (target_cat_for_eval or "").replace('_', ' ').title())
                    img_cat_disp = CATEGORY_DISPLAY_NAMES.get(img_analysis.category, {}).get(lang, (img_analysis.category or "").replace('_', ' ').title())
                    
                    buttons = [
                        {"text": f"1. {text_cat_disp}", "action": f"select_category_{target_cat_for_eval}"},
                        {"text": f"2. {img_cat_disp}", "action": f"select_category_{img_analysis.category}"}
                    ]
                    return {
                        "reply": reply_msg,
                        "intent": "IMAGE_COMPLAINT_MISMATCH",
                        "language": lang,
                        "category": target_cat_for_eval,  # Preserved!
                        "category_name": cat_name_current,
                        "ticket_data": None,
                        "action_prompt": "clarify_complaint_choice",
                        "buttons": buttons,
                        "image_analysis": img_analysis.model_dump(),
                        "conversation_state": state.to_dict()
                    }
                elif mtype == "multiple_issues":
                    return {
                        "reply": reply_msg,
                        "intent": "MULTIPLE_ISSUES_DETECTED",
                        "language": lang,
                        "category": target_cat_for_eval or img_analysis.category,
                        "category_name": cat_name_current,
                        "ticket_data": None,
                        "action_prompt": "select_issue",
                        "image_analysis": img_analysis.model_dump(),
                        "conversation_state": state.to_dict()
                    }
                elif mtype == "blurry_image":
                    return {
                        "reply": reply_msg,
                        "intent": "LOW_CONFIDENCE_IMAGE",
                        "language": lang,
                        "category": target_cat_for_eval or "pothole",
                        "category_name": cat_name_current,
                        "ticket_data": None,
                        "action_prompt": "upload_clear_photo",
                        "image_analysis": img_analysis.model_dump(),
                        "conversation_state": state.to_dict()
                    }
                elif mtype == "unrelated_image":
                    return {
                        "reply": reply_msg,
                        "intent": "EVIDENCE_REJECTED",
                        "language": lang,
                        "category": target_cat_for_eval,
                        "category_name": cat_name_current,
                        "ticket_data": None,
                        "action_prompt": "upload_valid_photo",
                        "image_analysis": img_analysis.model_dump(),
                        "conversation_state": state.to_dict()
                    }

            # If image is consistent: update state with visual observations
            state.image_uploaded = True
            state.image_analysis = img_analysis.model_dump()
            state.image.uploaded = True
            state.image.relevant = True
            state.image.detected_category = img_analysis.category
            state.image.confidence = img_analysis.confidence
            state.image.visual_status = img_analysis.visual_status
            state.image.observations = img_analysis.observations

            if img_analysis.visual_status in ["on", "off"]:
                state.visual_status = img_analysis.visual_status
                state.record_question_answered("visual_status", f"{img_analysis.visual_status} (from photo)")

            # Check for non-civic rejection via image_agent
            if target_cat_for_eval:
                ev_result = image_agent.verify_evidence(
                    target_cat_for_eval,
                    {
                        "category": img_analysis.category,
                        "confidence": img_analysis.confidence,
                        "raw_label": img_analysis.observations[0] if img_analysis.observations else ""
                    },
                    min_confidence=0.30,
                    image_path=saved_tmp_photo_path
                )
                if not ev_result["is_relevant"]:
                    reply_msg = ev_result["message_mr"] if lang == "mr" else ev_result["message_en"]
                    return {
                        "reply": reply_msg,
                        "intent": "EVIDENCE_REJECTED",
                        "language": lang,
                        "category": target_cat_for_eval,
                        "category_name": cat_name_current,
                        "ticket_data": None,
                        "action_prompt": "upload_valid_photo",
                        "image_analysis": img_analysis.model_dump(),
                        "conversation_state": state.to_dict()
                    }

            # Check 2: Low-confidence / Blurry image -> Request clearer photo
            if img_analysis.image_quality in ["blurry", "corrupt"] or img_analysis.issue in ["blurry_image", "corrupted_image"]:
                reply_msg = img_analysis.recommended_followup_mr if lang == "mr" else img_analysis.recommended_followup
                return {
                    "reply": reply_msg,
                    "intent": "LOW_CONFIDENCE_IMAGE",
                    "language": lang,
                    "category": target_cat_for_eval or "pothole",
                    "category_name": cat_name_current,
                    "ticket_data": None,
                    "action_prompt": "upload_clear_photo",
                    "image_analysis": img_analysis.model_dump(),
                    "conversation_state": state.to_dict()
                }

            # Check 3: Multiple distinct civic issues detected in one image
            if img_analysis.has_multiple_issues:
                reply_msg = img_analysis.recommended_followup_mr if lang == "mr" else img_analysis.recommended_followup
                return {
                    "reply": reply_msg,
                    "intent": "MULTIPLE_ISSUES_DETECTED",
                    "language": lang,
                    "category": img_analysis.category,
                    "category_name": cat_name_current,
                    "ticket_data": None,
                    "action_prompt": "select_issue",
                    "image_analysis": img_analysis.model_dump(),
                    "conversation_state": state.to_dict()
                }

            # Check if there is a prior text-derived category
            has_prior_text_category = bool(
                state.text_category or 
                (state.category_confirmed and state.category_source in ["text", "button", "correction"]) or
                (text and text_category)
            )

            # Adopt verified image category if not previously specified by citizen
            if not detected_category or detected_category == "other":
                detected_category = img_analysis.category
                if has_prior_text_category:
                    state.lock_category(detected_category)
                else:
                    state.complaint_category = detected_category
                    state.category_locked = False
                    state.category_confirmed = False
                cat_info = CATEGORY_DISPLAY_NAMES.get(detected_category, {"en": "Civic Grievance", "mr": "नागरी समस्या"})
                cat_name_current = cat_info.get(lang, cat_info["en"])

            # Context-Aware Follow-Up Engine
            followup = image_understanding_service.generate_followup(
                analysis=img_analysis,
                user_message=text,
                context={
                    "ward_number": ward_number,
                    "latitude": latitude,
                    "longitude": longitude,
                    "category": detected_category
                },
                lang=lang
            )

            all_info_known = followup.get("all_info_known", False)
            user_explicitly_confirms = self.is_registration_intent(text, action, confirm_register)

            # BUG 2: When photo is uploaded with no prior text category (e.g. photo + 'register this complaint', or photo with text when proceeding to location/registration),
            # Prompt classification confirmation first with Yes/No before proceeding to location/registration.
            if not has_prior_text_category and (text or user_explicitly_confirms or all_info_known):
                state.complaint_category = detected_category
                state.category_locked = False
                state.category_confirmed = False
                state.set_pending_field("image_category_confirmation")
                cat_info = CATEGORY_DISPLAY_NAMES.get(detected_category, {"en": detected_category.replace('_', ' ').title(), "mr": detected_category, "hi": detected_category})
                cat_name_current = cat_info.get(lang, cat_info["en"])
                turn_seed = len(getattr(state, "history", [])) or getattr(state, "turn_count", 0)
                reply = human_persona_service.generate_image_category_confirmation(
                    category_name=cat_name_current,
                    lang=lang,
                    turn_seed=turn_seed
                )
                buttons = [
                    {"text": "✅ Yes / हो", "action": "confirm_image_category_yes"},
                    {"text": "❌ No / नाही", "action": "confirm_image_category_no"}
                ]
                return {
                    "reply": reply,
                    "intent": "IMAGE_CATEGORY_CONFIRMATION_REQUIRED",
                    "language": lang,
                    "category": detected_category,
                    "category_name": cat_name_current,
                    "ticket_data": None,
                    "action_prompt": "confirm_image_category",
                    "buttons": buttons,
                    "image_analysis": img_analysis.model_dump(),
                    "conversation_state": state.to_dict()
                }

            # If user has not confirmed and missing required info -> Ask follow-up question
            if not user_explicitly_confirms and not all_info_known:
                return {
                    "reply": followup["followup_question"],
                    "intent": "IMAGE_FOLLOWUP_REQUIRED",
                    "language": lang,
                    "category": detected_category,
                    "category_name": cat_name_current,
                    "ticket_data": None,
                    "action_prompt": followup.get("next_action", "answer_followup"),
                    "image_analysis": img_analysis.model_dump(),
                    "conversation_state": state.to_dict()
                }

        # Streetlight without photo: If status is not known yet, ask ON/OFF
        if not photo_bytes and detected_category == "streetlight":
            if not state.visual_status and not state.is_question_answered("visual_status"):
                light_status = self._detect_light_visual_status(t_clean)
                if light_status == "off":
                    state.visual_status = "off"
                    state.record_question_answered("visual_status", "off")
                elif light_status == "on":
                    state.visual_status = "on"
                    state.record_question_answered("visual_status", "on")
                else:
                    state.set_pending_field("visual_status")
                    state.record_question_asked("visual_status")
                    if lang == "mr":
                        q_reply = "मी आपल्या पथदिव्याच्या समस्येची नोंद घेतली आहे. हा पथदिवा चालू आहे की बंद आहे?"
                    elif lang == "hi":
                        q_reply = "स्ट्रीट लाइट की शिकायत दर्ज करने के लिए, क्या लाइट चालू है या बंद है?"
                    else:
                        q_reply = "I have noted your streetlight complaint. Is the light currently ON or OFF?"
                    return {
                        "reply": q_reply,
                        "intent": "INFORMATION_REQUIRED",
                        "language": lang,
                        "category": "streetlight",
                        "category_name": cat_name_current,
                        "ticket_data": None,
                        "action_prompt": "clarify_visual_status",
                        "conversation_state": state.to_dict()
                    }

        # 7. Registration Flow (Only for verified evidence & confirmed complaints)
        user_wants_registration = self.is_registration_intent(text, action, confirm_register) or all_info_known
        if user_wants_registration and (text or photo_bytes):
            final_cat = detected_category or state.confirmed_category or state.complaint_category or "pothole"
            if nlp_agent.is_affirmative_yes(text) or len(text.strip()) < 5 or text.lower().strip() in ["हो", "होय", "yes", "yep", "sure", "करा", "नोंदवा", "submit", "register", "urgent", "proceed"]:
                if state.description and len(state.description.strip()) > 5:
                    final_desc = state.description.strip()
                elif state.location and len(state.location.strip()) > 5:
                    issue_tag = f" ({img_analysis.issue})" if img_analysis else ""
                    final_desc = f"{cat_name_current} issue at {state.location.strip()}{issue_tag}." if lang != "mr" else f"{state.location.strip()} येथे {cat_name_current} संदर्भात नागरी तक्रार{issue_tag}."
                else:
                    issue_tag = f" ({img_analysis.issue})" if img_analysis else ""
                    final_desc = f"पिंपरी चिंचवड परिसरातील {cat_name_current} संदर्भात नागरी तक्रार{issue_tag}." if lang == "mr" else f"Civic grievance regarding {cat_info['en']} in PCMC area{issue_tag}."
            else:
                final_desc = text

            allow_dup_override = action in ["register_new", "force_register"]
            if state.location and state.location not in final_desc:
                final_desc = f"{final_desc} [Location: {state.location}]"

            final_photo_filename = photo_filename
            final_photo_bytes = photo_bytes
            if not final_photo_bytes and getattr(state.video, "evidence_frame_path", None) and os.path.exists(state.video.evidence_frame_path):
                final_photo_filename = Path(state.video.evidence_frame_path).name
                try:
                    with open(state.video.evidence_frame_path, "rb") as vf:
                        final_photo_bytes = vf.read()
                except Exception:
                    pass

            reg_result = orchestrator.process_registration(
                db=db,
                description=final_desc,
                category=final_cat,
                latitude=latitude,
                longitude=longitude,
                citizen_phone=citizen_phone,
                photo_filename=final_photo_filename,
                photo_bytes=final_photo_bytes,
                allow_duplicate_override=allow_dup_override,
                ward_number=ward_number
            )

            # Check duplicate / moderation rejection
            if reg_result.get("is_duplicate"):
                existing_tid = reg_result.get("ticket_id")
                repeat_cnt = reg_result.get("repeat_count", 1)
                if lang == "mr":
                    reply = (
                        f"⚠️ **या समस्येबाबत या परिसरात आधीच तक्रार नोंदवलेली आहे:**\n\n"
                        f"• **तिकीट क्र:** {existing_tid}\n"
                        f"• **श्रेणी:** {cat_name_current}\n"
                        f"• **स्थिती:** {reg_result.get('status')}\n"
                        f"• **नागरिक पाठबळ (Upvotes):** {repeat_cnt} तक्रारी\n\n"
                        f"महापालिकेचे पथक यावर कार्यरत आहे. जर आपले ठिकाण किंवा समस्या वेगळी असेल, तर **'नवीन तक्रार नोंदवा'** असे सांगा किंवा खालील बटण दाबा."
                    )
                else:
                    reply = (
                        f"⚠️ **A complaint for this issue in this area is already active:**\n\n"
                        f"• **Ticket ID:** {existing_tid}\n"
                        f"• **Category:** {cat_name_current}\n"
                        f"• **Status:** {reg_result.get('status')}\n"
                        f"• **Citizen Reports (Upvotes):** {repeat_cnt}\n\n"
                        f"Our municipal field staff is already assigned. If this is a different spot or issue, tap 'Register New Complaint'."
                    )
                buttons = [
                    {"text": "➕ नवीन तक्रार नोंदवा / New Complaint", "action": "start_new_complaint"},
                    {"text": "🔍 स्थिती तपासा / Track Status", "action": f"track_{existing_tid}"}
                ]
                state.reset_after_registration()
                state.last_duplicate_ticket = existing_tid
                state.active_ticket_id = existing_tid
                return {
                    "reply": reply,
                    "intent": "DUPLICATE_DETECTED",
                    "language": lang,
                    "category": reg_result.get("detected_category"),
                    "category_name": cat_name_current,
                    "ticket_data": reg_result,
                    "buttons": buttons,
                    "action_prompt": "track_or_register_new"
                }

            if reg_result.get("moderation_status") == "REJECTED" or reg_result.get("is_fraud"):
                reason = reg_result.get("fraud_reason") or "Inappropriate content"
                reply = f"🚫 आपली तक्रार नाकारली आहे: {reason}" if lang == "mr" else f"🚫 Complaint rejected: {reason}"
                state.reset_after_registration()
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

            if reg_result.get("verification_status") == "REVIEW_REQUIRED":
                if lang == "mr":
                    reply = (
                        f"📋 **आपली तक्रार पुनरावलोकनासाठी (Review Required) स्वीकारली आहे.**\n\n"
                        f"• **तिकीट क्र:** {reg_result['ticket_id']}\n"
                        f"• **श्रेणी:** {cat_name_current}\n"
                        f"• **स्थिती:** {reg_result.get('status')}\n\n"
                        f"महानगरपालिका अधिकारी या तक्रारीची पडताळणी करून पुढील कारवाई करतील."
                    )
                else:
                    reply = (
                        f"📋 **Your complaint has been submitted for Human Review (Review Required).**\n\n"
                        f"• **Ticket ID:** {reg_result['ticket_id']}\n"
                        f"• **Category:** {cat_name_current}\n"
                        f"• **Status:** {reg_result.get('status')}\n\n"
                        f"A municipal officer will review the evidence and proceed with dispatch."
                    )
            else:
                w_num = reg_result.get('ward') or reg_result.get('ward_number')
                w_name = reg_result.get('ward_name', '')
                w_zone = reg_result.get('zone', '')
                c_lat = reg_result.get('latitude') or latitude
                c_lng = reg_result.get('longitude') or longitude

                ward_text_mr = f"प्रभाग {w_num}" if w_num else "PCMC मध्यवर्ती"
                if w_name:
                    ward_text_mr += f" - {w_name}"
                if w_zone:
                    ward_text_mr += f" ({w_zone})"

                ward_text_en = f"Ward {w_num}" if w_num else "PCMC Central"
                if w_name:
                    ward_text_en += f" - {w_name}"
                if w_zone:
                    ward_text_en += f" ({w_zone})"

                loc_text_mr = f"अक्षांश {c_lat:.4f}, रेखांश {c_lng:.4f}" if (c_lat and c_lng) else "स्थान निश्चित"
                loc_text_en = f"Lat {c_lat:.4f}, Lng {c_lng:.4f}" if (c_lat and c_lng) else "Location Resolved"

                turn_seed = len(getattr(state, "history", [])) or getattr(state, "turn_count", 0)
                reply = human_persona_service.generate_registration_success_message(
                    ticket_id=reg_result['ticket_id'],
                    category_name=cat_name_current,
                    ward_text=ward_text_mr if lang == "mr" else ward_text_en,
                    sla_hours=reg_result.get('sla_hours', 24),
                    worker_name=worker_name,
                    worker_contact=worker_contact,
                    lang=lang,
                    turn_seed=turn_seed,
                    loc_text=loc_text_mr if lang == "mr" else loc_text_en
                )

            state.reset_after_registration()

            return {
                "reply": reply,
                "intent": "COMPLAINT_REGISTERED",
                "language": lang,
                "category": reg_result.get("detected_category"),
                "category_name": cat_name_current,
                "ticket_data": reg_result,
                "action_prompt": "track"
            }

        # 8. Natural Conversational Response (Warm human-like municipal persona answering citizen questions)
        if not detected_category:
            reply = human_persona_service.generate_human_response(text, lang)
            return {
                "reply": reply,
                "intent": "CONVERSATIONAL",
                "language": lang,
                "category": None,
                "category_name": None,
                "ticket_data": None,
                "action_prompt": "describe_or_photo"
            }

        # If a civic grievance category WAS legitimately detected:
        # Specific landmark / address indicators (distinct from generic 'on the road')
        specific_location_patterns = [
            r'\b(?:chowk|nagar|colony|sector|station|prabhag|society|plaza|complex)\b',
            r'(?:चौक|नगर|कॉलनी|सेक्टर|गल्ली|प्रभाग|सोसायटी|संकुल)',
            r'\b(?:near|opposite|behind|beside|at)\s+[A-Za-z0-9\u0900-\u097F]{3,}',
            r'(?:जवळ|समोर|मागे|शेजारी)\s+[A-Za-z0-9\u0900-\u097F]{3,}',
            r'(?<!\bon\s)(?<!\bthe\s)[A-Za-z0-9\u0900-\u097F]{3,}\s+(?:road|street|मार्ग|रस्ता)\b'
        ]
        text_ward = geo_agent.detect_ward_from_text(text)
        detected_ward = text_ward
        if not detected_ward and latitude and longitude and (latitude != 0.0 or longitude != 0.0):
            geo_map = geo_agent.map_coordinates_to_ward(latitude, longitude)
            if not geo_map.get("is_fallback"):
                detected_ward = geo_map

        has_specific_location = bool(text_ward) or any(re.search(p, t_clean) for p in specific_location_patterns)
        has_substantive_desc = bool(has_specific_location and len(t_clean) > 20)

        ward_label = None
        # Only announce the ward in greeting dialogue if the citizen explicitly mentioned a location in their text or confirmed it.
        # Background device/browser GPS coordinates must NOT announce an unconfirmed default ward!
        if text_ward:
            z_str = f" ({text_ward['zone']})" if text_ward.get('zone') else ""
            ward_label = f"प्रभाग {text_ward['ward_number']} - {text_ward['ward_name']}{z_str}"
        elif getattr(state, "location", None) and any(re.search(p, (state.location or "").lower()) for p in specific_location_patterns) and detected_ward:
            z_str = f" ({detected_ward['zone']})" if detected_ward.get('zone') else ""
            ward_label = f"प्रभाग {detected_ward['ward_number']} - {detected_ward['ward_name']}{z_str}"

        turn_seed = len(getattr(state, "history", [])) or getattr(state, "turn_count", 0)
        reply = human_persona_service.generate_grievance_dialogue(
            text=text,
            category=detected_category or "pothole",
            category_name=cat_name_current,
            has_substantive_desc=has_substantive_desc,
            lang=lang,
            detected_ward_name=ward_label,
            turn_seed=turn_seed
        )
        if has_substantive_desc:
            action_prompt = "confirm_register"
            state.set_pending_field("registration_confirmation")
            state.complaint_status = "ready_for_submission"
            if not state.complaint_category and detected_category:
                state.lock_category(detected_category, source="text")
            if not state.description:
                state.description = text
            if not state.location:
                state.location = text
        else:
            action_prompt = "specify_location_and_details"
            state.set_pending_field("location")
            state.complaint_status = "collecting_information"
            if not state.complaint_category and detected_category:
                state.lock_category(detected_category, source="text")
            if not state.description:
                state.description = text

        return {
            "reply": reply,
            "intent": "CONVERSATIONAL",
            "language": lang,
            "category": detected_category,
            "category_name": cat_name_current,
            "ticket_data": None,
            "action_prompt": action_prompt,
            "conversation_state": state.to_dict()
        }


conversational_service = ConversationalService()
