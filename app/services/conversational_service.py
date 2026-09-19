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
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.orchestrator.orchestrator import orchestrator
from app.agents.nlp_agent import nlp_agent
from app.agents.image_agent import image_agent
from app.agents.routing_agent import routing_agent
from app.database.models import Complaint
from app.agents.geo_agent import geo_agent
from app.clients.llm_client import llm_client
from app.services.human_persona_service import human_persona_service
from app.rules.category_rules import (
    is_ambiguous_light_complaint,
    get_light_clarification_prompt,
    get_streetlight_evidence_prompt,
    get_category_clarification_prompt
)
from app.config.settings import settings

logger = logging.getLogger("pcms.conversational")

CATEGORY_DISPLAY_NAMES = {
    "streetlight": {"en": "Streetlight", "mr": "पथदिवा (Streetlight)"},
    "damaged_streetlights": {"en": "Streetlight", "mr": "पथदिवा (Streetlight)"},
    "pothole": {"en": "Pothole & Road Damage", "mr": "खड्डा व रस्ता दुरुस्ती"},
    "potholes": {"en": "Pothole & Road Damage", "mr": "खड्डा व रस्ता दुरुस्ती"},
    "road_damage": {"en": "Road Damage", "mr": "रस्ता दुरुस्ती"},
    "garbage": {"en": "Garbage & Cleanliness", "mr": "कचरा व स्वच्छता"},
    "overflowing_garbage": {"en": "Garbage & Cleanliness", "mr": "कचरा व स्वच्छता"},
    "illegal_debris_dumping": {"en": "Illegal Debris Dumping", "mr": "अनधिकृत मलबा व कचरा"},
    "drainage": {"en": "Drainage & Sewerage", "mr": "ड्रेनेज व सांडपाणी"},
    "drainage_failures": {"en": "Drainage & Sewerage", "mr": "ड्रेनेज व सांडपाणी"},
    "pipeline_water_leakage": {"en": "Water Pipeline Leakage", "mr": "पाणी गळती व पुरवठा"},
    "water_pipeline_leakages": {"en": "Water Pipeline Leakage", "mr": "पाणी गळती व पुरवठा"},
    "traffic_jams": {"en": "Traffic Congestion", "mr": "वाहतूक कोंडी"},
    "trees": {"en": "Fallen Trees & Garden", "mr": "झाड पडणे व उद्यान"},
    "noise_pollution": {"en": "Noise Pollution", "mr": "ध्वनी प्रदूषण"},
    "encroachment": {"en": "Footpath Encroachment", "mr": "अतिक्रमण"},
    "electricity": {"en": "Electric Hazard / DP", "mr": "विद्युत धोका / डीपी"},
    "unauthorized_banner_flex": {"en": "Illegal Banners & Flex", "mr": "अनधिकृत फ्लेक्स व बॅनर"},
    "banners_flex": {"en": "Illegal Banners & Flex", "mr": "अनधिकृत फ्लेक्स व बॅनर"},
    "health_sanitation": {"en": "Garbage & Sanitation", "mr": "आरोग्य व स्वच्छता"},
    "pipelinedefects": {"en": "Water Pipeline Leakage", "mr": "पाणी गळती व पुरवठा"},
    "road_incidents_traffic": {"en": "Traffic & Road Incidents", "mr": "वाहतूक कोंडी व रस्ता समस्या"}
}


class ConversationalService:
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
            r'(?:तक्रार\s+कशी|कशी\s+तक्रार|कशी\s+नोंदवा|कशी\s+करायची|कशी\s+करावी|नोंदवण्याची\s+पद्धत|नोंदणी\s+कशी|प्रक्रिया|पायऱ्या|माहिती\s+द्या|कशी\s+नोंदवू)',
            r'(?:मला\s+(?:एक\s+)?(?:तक्रार|कम्प्लेंट)|तक्रार\s+(?:नोंदवायची|करायची|द्यायची|दाखल\s+करायची)|कम्प्लेंट\s+(?:कशी|करायची|नोंदवायची)|नवीन\s+तक्रार\s+करायची)',
            r'\bi\s+(?:want|need|wish)\s+to\s+(?:register|file|lodge|raise|submit|make)\s+(?:a\s+)?(?:complaint|grievance|ticket)\b',
            r'\b(?:takrar\s+kashi|kashi\s+takrar|process\s+sanga|step\s*by\s*step|kashi\s+karaychi)\b'
        ]
        return any(re.search(p, cleaned) for p in patterns)

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
        action: Optional[str] = None,
        ward_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main conversation controller fulfilling WardMitra AI municipal requirements.
        """
        text = (message or "").strip()
        lang = self.detect_language(text)

        # If user asks how to register / procedure / step-by-step guidance, clear any category override
        if text and self.is_complaint_process_inquiry(text):
            category = None

        # 0. Unclear Language Handling (Requirement 3 & Test 5)
        if lang == "unclear" and text and not self.is_greeting(text):
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

        # 1. Greetings & Personal Introductions (Do NOT create complaints on greetings)
        intro_name = self.extract_name_introduction(text)
        if text and intro_name:
            if lang == "mr":
                reply = (
                    f"नमस्कार {intro_name}! 👋 PCMC सारथी (वॉर्डमित्र) मध्ये आपले मनःपूर्वक स्वागत आहे.\n\n"
                    "मी पिंपरी चिंचवड महानगरपालिकेचा अधिकृत नागरी सहाय्यक आहे. मी आज आपली काय मदत करू शकतो? "
                    "आपण खड्डे, पथदिवे, कचरा, पाणीपुरवठा किंवा ड्रेनेज यांसारखी कोणतीही नागरी समस्या मला सांगू शकता किंवा प्रश्न विचारू शकता."
                )
            else:
                reply = (
                    f"Hello {intro_name}! 👋 Welcome to PCMC Sarathi (WardMitra AI).\n\n"
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
                salutation = f"नमस्कार {name_token}! 👋" if name_token else "नमस्कार! 👋"
                reply = (
                    f"{salutation} मी PCMC सारथी / वॉर्डमित्र AI सहाय्यक आहे.\n\n"
                    "मी पिंपरी चिंचवड मधील रस्ते, पथदिवे, कचरा, ड्रेनेज, पाणी पुरवठा अशा नागरी समस्या सोडवण्यात मदत करतो. "
                    "मी आज आपली काय मदत करू शकतो?"
                )
            else:
                salutation = f"Hello {name_token}! 👋" if name_token else "Hello! 👋"
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
                reply = "आपले स्वागत आहे! पिंपरी चिंचवड परिसरातील कोणत्याही नागरी समस्येसाठी कधीही संपर्क करा. मी सदैव सेवेत आहे."
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
            evidence_prompt = get_streetlight_evidence_prompt(lang)
            return {
                "reply": evidence_prompt,
                "intent": "EVIDENCE_REQUIRED",
                "language": lang,
                "category": "streetlight",
                "category_name": CATEGORY_DISPLAY_NAMES["streetlight"][lang],
                "ticket_data": None,
                "action_prompt": "upload_photo"
            }

        # 5. Category Button / Bare Category Selection (Requirement 2 & Test 2)
        # e.g. User selects "Drainage" chip or types just "Drainage" / "ड्रेनेज"
        bare_category_target = None
        if action == "select_category" and category:
            bare_category_target = category
        elif t_clean in ["drainage", "ड्रेनेज", "pothole", "खड्डा", "garbage", "कचरा", "water", "पाणी"]:
            bare_category_target = t_clean

        if bare_category_target and not photo_bytes:
            clarification_prompt = get_category_clarification_prompt(bare_category_target, lang)
            cat_display = CATEGORY_DISPLAY_NAMES.get(bare_category_target, {}).get(lang, bare_category_target)
            return {
                "reply": clarification_prompt,
                "intent": "CATEGORY_CLARIFICATION_REQUIRED",
                "language": lang,
                "category": bare_category_target,
                "category_name": cat_display,
                "ticket_data": None,
                "action_prompt": "describe_problem"
            }

        # Detect Civic Category from NLP or Category input
        detected_category = category
        if not detected_category and text:
            for cat_key in CATEGORY_DISPLAY_NAMES.keys():
                if cat_key in t_clean:
                    detected_category = cat_key
                    break
            if not detected_category:
                nlp_res = nlp_agent.extract_intent_and_category(text)
                detected_category = nlp_res.get("category")

        cat_info = CATEGORY_DISPLAY_NAMES.get(detected_category, {"en": "Civic Grievance", "mr": "नागरी समस्या"})
        cat_name_current = cat_info.get(lang, cat_info["en"])

        # 6. Image Evidence Verification & Low Confidence Check (Requirements 4, 5 & Tests 6, 12)
        saved_tmp_photo_path = None
        if photo_filename and photo_bytes:
            unique_tmp_filename = f"chat_verify_{routing_agent.generate_ticket_id()}_{photo_filename}"
            tmp_path = Path(settings.UPLOAD_DIR) / unique_tmp_filename
            with open(tmp_path, "wb") as f:
                f.write(photo_bytes)
            saved_tmp_photo_path = str(tmp_path)

            img_eval = image_agent.classify_image(saved_tmp_photo_path)
            target_cat_for_eval = detected_category or "streetlight" if "light" in t_clean else (detected_category or "pothole")
            ev_result = image_agent.verify_evidence(target_cat_for_eval, img_eval, min_confidence=0.30)

            # Check 1: Irrelevant evidence (e.g. food plate for streetlight) -> Reject!
            if not ev_result["is_relevant"]:
                reply_msg = ev_result["message_mr"] if lang == "mr" else ev_result["message_en"]
                return {
                    "reply": reply_msg,
                    "intent": "EVIDENCE_REJECTED",
                    "language": lang,
                    "category": target_cat_for_eval,
                    "category_name": cat_name_current,
                    "ticket_data": None,
                    "action_prompt": "upload_valid_photo"
                }

            # Check 2: Low-confidence image -> Request clearer photo
            if not ev_result["is_confident"]:
                reply_msg = ev_result["message_mr"] if lang == "mr" else ev_result["message_en"]
                return {
                    "reply": reply_msg,
                    "intent": "LOW_CONFIDENCE_IMAGE",
                    "language": lang,
                    "category": target_cat_for_eval,
                    "category_name": cat_name_current,
                    "ticket_data": None,
                    "action_prompt": "upload_clear_photo"
                }

        # 7. Registration Flow (Only for verified evidence & confirmed complaints)
        user_wants_registration = self.is_registration_intent(text, action, confirm_register) or bool(photo_bytes)
        if user_wants_registration and (text or photo_bytes):
            final_cat = detected_category or "pothole"
            if len(text.strip()) < 5 or text.lower().strip() in ["हो", "होय", "yes", "yep", "sure", "करा", "नोंदवा", "submit", "register", "urgent", "proceed"]:
                final_desc = f"पिंपरी चिंचवड परिसरातील {cat_name_current} संदर्भात नागरी तक्रार." if lang == "mr" else f"Civic grievance regarding {cat_info['en']} in PCMC area."
            else:
                final_desc = text

            allow_dup_override = action in ["register_new", "force_register"]
            reg_result = orchestrator.process_registration(
                db=db,
                description=final_desc,
                latitude=latitude,
                longitude=longitude,
                citizen_phone=citizen_phone,
                photo_filename=photo_filename,
                photo_bytes=photo_bytes,
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
                        f"महापालिकेचे पथक यावर कार्यरत आहे. जर आपले ठिकाण किंवा समस्या वेगळी असेल, तर **'नवीन तक्रार नोंदवा'** असे सांगा."
                    )
                else:
                    reply = (
                        f"⚠️ **A complaint for this issue in this area is already active:**\n\n"
                        f"• **Ticket ID:** {existing_tid}\n"
                        f"• **Category:** {cat_name_current}\n"
                        f"• **Status:** {reg_result.get('status')}\n"
                        f"• **Citizen Reports (Upvotes):** {repeat_cnt}\n\n"
                        f"Our municipal field staff is already assigned. If this is a different spot, reply with 'Register as new complaint'."
                    )
                return {
                    "reply": reply,
                    "intent": "DUPLICATE_DETECTED",
                    "language": lang,
                    "category": reg_result.get("detected_category"),
                    "category_name": cat_name_current,
                    "ticket_data": reg_result,
                    "action_prompt": "track_or_register_new"
                }

            if reg_result.get("moderation_status") == "REJECTED" or reg_result.get("is_fraud"):
                reason = reg_result.get("fraud_reason") or "Inappropriate content"
                reply = f"🚫 आपली तक्रार नाकारली आहे: {reason}" if lang == "mr" else f"🚫 Complaint rejected: {reason}"
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

                if lang == "mr":
                    reply = (
                        f"🎉 **आपली तक्रार यशस्वीरित्या नोंदवली गेली आहे!**\n\n"
                        f"• **तिकीट क्र:** {reg_result['ticket_id']}\n"
                        f"• **श्रेणी:** {cat_name_current}\n"
                        f"• **🏛️ प्रभाग (Ward):** {ward_text_mr}\n"
                        f"• **📍 GPS स्थान (Location):** {loc_text_mr}\n"
                        f"• **👷 नेमलेले क्षेत्रीय कामगार:** {worker_name} ({worker_contact})\n"
                        f"• **⏱️ निकालाची मुदत (SLA):** {reg_result.get('sla_hours', 24)} तास\n\n"
                        f"कामगार लवकरच समस्येचे निवारण करतील."
                    )
                else:
                    reply = (
                        f"🎉 **Your complaint has been successfully registered!**\n\n"
                        f"• **Ticket ID:** {reg_result['ticket_id']}\n"
                        f"• **Category:** {cat_name_current}\n"
                        f"• **🏛️ Ward:** {ward_text_en}\n"
                        f"• **📍 GPS Location:** {loc_text_en}\n"
                        f"• **👷 Assigned Worker:** {worker_name} ({worker_contact})\n"
                        f"• **⏱️ Resolution Deadline (SLA):** {reg_result.get('sla_hours', 24)} Hours\n\n"
                        f"The assigned worker has been dispatched."
                    )

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
        has_location_clue = any(loc in t_clean for loc in [
            "road", "street", "chowk", "nagar", "colony", "sector", "lane", "near", "opposite", "area", "ward",
            "रस्ता", "चौक", "नगर", "कॉलनी", "सेक्टर", "गल्ली", "जवळ", "समोर", "परिसर", "प्रभाग", "गाव"
        ])
        has_substantive_desc = len(t_clean) > 20 or has_location_clue

        detected_ward = geo_agent.detect_ward_from_text(text)
        if not detected_ward and latitude and longitude and (latitude != 0.0 or longitude != 0.0):
            geo_map = geo_agent.map_coordinates_to_ward(latitude, longitude)
            if not geo_map.get("is_fallback"):
                detected_ward = geo_map

        ward_label = None
        if detected_ward:
            z_str = f" ({detected_ward['zone']})" if detected_ward.get('zone') else ""
            ward_label = f"प्रभाग {detected_ward['ward_number']} - {detected_ward['ward_name']}{z_str}"

        reply = human_persona_service.generate_grievance_dialogue(
            text=text,
            category=detected_category or "pothole",
            category_name=cat_name_current,
            has_substantive_desc=has_substantive_desc,
            lang=lang,
            detected_ward_name=ward_label
        )
        action_prompt = "confirm_register" if has_substantive_desc else "specify_location_and_details"

        return {
            "reply": reply,
            "intent": "CONVERSATIONAL",
            "language": lang,
            "category": detected_category,
            "category_name": cat_name_current,
            "ticket_data": None,
            "action_prompt": action_prompt
        }


conversational_service = ConversationalService()
