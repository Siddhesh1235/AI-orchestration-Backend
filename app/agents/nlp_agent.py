"""
Multilingual NLP Agent for PCMC Sarathi AI / WardMitra AI.
Extracts intent, detects language (English, Marathi, or Unclear), detects ambiguity,
and determines complaint category with high accuracy.
"""

import re
import yaml
import logging
from typing import Dict, Any, Optional

from app.config.settings import settings
from app.clients.llm_client import llm_client
from app.rules.category_rules import is_ambiguous_light_complaint, get_light_clarification_prompt

logger = logging.getLogger("pcms.nlp_agent")

MARATHI_ROMAN_KEYWORDS = {
    "aahe", "ahe", "nahi", "mala", "amcha", "amchya", "rasta", "khadda",
    "kachra", "pani", "samasya", "durust", "takraar", "takrar", "kara",
    "kiti", "kuthe", "madhe", "jawal", "kela", "zala", "namaskar",
    "kasa", "kashi", "kay", "aani", "pan", "ho", "aata", "vattit", "diva"
}

COMMON_ENGLISH_WORDS = {
    "the", "is", "on", "in", "my", "road", "there", "pothole", "light",
    "street", "water", "drainage", "garbage", "waste", "clean", "off",
    "broken", "please", "help", "hello", "complaint", "issue", "problem",
    "near", "at", "sector", "lane", "choke", "leak", "leaking", "traffic",
    "pole", "wire", "shock", "danger", "burst", "urgent", "not", "working"
}

NON_SUPPORTED_LANGUAGE_WORDS = {
    "bonjour", "merci", "comment", "hola", "amigo", "gracias", "que",
    "por", "favor", "hallo", "danke", "ciao", "guten", "tag", "buenos", "dias"
}


class NLPAgent:
    def __init__(self):
        self.categories_config = self._load_model_config()

    def _load_model_config(self) -> Dict[str, Any]:
        try:
            with open(settings.MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("categories", {})
        except Exception as e:
            logger.warning(f"[NLPAgent] Failed to load model_config.yaml: {e}")
            return {}

    def detect_language(self, text: str) -> str:
        """
        Detects whether input text is primarily 'mr' (Marathi), 'en' (English),
        or 'unclear' (unsupported language, unintelligible symbols, or non-EN/MR).
        """
        if not text or not text.strip():
            return "en"

        cleaned = text.strip().lower()

        # Check for Devanagari script
        devanagari_count = len(re.findall(r'[\u0900-\u097F]', cleaned))
        total_alpha = len(re.findall(r'[a-zA-Z\u0900-\u097F]', cleaned))

        if total_alpha == 0:
            return "unclear"

        if devanagari_count > 0:
            # If significant Devanagari characters, it is Marathi
            return "mr"

        # Check for unsupported foreign languages
        words = re.findall(r'\b\w+\b', cleaned)
        if any(w in NON_SUPPORTED_LANGUAGE_WORDS for w in words):
            return "unclear"

        # Check for transliterated Marathi keywords
        for w in words:
            if w in MARATHI_ROMAN_KEYWORDS:
                return "mr"

        # Check for recognized English words
        has_english_words = any(w in COMMON_ENGLISH_WORDS for w in words)
        if has_english_words:
            return "en"

        # Pure alphabet check: if Latin characters but zero recognizable words and short/odd, check clarity
        latin_count = len(re.findall(r'[a-zA-Z]', cleaned))
        if latin_count / max(1, len(cleaned)) > 0.6:
            # If standard readable text, default to English
            return "en"

        return "unclear"

    def check_ambiguity(self, text: str, lang: str = "en") -> Dict[str, Any]:
        """
        Requirement 1: Checks if complaint is vague (e.g. 'The light is off')
        and requires clarification before proceeding.
        """
        if is_ambiguous_light_complaint(text):
            q = get_light_clarification_prompt(lang)
            return {
                "is_ambiguous": True,
                "clarification_type": "home_vs_streetlight",
                "clarification_question": q
            }

        return {
            "is_ambiguous": False,
            "clarification_type": None,
            "clarification_question": None
        }

    def extract_intent_and_category(self, text: str) -> Dict[str, Any]:
        """
        Parses text to determine:
        1. Language ('en', 'mr', 'unclear')
        2. Ambiguity check
        3. Intent (REGISTER_COMPLAINT, TRACK_COMPLAINT, FEEDBACK, SERVICE_INQUIRY)
        4. Civic Category (one of the 11 target classes)
        """
        cleaned_text = (text or "").strip().lower()
        lang = self.detect_language(cleaned_text)
        ambiguity = self.check_ambiguity(cleaned_text, lang)

        # Step 1: Detect Intent
        intent = self._detect_intent(cleaned_text)

        # Step 2: Try Keyword Matching (Tier 1 - Instant 0-token match)
        category_match = self._keyword_match(cleaned_text)
        if category_match:
            logger.info(f"[NLPAgent] Keyword heuristic matched category: {category_match}")
            return {
                "intent": intent,
                "category": category_match,
                "confidence": 0.88,
                "language": lang,
                "is_ambiguous": ambiguity["is_ambiguous"],
                "clarification_question": ambiguity["clarification_question"],
                "source": "keyword_heuristic"
            }

        # Step 3: LLM Intent & Category Extraction (Tier 2 Ollama -> Tier 3 OpenAI)
        if intent not in ["GREETING", "SMALLTALK", "SERVICE_INQUIRY", "COMPLAINT_PROCESS_GUIDE"]:
            llm_category = self._llm_classify(cleaned_text)
            if llm_category:
                logger.info(f"[NLPAgent] LLM matched category: {llm_category}")
                return {
                    "intent": intent,
                    "category": llm_category,
                    "confidence": 0.82,
                    "language": lang,
                    "is_ambiguous": ambiguity["is_ambiguous"],
                    "clarification_question": ambiguity["clarification_question"],
                    "source": "smart_llm"
                }

        # Step 4: No category matched (Do NOT arbitrarily default to garbage!)
        return {
            "intent": intent,
            "category": None,
            "confidence": 0.0,
            "language": lang,
            "is_ambiguous": ambiguity["is_ambiguous"],
            "clarification_question": ambiguity["clarification_question"],
            "source": "no_civic_category"
        }

    def _detect_intent(self, text: str) -> str:
        clean = text.strip().lower()
        civic_keywords = [
            "pothole", "khadda", "drainage", "gutter", "garbage", "kachra",
            "streetlight", "light", "diwa", "pani", "water", "leak", "wire",
            "traffic", "encroachment", "banner", "खड्डा", "कचरा", "ड्रेनेज",
            "गटर", "पाणी", "लाईट", "दिवा", "तार", "वाहतूक", "अतिक्रमण", "झाड"
        ]
        has_civic = any(w in clean for w in civic_keywords)

        greeting_patterns = [
            r"^(h+[i|y]+|h+e+l+o+|h+e+y+|h+l+o+|g+m+|g+n+|good\s*(morning|afternoon|evening))\b",
            r"^(नमस्कार|रामराम|शुभ\s*सकाळ|शुभ\s*दुपार|शुभ\s*संध्याकाळ|प्रणाम|जय\s*महाराष्ट्र|जय\s*शिवराय|सुप्रभात)\b",
            r"^(namaskar|namaste|radhe\s*radhe|kasa\s*ahes|kashi\s*ahes|how\s*are\s*you|how\s*r\s*u)\b"
        ]
        if not has_civic and any(re.search(p, clean) for p in greeting_patterns):
            return "GREETING"

        if any(w in clean for w in ["track", "status", "तपासा", "स्थिती", "ticket", "तिकीट"]):
            return "TRACK_COMPLAINT"

        if any(w in clean for w in ["feedback", "rating", "अभिप्राय", "स्टार"]):
            return "FEEDBACK_SUBMISSION"

        if any(w in clean for w in ["who are you", "what can you do", "what is your name", "तुम्ही कोण", "नाव काय", "काय करू शकता", "मदत", "help"]):
            return "SERVICE_INQUIRY"

        if any(w in clean for w in ["thank", "thanks", "dhanyavad", "धन्यवाद", "आभार", "bye", "goodbye", "ok", "okay", "fine"]):
            return "SMALLTALK"

        process_inquiry_patterns = [
            r'\bhow\s+(?:to|can\s+i|do\s+i|should\s+i)\s+(?:register|file|lodge|raise|submit|make|put|report|complain)',
            r'\bhow\s+(?:to|can\s+i|do\s+i).*?\b(?:complaint|grievance|ticket)\b',
            r'\b(?:what\s+is\s+the\s+process|registration\s+process|complaint\s+process|steps\s+to\s+register|how\s+does\s+this\s+work)\b',
            r'\b(?:procedure|steps|guide|help)\s+.*?\b(?:register|file|complaint|grievance)\b',
            r'(?:तक्रार\s+कशी|कशी\s+तक्रार|कशी\s+नोंदवा|कशी\s+करायची|कशी\s+करावी|नोंदवण्याची\s+पद्धत|नोंदणी\s+कशी|प्रक्रिया|पायऱ्या|माहिती\s+द्या|कशी\s+नोंदवू)',
            r'(?:मला\s+तक्रार\s+करायची|तक्रार\s+नोंदवायची\s+आहे|कम्प्लेंट\s+कशी|कम्प्लेंट\s+करायची|तक्रार\s+द्यायची)',
            r'\b(?:takrar\s+kashi|kashi\s+takrar|process\s+sanga|step\s*by\s*step|kashi\s+karaychi)\b'
        ]
        if any(re.search(p, clean) for p in process_inquiry_patterns):
            return "COMPLAINT_PROCESS_GUIDE"

        if has_civic or any(w in clean for w in ["complaint", "issue", "problem", "तक्रार", "दुरुस्त", "नुकसान"]):
            return "REGISTER_COMPLAINT"

        return "GENERAL_CONVERSATION"

    def _keyword_match(self, text: str) -> Optional[str]:
        for cat_name, cat_info in self.categories_config.items():
            keywords = cat_info.get("keywords", [])
            for kw in keywords:
                if kw.lower() in text:
                    return cat_name
        return None

    def _llm_classify(self, text: str) -> Optional[str]:
        valid_classes = list(self.categories_config.keys())
        system_prompt = (
            "You are a civic grievance classifier for Pimpri Chinchwad Municipal Corporation (PCMC). "
            "Given a complaint in Marathi, Hindi, or English, output ONLY the single exact matching category name from: "
            f"{', '.join(valid_classes)}. If the input is NOT a civic complaint or problem, output 'NONE'. "
            "Do not write any other explanation or punctuation."
        )
        prompt = f"Complaint: '{text}'\nMatching category name:"
        try:
            result = llm_client.generate(prompt=prompt, system_prompt=system_prompt)
            result_clean = result.strip().lower()
            if "none" in result_clean:
                return None
            for cls in valid_classes:
                if cls in result_clean:
                    return cls
        except Exception as e:
            logger.warning(f"[NLPAgent] LLM classification error: {e}")
        return None


nlp_agent = NLPAgent()
