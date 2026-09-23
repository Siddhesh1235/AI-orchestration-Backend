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
from app.agents.profanity_agent import profanity_agent
from app.config.category_registry import normalize_category_key

# 15 Standard WardMitra Intent Constants (BUG 1)
INTENT_NEW_COMPLAINT = "NEW_COMPLAINT"
INTENT_CATEGORY_SELECTION = "CATEGORY_SELECTION"
INTENT_CATEGORY_CONFIRMATION = "CATEGORY_CONFIRMATION"
INTENT_CATEGORY_CORRECTION = "CATEGORY_CORRECTION"
INTENT_LOCATION_PROVIDED = "LOCATION_PROVIDED"
INTENT_DESCRIPTION_PROVIDED = "DESCRIPTION_PROVIDED"
INTENT_REGISTRATION_CONFIRMATION = "REGISTRATION_CONFIRMATION"
INTENT_REGISTRATION_REJECTION = "REGISTRATION_REJECTION"
INTENT_IMAGE_UPLOADED = "IMAGE_UPLOADED"
INTENT_IMAGE_CLARIFICATION = "IMAGE_CLARIFICATION"
INTENT_STATUS_QUERY = "STATUS_QUERY"
INTENT_FEEDBACK = "FEEDBACK"
INTENT_ABUSIVE_MESSAGE = "ABUSIVE_MESSAGE"
INTENT_GREETING = "GREETING"
INTENT_UNRELATED = "UNRELATED"
INTENT_CANCEL_COMPLAINT = "CANCEL_COMPLAINT"
INTENT_START_NEW_COMPLAINT = "START_NEW_COMPLAINT"
INTENT_CIVIC_KNOWLEDGE_INQUIRY = "CIVIC_KNOWLEDGE_INQUIRY"

YES_CONFIRMATION_WORDS = {
    "yes", "yeah", "y", "sure", "okay", "ok", "correct", "yes correct",
    "register", "register it", "yes please", "confirm", "proceed", "submit",
    "हो", "होय", "हं", "हो हो", "बरोबर", "नक्की", "नोंद करा", "करा", "नोंदवा",
    "chalel", "चालेल", "karun dya", "करून द्या", "done", "plz", "please", "yep",
    "haa", "ha", "ho", "ho na", "barobar", "yes ho", "haa barobar",
    "sahi", "sahi hai", "theek", "thik", "thik hai", "thik ahe"
}

NO_CONFIRMATION_WORDS = {
    "no", "nope", "not this", "not correct", "wrong", "cancel",
    "नाही", "नको", "चुकीचे", "चूक", "बरोबर नाही", "रद्द", "वेगळी", "different", "reject",
    "nahi", "nako", "नहीं", "नहीं चाहिए", "गलत", "गलत है", "na", "nhi", "chook", "galat"
}

CORRECTION_CATEGORY_KEYWORDS = {
    "pothole": ["pothole", "potholes", "khadda", "khadde", "खड्डा", "खड्डे", "road damage", "broken road", "खराब रस्ता"],
    "streetlight": ["streetlight", "street light", "streetlights", "light", "diva", "पथदिवा", "दिवा", "लाईट", "लाईट्स"],
    "garbage": ["garbage", "waste", "kachra", "कचरा", "कचराकुंडी", "dustbin", "kooda", "कूड़ा"],
    "drainage": ["drainage", "gutter", "gatar", "गटार", "ड्रेनेज", "sewer", "naali", "नाली"],
    "water_supply": ["water supply", "drinking water", "low pressure", "no water", "irregular water", "पाणीपुरवठा", "पिण्याचे पाणी", "कमी दाब", "पाणी नाही", "अनियमित पाणी", "नळाला पाणी", "पाणी येत नाही"],
    "pipeline_water_leakage": ["pipeline", "water leak", "leak", "leakage", "pipe burst", "pipe", "गळती", "पाईप फुटली", "पाईपलाईन गळती", "पाईप"],
    "trees": ["tree", "trees", "ped", "पेड़", "झाड", "झाडे", "फांदी", "branch"],
    "traffic_jams": ["traffic", "jam", "वाहतूक", "ट्रॅफिक", "कोंडी", "signal"],
    "noise_pollution": ["noise", "loudspeaker", "speaker", "ध्वनी", "आवाज", "डीजे", "dj"],
    "encroachment": ["encroachment", "footpath", "hawker", "अतिक्रमण", "फुटपाथ"],
    "electricity": ["electricity", "wire", "spark", "current", "विद्युत", "तार", "शॉक"],
    "unauthorized_banner_flex": ["banner", "flex", "hoarding", "बॅनर", "फ्लेक्स", "होर्डिंग"],
    "health_sanitation": ["toilet", "shauchalay", "शौचालय", "स्वच्छता", "sanitation", "health"]
}

logger = logging.getLogger("pcms.nlp_agent")

MARATHI_ROMAN_KEYWORDS = {
    "aahe", "ahe", "nahi", "mala", "amcha", "amchya", "rasta", "khadda",
    "kachra", "pani", "samasya", "durust", "takraar", "takrar", "kara",
    "kiti", "kuthe", "madhe", "jawal", "kela", "zala", "namaskar",
    "kasa", "kashi", "kay", "aani", "pan", "ho", "aata", "vattit", "diva",
    "jhali", "padla", "padli", "bharli", "tumble", "cha", "chi", "che"
}

HINDI_ROMAN_KEYWORDS = {
    "bhai", "hai", "nhi", "nahi", "raha", "rahi", "pada", "sadak", "gaya",
    "karta", "karti", "kooda", "chalu", "bohot", "bahut", "yeh", "ye", "pe",
    "ka", "ki", "ke", "ho", "band", "wala", "wali", "mera", "meri", "karo", "kardo",
    "toot", "kharab", "bada", "phoot", "barbad", "dher"
}

COMMON_ENGLISH_WORDS = {
    "the", "is", "on", "in", "my", "road", "there", "pothole", "light",
    "street", "water", "drainage", "garbage", "waste", "clean", "off",
    "broken", "please", "help", "hello", "complaint", "issue", "problem",
    "near", "at", "sector", "lane", "choke", "leak", "leaking", "traffic",
    "pole", "wire", "shock", "danger", "burst", "urgent", "not", "working",
    "dead", "full", "totally", "check", "status", "of", "track", "ticket",
    "update", "details", "info", "yes", "no", "to", "for", "from", "with",
    "and", "or", "what", "how", "who", "where", "when", "can", "you", "i",
    "me", "give", "tell", "closed", "resolved", "reopen", "rating", "star",
    "good", "bad", "service", "cancel", "withdraw", "guide", "teach", "show", "register"
}

NON_SUPPORTED_LANGUAGE_WORDS = {
    "bonjour", "merci", "comment", "hola", "amigo", "gracias", "que",
    "por", "favor", "hallo", "danke", "ciao", "guten", "tag", "buenos", "dias"
}

INFORMAL_CIVIC_PATTERNS = [
    # Streetlight
    {
        "patterns": [
            r"\bbhai\s+(?:ye\s+)?light\s+band\s+hai\b",
            r"\blight\s+band\s+hai\b",
            r"\blight\s+off\s+hai\b",
            r"\bstreet\s*light\s+nhi\s+chal\s+rahi\b",
            r"\bstreet\s*light\s+nahi\s+chal\s+rahi\b",
            r"\bstreet\s*light\s+band\s+(?:hai|aa?he)\b",
            r"\blight\s+dead\s+hai\b",
            r"\bye\s+light\s+kaam\s+na?hi\s+karti\b",
            r"\blight\s+kaam\s+na?hi\s+kar\s+rahi\b",
            r"\bstreetlight\s+cha\s+problem\s+aa?he\b",
            r"\bstreetlight\s+(?:is\s+)?not\s+working\b",
            r"\bstreet\s+light\s+(?:is\s+)?not\s+working\b",
            r"\bपथदिवा\s+बंद\s+आहे\b",
            r"\bस्ट्रीट\s*लाइट\s+बंद\s+है\b",
            r"\bलाईट\s+बंद\s+आहे\b",
            r"\blight\s+geli\s+aa?he\b",
            r"\bdiva\s+band\s+aa?he\b",
            r"\bpathdiva\s+band\s+aa?he\b",
            r"\blight\s+chalu\s+na?hi\s+hai\b",
            r"\bandhera\s+hai\s+road\s+pe\b",
            r"\bkhamba\s+ki\s+light\s+band\b",
            r"\bpole\s+light\s+off\b",
        ],
        "category": "streetlight",
        "issue": "light_not_working"
    },
    # Pothole
    {
        "patterns": [
            r"\broad\s+pe\s+bada\s+pothole\s+hai\b",
            r"\broad\s+totally\s+kharab\s+hai\b",
            r"\bsadak\s+kharab\s+hai\b",
            r"\bsadak\s+pe\s+gaddha\s+hai\b",
            r"\bsadak\s+pe\s+khadde?\s+hai\b",
            r"\brasta\s+kharab\s+aa?he\b",
            r"\brastyavar\s+khadda\s+padla\s+aa?he\b",
            r"\bkhadde?\s+padle?\s+aa?he\b",
            r"\bpotholes?\s+on\s+road\b",
            r"\bbada\s+pothole\b",
            r"\bbada\s+khadda\b",
            r"\bसड़क\s+पर\s+गड्ढा\s+है\b",
            r"\bरस्त्यावर\s+खड्डा\s+आहे\b",
            r"\bरस्ता\s+खूप\s+खराब\s+आहे\b",
            r"\bसड़क\s+टूटी\s+है\b"
        ],
        "category": "pothole",
        "issue": "broken_road_surface"
    },
    # Garbage
    {
        "patterns": [
            r"\bgarbage\s+pada\s+hai\b",
            r"\bkachra\s+pada\s+hai\b",
            r"(?:kachra.*?jamla|kachra.*?pada|kachryacha.*?dhig)",
            r"\bkachre\s+ka\s+dher\s+hai\b",
            r"\bkooda\s+pada\s+hai\b",
            r"\bkachra\s+na?hi\s+uthaya\b",
            r"\bkachra\s+uchalla\s+nahi\b",
            r"\boverflowing\s+dustbin\b",
            r"\bgarbage\s+dump\b",
            r"\bकूड़ा\s+पड़ा\s+है\b",
            r"(?:कचरा.*?साचला|साचलेला\s+कचरा|कचऱ्याचा\s+ढीग)",
            r"\bकचरा\s+उचलला\s+नाही\b"
        ],
        "category": "garbage",
        "issue": "uncollected_waste"
    },
    # Drainage
    {
        "patterns": [
            r"\bdrain\s+full\s+jam\s+hai\b",
            r"\bdrainage\s+overflow\s+ho\s+raha\b",
            r"\bgatar\s+bharli\s+aa?he\b",
            r"\bgatar\s+tumble\s+aa?he\b",
            r"\bsewer\s+line\s+choked\b",
            r"\bnaali\s+jam\s+hai\b",
            r"\bnaali\s+block\s+hai\b",
            r"\bmanhole\s+(?:cover\s+)?(?:tut\s+gaya|khula\s+hai|open\s+hai)\b",
            r"\bगटार\s+तुंबले\s+आहे\b",
            r"\bनाली\s+जाम\s+है\b",
            r"\bउघडे\s+मॅनहोल\b"
        ],
        "category": "drainage",
        "issue": "blocked_overflowing_drain"
    },
    # Pipeline Water Leakage
    {
        "patterns": [
            r"\bpaani\s+leak\s+ho\s+raha\b",
            r"\bpani\s+leak\s+ho\s+raha\s+hai\b",
            r"\bpani\s+barbad\s+ho\s+raha\b",
            r"\bpipe\s+phoot\s+gaya\b",
            r"\bpipeline\s+burst\b",
            r"\bpani\s+galti\s+aa?he\b",
            r"\bpani\s+vahat\s+aa?he\b",
            r"\bwater\s+leakage\s+on\s+road\b",
            r"\bclean\s+water\s+wasted\b",
            r"\bपानी\s+लीकेज\s+हो\s+रहा\s+है\b",
            r"\bपाणी\s+गळती\s+होत\s+आहे\b",
            r"\bपाईप\s+फुटला\s+आहे\b"
        ],
        "category": "pipeline_water_leakage",
        "issue": "pipe_leakage"
    },
    # Health Sanitation
    {
        "patterns": [
            r"\bpublic\s+toilet\s+ganda\s+hai\b",
            r"\btoilet\s+saf\s+na?hi\s+hai\b",
            r"\bshauchalay\s+ghan\s+aa?he\b",
            r"\bmachhar\s+bohot\s+hai\b",
            r"\bgandagi\s+faili\s+hai\b",
            r"\bसार्वजनिक\s+शौचालय\s+घाण\s+आहे\b",
            r"\bमच्छर\s+पनप\s+रहे\s+हैं\b"
        ],
        "category": "health_sanitation",
        "issue": "unhygienic_public_facility"
    },
    # Trees
    {
        "patterns": [
            r"\bped\s+gir\s+gaya\b",
            r"\btree\s+gir\s+gaya\b",
            r"\bzhad\s+padla\s+aa?he\b",
            r"\bfandi\s+tutli\s+aa?he\b",
            r"\btree\s+branch\s+broken\b",
            r"\bfallen\s+tree\b",
            r"\bझाड\s+पडले\s+आहे\b",
            r"\bपेड़\s+गिर\s+गया\s+है\b"
        ],
        "category": "trees",
        "issue": "fallen_tree_hazard"
    },
    # Traffic
    {
        "patterns": [
            r"\btraffic\s+jam\s+hai\b",
            r"\bfull\s+jam\s+laga\s+hai\b",
            r"\bsignal\s+band\s+hai\b",
            r"\bsignal\s+kaam\s+na?hi\s+kar\s+raha\b",
            r"\bvahtuk\s+kondi\s+jhali\b",
            r"\bवाहतूक\s+कोंडी\b",
            r"\bट्रैफिक\s+जाम\b"
        ],
        "category": "traffic_jams",
        "issue": "severe_traffic_congestion"
    },
    # Electricity
    {
        "patterns": [
            r"\bwire\s+latak\s+rahi\s+hai\b",
            r"\bopen\s+wire\b",
            r"\bsparking\s+ho\s+rahi\b",
            r"\bshort\s*circuit\b",
            r"\bdp\s+spark\b",
            r"\bcurrent\s+lag\s+sakta\b",
            r"\bतार\s+उघडी\s+आहे\b",
            r"\bकरंट\s+लगने\s+का\s+डर\b"
        ],
        "category": "electricity",
        "issue": "electrical_hazard_wires"
    },
    # Encroachment
    {
        "patterns": [
            r"\bfootpath\s+pe\s+dukaan\b",
            r"\bhawkers\s+ne\s+road\s+block\s+kiya\b",
            r"\bfootpath\s+encroachment\b",
            r"\bferiwale\s+bhed\b",
            r"\bअतिक्रमण\b",
            r"\bफुटपाथ\s+पर\s+कब्जा\b"
        ],
        "category": "encroachment",
        "issue": "pedestrian_footpath_encroachment"
    },
    # Banners Flex
    {
        "patterns": [
            r"\billegal\s+banner\b",
            r"\bflex\s+latak\s+raha\b",
            r"\bpolitical\s+hoarding\b",
            r"\bunauthorized\s+banner\b",
            r"\bअनधिकृत\s+बॅनर\b",
            r"\bअवैध\s+होर्डिंग\b"
        ],
        "category": "unauthorized_banner_flex",
        "issue": "illegal_hoarding_flex"
    },
    # Noise Pollution
    {
        "patterns": [
            r"\bdj\s+baj\s+raha\b",
            r"\bloudspeaker\s+band\s+karao\b",
            r"\bshor\s+bohot\s+hai\b",
            r"\bmohtya\s+aawazat\s+speaker\b",
            r"\bध्वनी\s+प्रदूषण\b",
            r"\bतेज\s+आवाज\s+में\s+गाना\b"
        ],
        "category": "noise_pollution",
        "issue": "unauthorized_loudspeaker_noise"
    }
]


def normalize_informal_civic_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Normalizes modern informal / Gen-Z civic complaints (Hindi, Marathi, Hinglish, Marathi-English)
    into structured category and issue_type.
    """
    if not text:
        return None
    cleaned = text.strip().lower()
    for entry in INFORMAL_CIVIC_PATTERNS:
        for pat in entry["patterns"]:
            if re.search(pat, cleaned, re.IGNORECASE):
                return {
                    "category": entry["category"],
                    "issue_type": entry["issue"],
                    "pattern_matched": pat
                }
    return None


class NLPAgent:
    def __init__(self):
        self.categories_config = self._load_model_config()

    def _load_model_config(self) -> Dict[str, Any]:
        try:
            from app.config.category_registry import CIVIC_12_CATEGORIES
            config = {}
            for cat in CIVIC_12_CATEGORIES:
                config[cat["key"]] = {
                    "department": cat["department_code"],
                    "sla_hours": cat["sla_hours"],
                    "keywords": cat.get("keywords", [])
                }
            return config
        except Exception:
            pass

        try:
            with open(settings.MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("categories", {})
        except Exception as e:
            logger.warning(f"[NLPAgent] Failed to load model_config.yaml: {e}")
            return {}

    def detect_language(self, text: str) -> str:
        """
        Detects whether input text is primarily 'mr' (Marathi), 'hi' (Hindi/Hinglish),
        'en' (English), or 'unclear'.
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
            # Check Hindi specific Devanagari words
            if any(w in cleaned for w in ["है", "नहीं", "सड़क", "गड्ढा", "बिजली", "पेड़", "रहा", "रही", "गया", "गई", "बत्ती", "क्या", "मेरा", "मेरी", "मेरे", "दूसरों", "दिखेगा", "होगा", "होगी", "सकता", "सकती", "करना", "करूं", "चाहिए", "मुझे", "किसको"]):
                return "hi"
            return "mr"

        # Check for unsupported foreign languages
        words = re.findall(r'\b\w+\b', cleaned)
        if any(w in NON_SUPPORTED_LANGUAGE_WORDS for w in words):
            return "unclear"

        # Check for transliterated Hindi / Hinglish keywords
        hindi_count = sum(1 for w in words if w in HINDI_ROMAN_KEYWORDS)
        marathi_count = sum(1 for w in words if w in MARATHI_ROMAN_KEYWORDS)

        if hindi_count > 0 and hindi_count >= marathi_count:
            return "hi"
        if marathi_count > 0:
            return "mr"

        # Check for recognized English words
        has_english_words = any(w in COMMON_ENGLISH_WORDS for w in words)
        if has_english_words:
            return "en"

        # Pure alphabet check: if Latin characters but zero recognizable words and short/odd, check clarity
        latin_count = len(re.findall(r'[a-zA-Z]', cleaned))
        if latin_count / max(1, len(cleaned)) > 0.6:
            return "en"

        return "unclear"

    def check_ambiguity(self, text: str, lang: str = "en") -> Dict[str, Any]:
        """
        Checks if complaint is vague (e.g. 'The light is off')
        and requires clarification before proceeding.
        """
        # If text explicitly maps to a clear informal complaint (e.g. "bhai light band hai"), do not flag ambiguous
        if normalize_informal_civic_text(text):
            return {
                "is_ambiguous": False,
                "clarification_type": None,
                "clarification_question": None
            }

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
        1. Language ('en', 'mr', 'hi', 'unclear')
        2. Ambiguity check
        3. Intent (REGISTER_COMPLAINT, TRACK_COMPLAINT, FEEDBACK, SERVICE_INQUIRY)
        4. Civic Category (one of the 12 target classes)
        """
        cleaned_text = (text or "").strip().lower()
        lang = self.detect_language(cleaned_text)
        ambiguity = self.check_ambiguity(cleaned_text, lang)

        # Step 0: Check Gen-Z / Informal Civic Normalizer (Immediate exact match)
        informal_match = normalize_informal_civic_text(cleaned_text)
        if informal_match:
            logger.info(f"[NLPAgent] Informal civic match: {informal_match}")
            return {
                "intent": "REGISTER_COMPLAINT",
                "category": informal_match["category"],
                "issue_type": informal_match["issue_type"],
                "confidence": 0.95,
                "language": lang,
                "is_ambiguous": False,
                "clarification_question": None,
                "source": "informal_slang_normalizer"
            }

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

    def is_complaint_process_inquiry(self, text: str) -> bool:
        """
        Checks if citizen is asking for process guidance or how to register.
        """
        if not text:
            return False
        clean = text.strip().lower()
        patterns = [
            r'\bhow\s+(?:to|can\s+i|do\s+i|should\s+i)\s+(?:register|file|lodge|raise|submit|make|put|report|complain)',
            r'\bhow\s+(?:to|can\s+i|do\s+i).*?\b(?:complaint|grievance|ticket)\b',
            r'\b(?:what\s+is\s+the\s+process|registration\s+process|complaint\s+process|steps\s+to\s+register|how\s+does\s+this\s+work)\b',
            r'\b(?:procedure|steps|guide|help)\s+.*?\b(?:register|file|complaint|grievance)\b',
            r'\b(?:guide\s+me|teach\s+me|show\s+me\s+how|help\s+me\s+register)\b',
            r'(?:तक्रार\s+कशी|कशी\s+तक्रार|कशी\s+नोंदवा|कशी\s+करायची|कशी\s+करावी|नोंदवण्याची\s+पद्धत|नोंदणी\s+कशी|प्रक्रिया|पायऱ्या|माहिती\s+द्या|कशी\s+नोंदवू)',
            r'(?:मला\s+तक्रार\s+करायची|तक्रार\s+नोंदवायची\s+आहे|कम्प्लेंट\s+कशी|कम्प्लेंट\s+करायची|तक्रार\s+द्यायची)',
            r'(?:मला\s+मार्गदर्शन\s*करा|मार्गदर्शन\s*करा|मला\s*शिकवा|शिकवा|समजावून\s*सांगा|कसं\s*करायचं)',
            r'\b(?:मार्गदर्शन|शिकवा)\b',
            r'\b(?:takrar\s+kashi|kashi\s+takrar|process\s+sanga|step\s*by\s*step|kashi\s+karaychi)\b'
        ]
        return any(re.search(p, clean) for p in patterns)

    def _keyword_match(self, text: str) -> Optional[str]:
        if not text:
            return None
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for cat_name, cat_info in self.categories_config.items():
            keywords = cat_info.get("keywords", [])
            for kw in keywords:
                kw_l = kw.lower()
                if kw_l and kw_l in text_lower:
                    # Longer and more specific keywords score higher
                    score = len(kw_l) if len(kw_l) > 3 else 3
                    scores[cat_name] = scores.get(cat_name, 0) + score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
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
    def is_abusive_or_frustrated(self, text: str) -> bool:
        """
        Detects casual insults, hostility, sarcasm or abusive messages (BUG 6).
        Must run BEFORE complaint category processing.
        """
        if not text:
            return False
        clean = text.strip().lower()
        
        # 1. Casual insults / frustration patterns
        abusive_patterns = [
            r"\b(?:are\s+you\s+mad|ru\s+mad|are\s+u\s+mad|u\s+mad|r\s+u\s+mad)\b",
            r"\b(?:stupid|crazy|idiot|dumb|nonsense|shut\s+up|bloody\s+fool|useless|waste\s+of\s+time|blind|fool|moron|brainless)\b",
            r"(?:वेडा\s*आहेस\s*का|वेडा\s*आहे\s*का|तू\s*वेडा|वेडा|मूर्ख|पागल\s*हो\s*क्या|पागल|दिमाग\s*खराब|काही\s*समजत\s*नाही|नालायक|बकवास|वेडेपणा|झाट|भिकार)"
        ]
        if any(re.search(pat, clean, re.IGNORECASE) for pat in abusive_patterns):
            return True
            
        # 2. Profanity check
        try:
            prof_eval = profanity_agent.evaluate_text(clean)
            if prof_eval.get("is_profane"):
                return True
        except Exception:
            pass
            
        return False

    def is_affirmative_yes(self, text: str) -> bool:
        """Deterministic whole-word check for YES variants, including elongated slang (BUG 1)."""
        if not text:
            return False
        clean = text.strip().lower()
        if clean in YES_CONFIRMATION_WORDS:
            return True
        words = set(re.findall(r'[\w\u0900-\u097F]+', clean))
        if any(w in NO_CONFIRMATION_WORDS for w in words):
            return False
        if any(w in YES_CONFIRMATION_WORDS for w in words):
            return True
        # Collapse repeated characters (e.g. 'yesssssssss' -> 'yes', 'yoooo' -> 'yo')
        collapsed_words = {re.sub(r'([a-z])\1+', r'\1', w) for w in words}
        if any(w in YES_CONFIRMATION_WORDS or w in {"yes", "yep", "yeah", "yo"} for w in collapsed_words):
            return True
        # Direct regex check for affirmative slang
        return any(bool(re.match(r'^(y+e+s+|y+e+p+|y+a+h+|y+o+|s+u+r+e+|o+k+a*y*|h+a+|h+o+|h+m+|h+\u0902)$', w)) for w in words)

    def is_negative_no(self, text: str) -> bool:
        """Deterministic whole-word check for NO variants (BUG 4)."""
        if not text:
            return False
        clean = text.strip().lower()
        if clean in NO_CONFIRMATION_WORDS:
            return True
        words = set(re.findall(r'[\w\u0900-\u097F]+', clean))
        return any(w in NO_CONFIRMATION_WORDS for w in words) or any(w in clean for w in ["not correct", "बरोबर नाही", "galat hai"])

    def is_affirmative(self, text: str) -> bool:
        """Alias for is_affirmative_yes."""
        return self.is_affirmative_yes(text)

    def is_negative(self, text: str) -> bool:
        """Alias for is_negative_no."""
        return self.is_negative_no(text)

    def is_cancel_complaint_intent(self, text: str) -> bool:
        """Detects if user explicitly requests cancelling or withdrawing a complaint (BUG 3)."""
        if not text:
            return False
        clean = text.strip().lower()
        cancel_patterns = [
            r"\b(?:cancel|withdraw|close|delete)\s+(?:my\s+|this\s+)?(?:complaint|ticket|grievance)\b",
            r"\b(?:want\s+to\s+)?(?:cancel|withdraw)\s+(?:my\s+|this\s+)?(?:complaint|ticket)\b",
            r"(?:तक्रार\s*(?:रद्द|मागे|कॅन्सल)|तिकीट\s*(?:रद्द|कॅन्सल))",
            r"(?:माझी\s*तक्रार\s*(?:रद्द\s*करा|मागे\s*घ्या)|तक्रार\s*रद्द\s*करा|तक्रार\s*मागे\s*घ्या)",
            r"\b(?:cancel\s*takrar|takrar\s*cancel|takrar\s*radd|radd\s*kara)\b",
            r"\bcancel\b.*?\bwm-\d{8}-\d{4}\b",
            r"\bwm-\d{8}-\d{4}\b.*?\bcancel\b"
        ]
        return any(re.search(p, clean) for p in cancel_patterns)

    def extract_category_correction(self, text: str) -> Optional[str]:
        """
        Extracts corrected category when user says:
        'no, my complaint is about potholes', 'actually about streetlights', 'नाही, खड्डा आहे', etc. (BUG 5)
        Guarded against informational questions and normal Marathi auxiliary verbs ('आली नाही', 'पाणी नाही').
        """
        if not text:
            return None
        clean = text.strip().lower()
        
        # Guard: Inquiries/questions should never be treated as category corrections
        from app.services.rag_service import is_knowledge_inquiry
        if is_knowledge_inquiry(clean):
            return None

        correction_patterns = [
            r"\b(?:no|nope|not\s+this|actually|instead|rather|wrong|wrong\s+category)\b",
            r"^(?:नाही|नको|nahi|nako|no)\b",
            r"(?:^|[,\.\?!])\s*(?:नाही|नको|nahi|nako|no)\b",
            r"(?:चुकीचे\s+आहे|हे\s+नाही|दुसरी\s+तक्रार|वेगळी\s+तक्रार|तक्रार\s+बदला)",
            r"\b(?:complaint\s+is\s+(?:about|actually)|actually\s+about|issue\s+is\s+about)\b",
            r"(?:तक्रार\s+ही|माझी\s+तक्रार|तक्रार\s+.*?(?:बद्दल|आहे))"
        ]
        has_correction_intent = any(re.search(pat, clean, re.IGNORECASE) for pat in correction_patterns)
        if not has_correction_intent:
            return None
            
        for cat_key, kws in CORRECTION_CATEGORY_KEYWORDS.items():
            for kw in kws:
                if re.search(r'\b' + re.escape(kw) + r'\b', clean, re.IGNORECASE) or kw in clean:
                    return normalize_category_key(cat_key)
        return None

    def classify_intent(
        self,
        text: str,
        state: Optional[Any] = None,
        action: Optional[str] = None,
        has_photo: bool = False,
        has_video: bool = False
    ) -> Dict[str, Any]:
        """
        Deterministic Intent Classification Layer (BUG 1, BUG 10).
        Evaluates message intent before complaint category classification.
        Returns standard intent from the 15 standard WardMitra intents.
        """
        clean = (text or "").strip().lower()
        
        # 1. Moderation & Abuse Check First (BUG 6)
        if clean and self.is_abusive_or_frustrated(clean):
            return {
                "intent": INTENT_ABUSIVE_MESSAGE,
                "category": None,
                "confidence": 0.99,
                "reason": "Hostile, sarcastic, or abusive message"
            }
            
        # 2. Cancel / Withdraw Complaint Query (BUG 3)
        if action in ["cancel_complaint", "confirm_cancel"] or self.is_cancel_complaint_intent(clean):
            return {
                "intent": INTENT_CANCEL_COMPLAINT,
                "category": None,
                "confidence": 0.95
            }

        # 2.5 Tracking / Status Query
        if (action and (action == "track" or action.startswith("track_"))) or (clean and any(w in clean for w in ["track", "status", "तपासा", "स्थिती", "ticket", "तिकीट"])) or re.search(r'\bWM-\d{8}-\d{4}\b', clean.upper()):
            return {
                "intent": INTENT_STATUS_QUERY,
                "category": None,
                "confidence": 0.95
            }
            
        # 3. Citizen Feedback / Rating
        if (action and action.startswith("rate_")) or (clean and any(w in clean for w in ["feedback", "rating", "अभिप्राय", "स्टार", "star"])):
            return {
                "intent": INTENT_FEEDBACK,
                "category": None,
                "confidence": 0.95
            }
            
        # 4. Greetings
        if clean and not has_photo and not has_video:
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
                return {
                    "intent": INTENT_GREETING,
                    "category": None,
                    "confidence": 0.95
                }
                
            unrelated_patterns = [
                r"\b(?:what\s+is\s+the\s+weather|weather\s+today|tell\s+me\s+a\s+joke|sing\s+a\s+song|how\s+old\s+are\s+you)\b",
                r"(?:आजचे\s*हवामान|हवामान\s*कसे\s*आहे|विनोद\s*सांगा|गाणे\s*म्हणा|तुम\s*कौन\s*हो|मौसम\s*कैसा\s*है|चुटकुला\s*सुनाओ)"
            ]
            if not has_civic and any(re.search(p, clean) for p in unrelated_patterns):
                return {
                    "intent": INTENT_UNRELATED,
                    "category": None,
                    "confidence": 0.95
                }

        # Check for Ambiguous Light Complaint
        if is_ambiguous_light_complaint(clean):
            return {
                "intent": "CLARIFICATION_REQUIRED",
                "category": "streetlight",
                "confidence": 0.95
            }

        # Check for Explicit Start New Complaint Request (e.g. "नवीन तक्रार नोंदवा", "दुसरी तक्रार", "new complaint")
        new_complaint_patterns = [
            r"\b(?:start|register|file|lodge|create)\s+(?:a\s+)?(?:new|another|different)\s+(?:complaint|ticket|issue|grievance)\b",
            r"\b(?:new|another|different)\s+(?:complaint|ticket|issue|grievance)\b",
            r"(?:नवीन|दुसरी|वेगळी)\s*(?:तक्रार|समस्या)",
            r"(?:नवीन\s*तक्रार\s*नोंदवा|दुसरी\s*तक्रार\s*नोंदवा|नवीन\s*तक्रार\s*करायची|नवीन\s*समस्या\s*आहे)",
            r"(?:दुसरी\s*तक्रार|नवीन\s*तक्रार|दुसरी\s*समस्या)",
            r"(?:नई|दूसरी|अलग)\s*(?:शिकायत|समस्या)",
            r"(?:नई\s*शिकायत\s*दर्ज\s*करें|दूसरी\s*शिकायत\s*दर्ज\s*करें)"
        ]
        if action in ["start_new_complaint", "new_complaint"] or (clean and any(re.search(p, clean, re.IGNORECASE) for p in new_complaint_patterns)):
            return {
                "intent": INTENT_START_NEW_COMPLAINT,
                "category": None,
                "confidence": 0.98
            }

        # Check for Step-by-Step Complaint Process Guidance Inquiry (BUG 4)
        process_inquiry_patterns = [
            r'\bhow\s+(?:to|can\s+i|do\s+i|should\s+i)\s+(?:register|file|lodge|raise|submit|make|put|report|complain)',
            r'\bhow\s+(?:to|can\s+i|do\s+i).*?\b(?:complaint|grievance|ticket)\b',
            r'\b(?:what\s+is\s+the\s+process|registration\s+process|complaint\s+process|steps\s+to\s+register|how\s+does\s+this\s+work)\b',
            r'\b(?:procedure|steps|guide|help)\s+.*?\b(?:register|file|complaint|grievance)\b',
            r'\b(?:guide\s+me|teach\s+me|show\s+me\s+how|walk\s+me\s+through|onboard\s+me)\b',
            r'(?:तक्रार\s+कशी|कशी\s+तक्रार|कशी\s+नोंदवा|कशी\s+करायची|कशी\s+करावी|नोंदवण्याची\s+पद्धत|नोंदणी\s+कशी|प्रक्रिया|पायऱ्या|माहिती\s+द्या|कशी\s+नोंदवू)',
            r'(?:मला\s+(?:एक\s+)?(?:तक्रार|कम्प्लेंट)|तक्रार\s+(?:नोंदवायची|करायची|द्यायची|दाखल\s+करायची)|कम्प्लेंट\s+(?:कशी|करायची|नोंदवायची))',
            r'(?:मला\s*शिकवा|मला\s*मार्गदर्शन\s*करा|मार्गदर्शन\s*करा|समजावून\s*सांगा|शिकवा\s*मला)',
            r'\bi\s+(?:want|need|wish)\s+to\s+(?:register|file|lodge|raise|submit|make)\s+(?:a\s+)?(?:complaint|grievance|ticket)\b',
            r'\b(?:takrar\s+kashi|kashi\s+takrar|process\s+sanga|step\s*by\s*step|kashi\s+karaychi)\b'
        ]
        if any(re.search(p, clean) for p in process_inquiry_patterns):
            return {
                "intent": "COMPLAINT_PROCESS_GUIDE",
                "category": None,
                "confidence": 0.95
            }

        # Check for Smalltalk / Gratitude
        smalltalk_patterns = [
            r"\b(thank\s*you|thanks|thx|dhanyavad|धन्यवाद|आभार|थँक्यू|थँक्स|थॅन्क्स)\b",
            r"^(ok|okay|fine|alright|cool|छान|बरं|ठीक\s*आहे|bye|goodbye|टाटा|अलविदा)[\s!\.]*$"
        ]
        if any(re.search(p, clean) for p in smalltalk_patterns):
            return {
                "intent": "SMALLTALK",
                "category": None,
                "confidence": 0.95
            }

        # Check for Bot Identity / Service Inquiry
        bot_inquiry_patterns = [
            r"\b(?:who\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+your\s+name|who\s+made\s+you|what\s+do\s+you\s+do)\b",
            r"(?:तुम्ही\s*कोण\s*आहात|तुम्ही\s*कोण|नाव\s*काय|काय\s*करू\s*शकता|आप\s*कौन\s*हैं|तुम\s*कौन\s*हो)"
        ]
        if any(re.search(p, clean) for p in bot_inquiry_patterns):
            return {
                "intent": "SERVICE_INQUIRY",
                "category": None,
                "confidence": 0.95
            }
                
        # 5. User Category Correction (BUG 5)
        if clean:
            corrected_category = self.extract_category_correction(clean)
            if corrected_category:
                return {
                    "intent": INTENT_CATEGORY_CORRECTION,
                    "category": corrected_category,
                    "confidence": 0.95
                }
                
        # 6. Pending Field Routing (BUG 3, BUG 4, BUG 9)
        pending_field = getattr(state, "pending_field", None) if state else None
        
        # Registration confirmation step
        if pending_field == "registration_confirmation" or (state and getattr(state, "complaint_status", "") == "ready_for_submission"):
            if action in ["register", "register_new", "force_register", "confirm_register_yes"] or (clean and self.is_affirmative_yes(clean)):
                return {
                    "intent": INTENT_REGISTRATION_CONFIRMATION,
                    "category": getattr(state, "complaint_category", None),
                    "confidence": 0.98
                }
            elif action in ["confirm_register_no", "cancel_register", "cancel", "no"] or (clean and self.is_negative_no(clean)):
                return {
                    "intent": INTENT_REGISTRATION_REJECTION,
                    "category": None,
                    "confidence": 0.98
                }

        # Cancel confirmation step (BUG 3)
        if pending_field == "cancel_confirmation":
            clean_words = set(re.findall(r'[\w\u0900-\u097F]+', clean))
            is_cancel_neg = any(w in {"no", "nope", "keep", "नाही", "नको", "nahi", "nako"} for w in clean_words) or any(p in clean for p in ["don't", "dont", "do not"]) or action in ["confirm_cancel_no", "no"]
            if is_cancel_neg:
                return {
                    "intent": "CANCEL_REJECTED",
                    "category": None,
                    "confidence": 0.98
                }
            elif any(w in {"yes", "yeah", "y", "sure", "ok", "okay", "cancel", "रद्द", "करा", "हो", "होय"} for w in clean_words) or action in ["confirm_cancel_yes", "yes"] or self.is_affirmative_yes(clean):
                return {
                    "intent": "CANCEL_CONFIRMED",
                    "category": getattr(state, "complaint_category", None),
                    "confidence": 0.98
                }

        # Image Category Confirmation step (BUG 2)
        if pending_field == "image_category_confirmation":
            if action in ["confirm_image_category_yes", "yes"] or (clean and self.is_affirmative_yes(clean)):
                return {
                    "intent": INTENT_CATEGORY_CONFIRMATION,
                    "category": getattr(state, "complaint_category", None),
                    "confidence": 0.98
                }
            elif action in ["confirm_image_category_no", "no"] or (clean and self.is_negative_no(clean)):
                return {
                    "intent": INTENT_REGISTRATION_REJECTION,
                    "category": None,
                    "confidence": 0.98
                }
                
        # Video / Category Confirmation step
        if pending_field == "category_confirmation" or (state and getattr(state, "complaint_status", "") == "awaiting_confirmation"):
            if action == "confirm_video_yes" or (clean and self.is_affirmative_yes(clean)):
                return {
                    "intent": INTENT_CATEGORY_CONFIRMATION,
                    "category": getattr(state, "complaint_category", None),
                    "confidence": 0.98
                }
            elif action == "confirm_video_no" or (clean and self.is_negative_no(clean)):
                return {
                    "intent": INTENT_REGISTRATION_REJECTION,
                    "category": None,
                    "confidence": 0.98
                }
                
        # Check if citizen is asking a civic knowledge inquiry / informational question
        from app.services.rag_service import is_knowledge_inquiry
        if clean and not has_photo and not has_video and is_knowledge_inquiry(clean):
            kw_cat_inq = self._keyword_match(clean)
            return {
                "intent": INTENT_CIVIC_KNOWLEDGE_INQUIRY,
                "category": kw_cat_inq,
                "confidence": 0.95
            }

        # Check if citizen is introducing a new or different complaint category even if location/description was pending
        kw_cat = self._keyword_match(clean)
        informal_match = normalize_informal_civic_text(clean)
        matched_new_cat = informal_match["category"] if informal_match else kw_cat
        locked_cat = getattr(state, "complaint_category", None) if state else None
        is_completed_or_idle = getattr(state, "complaint_status", "") in ["completed", "idle"]

        if matched_new_cat and (not locked_cat or is_completed_or_idle or matched_new_cat != locked_cat):
            return {
                "intent": INTENT_NEW_COMPLAINT,
                "category": matched_new_cat,
                "issue_type": informal_match.get("issue_type") if informal_match else None,
                "confidence": 0.95
            }

        # Location provided step (BUG 3)
        if pending_field == "location" and clean:
            return {
                "intent": INTENT_LOCATION_PROVIDED,
                "location": text.strip(),
                "category": getattr(state, "complaint_category", None),
                "confidence": 0.95
            }
            
        # Description provided step
        if pending_field == "description" and clean:
            return {
                "intent": INTENT_DESCRIPTION_PROVIDED,
                "description": text.strip(),
                "category": getattr(state, "complaint_category", None),
                "confidence": 0.95
            }
            
        # Category selection step
        if pending_field == "category" and clean:
            matched_cat = self._keyword_match(clean)
            if matched_cat:
                return {
                    "intent": INTENT_CATEGORY_SELECTION,
                    "category": matched_cat,
                    "confidence": 0.95
                }
                
        # 7. Media Uploaded
        if has_photo or has_video:
            return {
                "intent": INTENT_IMAGE_UPLOADED,
                "category": getattr(state, "complaint_category", None),
                "confidence": 0.90
            }
            
        # 8. Category Selection button or direct name
        if action == "select_category" or (action and action.startswith("select_video_candidate_")):
            cat_k = action.replace("select_video_candidate_", "") if action.startswith("select_video_candidate_") else clean
            norm_k = normalize_category_key(cat_k)
            return {
                "intent": INTENT_CATEGORY_SELECTION,
                "category": norm_k,
                "confidence": 0.95
            }
            
        # Check if bare word is a category name
        if clean in [
            "drainage", "ड्रेनेज", "pothole", "खड्डा", "garbage", "कचरा", "water", "पाणी",
            "streetlight", "street light", "पथदिवा", "trees", "झाड", "traffic", "वाहतूक"
        ]:
            norm_k = normalize_category_key(clean)
            return {
                "intent": INTENT_CATEGORY_SELECTION,
                "category": norm_k,
                "confidence": 0.95
            }
            
        # 9. New Complaint detection
        informal_match = normalize_informal_civic_text(clean)
        if informal_match:
            return {
                "intent": INTENT_NEW_COMPLAINT,
                "category": informal_match["category"],
                "issue_type": informal_match["issue_type"],
                "confidence": 0.95
            }
            
        kw_cat = self._keyword_match(clean)
        locked_cat = getattr(state, "complaint_category", None) if state else None
        is_locked = getattr(state, "category_locked", False) if state else False
        complaint_status = getattr(state, "complaint_status", "") if state else ""

        if kw_cat:
            # If not locked, or previous complaint was finished/idle, or user explicitly mentions a DIFFERENT category
            if not is_locked or complaint_status in ["completed", "idle"] or kw_cat != locked_cat:
                return {
                    "intent": INTENT_NEW_COMPLAINT,
                    "category": kw_cat,
                    "confidence": 0.88
                }
            
        # If user message is smalltalk/gratitude
        if clean and any(w in clean for w in ["thank", "thanks", "dhanyavad", "धन्यवाद", "आभार", "bye", "goodbye"]):
            return {
                "intent": INTENT_UNRELATED,
                "category": None,
                "confidence": 0.90
            }
            
        # Default intent based on state
        if state and getattr(state, "category_locked", False):
            # Category is locked, ordinary message
            return {
                "intent": INTENT_DESCRIPTION_PROVIDED if getattr(state, "pending_field", None) == "description" else INTENT_LOCATION_PROVIDED if getattr(state, "pending_field", None) == "location" else INTENT_NEW_COMPLAINT,
                "category": getattr(state, "complaint_category", None),
                "confidence": 0.80
            }
            
        return {
            "intent": INTENT_NEW_COMPLAINT if kw_cat else INTENT_UNRELATED,
            "category": kw_cat,
            "confidence": 0.70
        }


nlp_agent = NLPAgent()
