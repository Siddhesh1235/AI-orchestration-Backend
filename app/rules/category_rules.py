"""
Category Rules & Clarification Definitions for WardMitra AI.
Ensures complaints are not accepted prematurely without required context and evidence.
"""

import re
from typing import Optional, Dict, Any

# Ambiguous triggers that MUST NOT create a complaint directly
AMBIGUOUS_LIGHT_PATTERNS = [
    r"\b(the\s+)?light(\s+is)?\s+off\b",
    r"\blight\s+band\s+aa?he\b",
    r"\blight\s+geli\b",
    r"\bbulb\s+off\b",
    r"\b(दिवा|लाईट|लाईट)\s+(गेली|बंद\s*आहे|नाही)\b"
]

CATEGORY_CLARIFICATION_PROMPTS = {
    "drainage": {
        "en": "Please describe the actual drainage problem, such as a blocked drain, waterlogging, overflowing drain, or sewage leakage.",
        "mr": "कृपया ड्रेनेजची नेमकी समस्या सांगा, जसे की तुंबलेले गटार, रस्त्यावर पाणी साचणे, उघडे मॅनहोल किंवा सांडपाणी गळती."
    },
    "drainage_failures": {
        "en": "Please describe the actual drainage problem, such as a blocked drain, waterlogging, overflowing drain, or sewage leakage.",
        "mr": "कृपया ड्रेनेजची नेमकी समस्या सांगा, जसे की तुंबलेले गटार, रस्त्यावर पाणी साचणे, उघडे मॅनहोल किंवा सांडपाणी गळती."
    },
    "streetlight": {
        "en": "Please describe the actual streetlight issue, such as dark street, flickering lamp, or pole damage, and send a clear photo.",
        "mr": "कृपया पथदिव्याची नेमकी समस्या सांगा (उदा. पथदिवा बंद, लुकलुकणारा दिवा किंवा खांब नादुरुस्त) आणि स्पष्ट फोटो पाठवा."
    },
    "damaged_streetlights": {
        "en": "Please describe the actual streetlight issue, such as dark street, flickering lamp, or pole damage, and send a clear photo.",
        "mr": "कृपया पथदिव्याची नेमकी समस्या सांगा (उदा. पथदिवा बंद, लुकलुकणारा दिवा किंवा खांब नादुरुस्त) आणि स्पष्ट फोटो पाठवा."
    },
    "pothole": {
        "en": "Please describe the road damage or pothole problem and share a clear photo of the road surface.",
        "mr": "कृपया रस्त्यावरील खड्ड्याची नेमकी समस्या सांगा आणि खड्ड्याचा स्पष्ट फोटो जोडा."
    },
    "potholes": {
        "en": "Please describe the road damage or pothole problem and share a clear photo of the road surface.",
        "mr": "कृपया रस्त्यावरील खड्ड्याची नेमकी समस्या सांगा आणि खड्ड्याचा स्पष्ट फोटो जोडा."
    },
    "road_damage": {
        "en": "Please describe the road damage or pothole problem and share a clear photo of the road surface.",
        "mr": "कृपया रस्त्यावरील खड्ड्याची नेमकी समस्या सांगा आणि खड्ड्याचा स्पष्ट फोटो जोडा."
    },
    "garbage": {
        "en": "Please describe the actual waste issue, such as overflowing garbage bin, uncollected waste, or illegal dumping.",
        "mr": "कृपया कचऱ्याची नेमकी समस्या सांगा, जसे की कचराकुंडी भरून वाहणे, कचरा न उचलणे किंवा अनधिकृत कचरा टाकणे."
    },
    "overflowing_garbage": {
        "en": "Please describe the actual waste issue, such as overflowing garbage bin, uncollected waste, or illegal dumping.",
        "mr": "कृपया कचऱ्याची नेमकी समस्या सांगा, जसे की कचराकुंडी भरून वाहणे, कचरा न उचलणे किंवा अनधिकृत कचरा टाकणे."
    },
    "pipeline_water_leakage": {
        "en": "Please describe the water problem, such as a pipeline burst, clean drinking water leakage, or low pressure.",
        "mr": "कृपया पाणी समस्येचे स्वरूप सांगा, जसे की पाईपलाईन फुटणे, पिण्याच्या पाण्याची गळती किंवा कमी दाबाने पाणी येणे."
    },
    "water_pipeline_leakages": {
        "en": "Please describe the water problem, such as a pipeline burst, clean drinking water leakage, or low pressure.",
        "mr": "कृपया पाणी समस्येचे स्वरूप सांगा, जसे की पाईपलाईन फुटणे, पिण्याच्या पाण्याची गळती किंवा कमी दाबाने पाणी येणे."
    }
}


def is_ambiguous_light_complaint(text: str) -> bool:
    """Detects if user said something vague like 'The light is off'."""
    t = (text or "").lower().strip()
    for pat in AMBIGUOUS_LIGHT_PATTERNS:
        if re.search(pat, t):
            # Check if user already clarified "streetlight" or "street" or "home"
            if "streetlight" in t or "street" in t or "road" in t or "रस्ता" in t or "पथदिवा" in t:
                return False
            return True
    return False


def get_light_clarification_prompt(lang: str = "en") -> str:
    """Returns question asking whether home light or streetlight."""
    if lang == "mr":
        return "हा घरातील दिवा आहे की रस्त्यावरील पथदिवा?"
    return "Is this a home light or a streetlight?"


def get_streetlight_evidence_prompt(lang: str = "en") -> str:
    """Returns prompt requesting streetlight evidence photo."""
    if lang == "mr":
        return "कृपया पथदिव्याचा किंवा संबंधित विद्युत/मीटर पेटीचा स्पष्ट फोटो पाठवा."
    return "Please send a clear photo of the streetlight or the related electrical/meter box."


try:
    from app.config.category_registry import normalize_category_key, get_category_info
except ImportError:
    normalize_category_key = None
    get_category_info = None


def get_category_clarification_prompt(category: str, lang: str = "en") -> str:
    """Returns clarification question for bare category selections."""
    cat_key = (category or "").lower().strip()
    prompts = CATEGORY_CLARIFICATION_PROMPTS.get(cat_key)
    if prompts:
        return prompts.get(lang, prompts["en"])
    
    if get_category_info and normalize_category_key:
        norm = normalize_category_key(cat_key)
        info = get_category_info(norm)
        if info:
            if lang == "mr" and "clarification_prompt_mr" in info:
                return info["clarification_prompt_mr"]
            elif lang == "hi" and "clarification_prompt_hi" in info:
                return info["clarification_prompt_hi"]
            elif "clarification_prompt_en" in info:
                return info["clarification_prompt_en"]
    
    if lang == "mr":
        return f"कृपया {cat_key} संदर्भातील आपली नेमकी समस्या आणि ठिकाण सांगा."
    if lang == "hi":
        return f"कृपया {cat_key} से संबंधित अपनी सटीक समस्या और स्थान बताएं।"
    return f"Please describe the specific issue with {cat_key} and its location."

