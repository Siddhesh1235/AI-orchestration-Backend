"""
Multilingual Profanity & Abusive Language Filter for PCMC Sarathi AI.
Provides strict detection for Marathi, Hindi, and English profanities, hate speech,
harassment, and vulgar slurs to protect municipal staff and ensure civility.
"""

import re
import unicodedata
import logging
from typing import Dict, Any, List, Set, Optional

logger = logging.getLogger("pcms.profanity_agent")

# Curated Multilingual Profanity & Slur Lexicons
# Grouped by language with phonetic / transliterated equivalents

ENGLISH_PROFANITIES: Set[str] = {
    # Severe profanities & vulgarities
    "fuck", "fucking", "fucked", "fucker", "motherfucker", "shit", "bullshit",
    "bitch", "bitches", "asshole", "bastard", "cunt", "dick", "pussy",
    "whore", "slut", "moron", "retard", "faggot", "idiot", "bloody fool",
    "dumbass", "wanker", "prick", "cock", "jackass"
}

HINDI_PROFANITIES: Set[str] = {
    # Devanagari Hindi slurs & abuses
    "मादरचोद", "भोसडीके", "बहनचोद", "चूतिया", "गांडू", "हरामी",
    "कुत्ता", "कमीना", "रांड", "रंडी", "लौड़ा", "लंड", "झांट",
    "भडवा", "गधा", "सुअर", "हरामखोर", "छक्का", "हिजड़ा",
    # Transliterated Hinglish
    "madarchod", "mc", "bhosdike", "bhenchod", "bc", "chutiya", "gaandu",
    "gandu", "harami", "kutta", "kamina", "randi", "lauda", "lodu", "lund",
    "bhadwa", "haramkhor", "behenchod", "bsdk", "maderchod", "bhosdi"
}

MARATHI_PROFANITIES: Set[str] = {
    # Devanagari Marathi slurs & abuses
    "झवाड्या", "आईघाल्या", "रानड्या", "भिकारचोट", "चावट",
    "रांडच्या", "गांड", "झवला", "झवून", "माकडतोंड्या",
    "भडव्या", "चाट्या", "कमीन्या", "नाठाळ", "नालायक", "हरामखोर",
    "आईझवल्या", "लांड्या", "गांडमऱ्या", "भिकारड्या",
    # Transliterated Marathi
    "zhavadya", "aaighalya", "aaizhavlya", "randchya", "bhikarchot",
    "bhadavya", "gandmarya", "chatya", "nakatyachya", "ranadya"
}

# Combine all target profanities
ALL_PROFANITIES: Set[str] = ENGLISH_PROFANITIES | HINDI_PROFANITIES | MARATHI_PROFANITIES

# Regex substitutions to normalize common obfuscations (e.g. f*ck, b!tch, a$$hole)
LEET_SUBS = {
    "@": "a", "$": "s", "0": "o", "1": "i", "!": "i", "3": "e", "5": "s", "*": "",
    "+": "t", "#": "h", "_": "", "-": ""
}


class ProfanityAgent:
    """Agent specialized in evaluating multilingual textual decency and civility."""

    def __init__(self):
        self.profanities = ALL_PROFANITIES
        logger.info(f"[ProfanityAgent] Loaded {len(self.profanities)} multilingual profanity roots.")

    def normalize_text(self, text: str) -> str:
        """Normalizes Unicode characters, strips accents, removes punctuation tricks and repeated characters."""
        if not text:
            return ""
        
        # 1. Normalize Unicode forms
        normalized = unicodedata.normalize("NFKD", text).strip().lower()

        # 2. Substitute common leetspeak symbols
        for char, sub in LEET_SUBS.items():
            normalized = normalized.replace(char, sub)

        # 3. Collapse consecutive repeated characters (e.g. fuuuuck -> fuck, चूतियाaaa -> चूतिया)
        normalized = re.sub(r'(.)\1{2,}', r'\1\1', normalized)
        return normalized

    def evaluate_text(self, text: str) -> Dict[str, Any]:
        """
        Evaluates input text for profanities, abuses, and harassment.
        Returns:
            is_profane: bool
            severity: "HIGH" | "MEDIUM" | "NONE"
            detected_words: List[str]
            cleaned_text: str
            reason: Optional[str]
        """
        raw_text = (text or "").strip()
        if not raw_text:
            return {
                "is_profane": False,
                "severity": "NONE",
                "detected_words": [],
                "cleaned_text": raw_text,
                "reason": None
            }

        normalized = self.normalize_text(raw_text)
        detected: List[str] = []

        # Tokenize by word boundaries and clean tokens
        tokens = re.findall(r'[\w\u0900-\u097F]+', normalized)
        for token in tokens:
            if token in self.profanities:
                detected.append(token)

        # Also test for substring matches for compound/slur terms
        for prof in self.profanities:
            # Check only for terms of length >= 4 to avoid false positive short collisions
            if len(prof) >= 4:
                # Word boundary check for regex
                pattern = r'\b' + re.escape(prof) + r'\b'
                if re.search(pattern, normalized, re.IGNORECASE):
                    if prof not in detected:
                        detected.append(prof)

        if detected:
            unique_detected = list(set(detected))
            severity = "HIGH" if any(len(w) > 4 for w in unique_detected) else "MEDIUM"
            cleaned = self.mask_profanity(raw_text, unique_detected)
            
            logger.warning(f"[ProfanityAgent] Profanity detected: {unique_detected} (Severity: {severity})")
            return {
                "is_profane": True,
                "severity": severity,
                "detected_words": unique_detected,
                "cleaned_text": cleaned,
                "reason": f"अयोग्य किंवा आक्षेपार्ह भाषा आढळली (Offensive or abusive language detected: {', '.join(unique_detected)})"
            }

        return {
            "is_profane": False,
            "severity": "NONE",
            "detected_words": [],
            "cleaned_text": raw_text,
            "reason": None
        }

    def mask_profanity(self, text: str, detected_words: List[str]) -> str:
        """Replaces detected profanities in text with asterisks (e.g. f***k)."""
        masked = text
        for word in detected_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            masked = pattern.sub("*" * len(word), masked)
        return masked


profanity_agent = ProfanityAgent()
