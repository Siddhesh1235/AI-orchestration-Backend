"""
Fraud & Fake Complaint Detection Agent for PCMC Sarathi AI.
Identifies spam, gibberish, dummy test inputs, and out-of-bounds coordinates
to prevent municipal resources from being dispatched to fraudulent grievances.
"""

import re
import logging
from typing import Dict, Any, Optional

from app.config.settings import settings

logger = logging.getLogger("pcms.fraud_agent")

SPAM_PATTERNS = [
    r"^(.)\1{4,}$",                     # Repetitive characters like "aaaaaa", "......"
    r"^(asdf|qwerty|zxcv|1234|test|dummy|fake)",  # Common keyboard mashing
    r"^(test|testing|sample|demo|spam|blabla|xyz)\b"
]

SPAM_EXACT_WORDS = {
    "test", "testing", "fake", "dummy", "sample", "demo", "spam",
    "asdf", "qwerty", "xyz", "abcd", "trial"
}


class FraudAgent:
    def evaluate_authenticity(
        self,
        description: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates whether a complaint is authentic or fraudulent/spam.
        Returns:
            is_fraud: bool
            reason: str
            category: str ("AUTHENTIC", "GIBBERISH", "SPAM_TEXT", "OUT_OF_BOUNDS")
        """
        text = (description or "").strip().lower()

        # 1. Check for empty or excessively short text
        if len(text) < 4:
            logger.warning("[FraudAgent] Description too short or empty.")
            return {
                "is_fraud": True,
                "reason": "तक्रारीचे वर्णन अपूर्ण किंवा अतिशय संक्षिप्त आहे (Description too short)",
                "category": "GIBBERISH"
            }

        # 2. Check for exact spam words
        words = text.split()
        if len(words) == 1 and words[0] in SPAM_EXACT_WORDS:
            logger.warning(f"[FraudAgent] Exact spam word detected: '{words[0]}'")
            return {
                "is_fraud": True,
                "reason": f"चाचणी किंवा बनावट मजकूर आढळला: '{words[0]}' (Spam/Test input detected)",
                "category": "SPAM_TEXT"
            }

        # 3. Check regex spam patterns
        for pattern in SPAM_PATTERNS:
            if re.search(pattern, text):
                logger.warning(f"[FraudAgent] Pattern match fraud: '{pattern}' on '{text}'")
                return {
                    "is_fraud": True,
                    "reason": "निरर्थक किंवा स्पॅम मजकूर आढळला (Gibberish or repetitive text detected)",
                    "category": "GIBBERISH"
                }

        # 4. Check Out-of-Bounds GPS Coordinates
        if latitude is not None and longitude is not None:
            if not (latitude == 0.0 and longitude == 0.0):
                lat_min = settings.PCMC_LAT_MIN - 0.03
                lat_max = settings.PCMC_LAT_MAX + 0.03
                lng_min = settings.PCMC_LNG_MIN - 0.03
                lng_max = settings.PCMC_LNG_MAX + 0.03

                if not (lat_min <= latitude <= lat_max and lng_min <= longitude <= lng_max):
                    logger.warning(f"[FraudAgent] Out of bounds GPS: lat={latitude}, lng={longitude}")
                    return {
                        "is_fraud": True,
                        "reason": f"स्थान पिंपरी चिंचवड (PCMC) हद्दीबाहेर आहे ({latitude:.4f}, {longitude:.4f})",
                        "category": "OUT_OF_BOUNDS"
                    }

        return {
            "is_fraud": False,
            "reason": "तक्रार वैध व अस्सल आढळली (Authentic grievance)",
            "category": "AUTHENTIC"
        }


fraud_agent = FraudAgent()
