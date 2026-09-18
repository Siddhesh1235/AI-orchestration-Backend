"""
Fraud & Fake Complaint Detection Agent for WardMitra AI / PCMC Sarathi.
Identifies spam, gibberish, dummy test inputs, out-of-bounds coordinates,
and multi-signal inconsistencies. Computes normalized fraud_score (0.0 to 1.0)
and sets status to REVIEW_REQUIRED.
"""

import re
import logging
from typing import Dict, Any, Optional, List

from app.config.settings import settings
from app.rules.fraud_rules import compute_fraud_risk_score

logger = logging.getLogger("pcms.fraud_agent")

SPAM_PATTERNS = [
    r"^(.)\1{4,}$",                               # Repetitive characters like "aaaaaa", "......"
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
        longitude: Optional[float] = None,
        text_image_inconsistent: bool = False,
        is_irrelevant_evidence: bool = False,
        is_duplicate: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates complaint authenticity across multiple signals.
        Returns:
            is_fraud: bool
            fraud_score: float (0.0 to 1.0)
            status: "APPROVED" | "REVIEW_REQUIRED"
            reason: str
            category: str ("AUTHENTIC", "GIBBERISH", "SPAM_TEXT", "OUT_OF_BOUNDS", "INCONSISTENT")
        """
        text = (description or "").strip().lower()
        is_gibberish = False
        is_spam_word = False
        is_out_of_bounds = False
        category = "AUTHENTIC"
        primary_reason = None

        # 1. Check for empty or excessively short text
        if len(text) < 4:
            logger.warning("[FraudAgent] Description too short or empty.")
            is_gibberish = True
            category = "GIBBERISH"
            primary_reason = "तक्रारीचे वर्णन अपूर्ण किंवा अतिशय संक्षिप्त आहे (Description too short)"

        # 1.5. Check for greetings mistakenly submitted as complaints
        greeting_patterns = [
            r"^(h+[i|y]+|h+e+l+o+|h+e+y+|h+l+o+|g+m+|g+n+|good\s*morning|good\s*afternoon|good\s*evening)[\s!\.]*$",
            r"^(नमस्कार|रामराम|शुभ\s*सकाळ|शुभ\s*दुपार|शुभ\s*संध्याकाळ|प्रणाम|जय\s*महाराष्ट्र)[\s!\.]*$",
            r"^(namaskar|namaste|radhe\s*radhe|kasa\s*ahes|kashi\s*ahes|how\s*are\s*you)[\s!\.]*$"
        ]
        for gp in greeting_patterns:
            if re.match(gp, text):
                logger.warning(f"[FraudAgent] Greeting submitted as complaint: '{text}'")
                is_spam_word = True
                category = "GREETING_ONLY"
                primary_reason = "केवळ अभिवादन मजकूर (Greeting text cannot be registered as a complaint ticket)"
                break

        # 2. Check for exact spam words
        words = text.split()
        if len(words) == 1 and words[0] in SPAM_EXACT_WORDS:
            logger.warning(f"[FraudAgent] Exact spam word detected: '{words[0]}'")
            is_spam_word = True
            category = "SPAM_TEXT"
            primary_reason = f"चाचणी किंवा बनावट मजकूर आढळला: '{words[0]}' (Spam/Test input detected)"

        # 3. Check regex spam patterns
        if not is_gibberish:
            for pattern in SPAM_PATTERNS:
                if re.search(pattern, text):
                    logger.warning(f"[FraudAgent] Pattern match fraud: '{pattern}' on '{text}'")
                    is_gibberish = True
                    category = "GIBBERISH"
                    primary_reason = "निरर्थक किंवा स्पॅम मजकूर आढळला (Gibberish or repetitive text detected)"
                    break

        # 4. Check Out-of-Bounds GPS Coordinates
        if latitude is not None and longitude is not None:
            if not (latitude == 0.0 and longitude == 0.0):
                lat_min = settings.PCMC_LAT_MIN - 0.03
                lat_max = settings.PCMC_LAT_MAX + 0.03
                lng_min = settings.PCMC_LNG_MIN - 0.03
                lng_max = settings.PCMC_LNG_MAX + 0.03

                if not (lat_min <= latitude <= lat_max and lng_min <= longitude <= lng_max):
                    logger.warning(f"[FraudAgent] Out of bounds GPS: lat={latitude}, lng={longitude}")
                    is_out_of_bounds = True
                    category = "OUT_OF_BOUNDS"
                    primary_reason = f"स्थान पिंपरी चिंचवड (PCMC) हद्दीबाहेर आहे ({latitude:.4f}, {longitude:.4f})"

        # Multi-signal risk score computation
        risk_result = compute_fraud_risk_score(
            is_gibberish=is_gibberish,
            is_spam_word=is_spam_word,
            is_out_of_bounds=is_out_of_bounds,
            text_image_inconsistent=text_image_inconsistent,
            is_irrelevant_evidence=is_irrelevant_evidence,
            is_duplicate=is_duplicate
        )

        final_is_fraud = is_gibberish or is_spam_word or is_out_of_bounds or risk_result["is_fraud"]
        final_reason = primary_reason or risk_result["reason"]

        return {
            "is_fraud": final_is_fraud,
            "fraud_score": risk_result["fraud_score"],
            "status": risk_result["status"],
            "reason": final_reason,
            "category": category,
            "risk_signals": risk_result["signals"]
        }


fraud_agent = FraudAgent()
