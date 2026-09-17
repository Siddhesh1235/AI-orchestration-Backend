"""
Multilingual NLP Agent for PCMC Sarathi AI.
Extracts intent and detects complaint category from free-text messages in Marathi, Hindi, and English.
Integrates keyword heuristics with SmartLLMClient (Ollama-first with OpenAI fallback).
"""

import re
import yaml
import logging
from typing import Dict, Any, Optional

from app.config.settings import settings
from app.clients.llm_client import llm_client

logger = logging.getLogger("pcms.nlp_agent")


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

    def extract_intent_and_category(self, text: str) -> Dict[str, Any]:
        """
        Parses text to determine:
        1. Intent (REGISTER_COMPLAINT, TRACK_COMPLAINT, FEEDBACK, SERVICE_INQUIRY)
        2. Civic Category (one of the 11 target classes)
        """
        cleaned_text = (text or "").strip().lower()

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
                "source": "keyword_heuristic"
            }

        # Step 3: LLM Intent & Category Extraction (Tier 2 Ollama -> Tier 3 OpenAI)
        llm_category = self._llm_classify(cleaned_text)
        if llm_category:
            logger.info(f"[NLPAgent] LLM matched category: {llm_category}")
            return {
                "intent": intent,
                "category": llm_category,
                "confidence": 0.82,
                "source": "smart_llm"
            }

        # Step 4: Default Fallback
        return {
            "intent": intent,
            "category": "garbage",  # most common municipal complaint default
            "confidence": 0.60,
            "source": "default_fallback"
        }

    def _detect_intent(self, text: str) -> str:
        if any(w in text for w in ["track", "status", "तपासा", "स्थिती", "ticket", "तिकीट"]):
            return "TRACK_COMPLAINT"
        if any(w in text for w in ["feedback", "rating", "अभिप्राय", "स्टार"]):
            return "FEEDBACK_SUBMISSION"
        if any(w in text for w in ["info", "office", "contact", "कार्यालय", "माहिती", "वेळ"]):
            return "SERVICE_INQUIRY"
        return "REGISTER_COMPLAINT"

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
            f"{', '.join(valid_classes)}. Do not write any other explanation or punctuation."
        )
        prompt = f"Complaint: '{text}'\nMatching category name:"
        try:
            result = llm_client.generate(prompt=prompt, system_prompt=system_prompt)
            result_clean = result.strip().lower()
            for cls in valid_classes:
                if cls in result_clean:
                    return cls
        except Exception as e:
            logger.warning(f"[NLPAgent] LLM classification error: {e}")
        return None


nlp_agent = NLPAgent()
