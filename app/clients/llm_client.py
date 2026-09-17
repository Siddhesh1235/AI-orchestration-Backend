"""
Smart Hybrid LLM Client for PCMC Sarathi AI.
Follows strict 3-tier cost-optimization strategy:
  Tier 1: Fast Rule / Keyword matching (0 cost, instant)
  Tier 2: Free Local Ollama (Primary LLM, 0 cost)
  Tier 3: OpenAI API Fallback (Emergency only, strictly cost-optimized)
"""

import logging
import re
from typing import Optional, Dict, Any
import httpx

from app.config.settings import settings

logger = logging.getLogger("pcms.llm_client")
logging.basicConfig(level=logging.INFO)


class SmartLLMClient:
    def __init__(self):
        self.ollama_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.ollama_model = settings.OLLAMA_MODEL
        self.openai_key = settings.OPENAI_API_KEY
        self.openai_model = settings.OPENAI_MODEL

    def _call_ollama(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Calls local Ollama instance via HTTP."""
        try:
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 128
                }
            }
            with httpx.Client(timeout=settings.OLLAMA_TIMEOUT_SECONDS) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    response_text = data.get("response", "").strip()
                    logger.info(f"[LLM] Ollama responded successfully with {len(response_text)} chars.")
                    return response_text
        except Exception as e:
            logger.warning(f"[LLM] Ollama is unreachable or timed out ({e}). Testing fallback.")
        return None

    def _call_openai(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Calls OpenAI API only as secondary fallback if key is configured."""
        if not self.openai_key:
            logger.info("[LLM] OpenAI API Key is not configured. Skipping Tier 3.")
            return None

        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.openai_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 128
            }
            with httpx.Client(timeout=settings.OPENAI_TIMEOUT_SECONDS) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    logger.info("[LLM] OpenAI Fallback responded successfully.")
                    return text
                else:
                    logger.warning(f"[LLM] OpenAI returned HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"[LLM] OpenAI fallback call failed: {e}")
        return None

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Executes query following Ollama-first, OpenAI-fallback architecture.
        """
        # Tier 2: Try Ollama
        ollama_response = self._call_ollama(prompt, system_prompt)
        if ollama_response:
            return ollama_response

        # Tier 3: Try OpenAI
        openai_response = self._call_openai(prompt, system_prompt)
        if openai_response:
            return openai_response

        # Fallback message
        return ""


llm_client = SmartLLMClient()
