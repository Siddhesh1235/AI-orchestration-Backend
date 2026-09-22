"""
Local Text-to-Speech (TTS) Service for WardMitra AI / PCMC Sarathi.
Powered by Meta MMS-TTS (Massively Multilingual Speech VITS models).
100% Free, Local & Self-Hosted. Zero External or Paid APIs.

Supports:
- Marathi (facebook/mms-tts-mar)
- Hindi (facebook/mms-tts-hin)
- English (facebook/mms-tts-eng)
- Concise, clear spoken phrasing tailored for low-literacy citizens
- Exact ticket ID preservation (spelling out alphanumeric letters clearly)
- Zero-crash fallback with state and text preservation
"""

import os
import io
import re
import time
import base64
import struct
import logging
import threading
from typing import Dict, Any, Optional, Tuple
import torch
import soundfile as sf
import numpy as np

from app.config.settings import settings

logger = logging.getLogger("pcms.tts_service")

# Model identifiers on Hugging Face
MMS_MODEL_MAP = {
    "mr": "facebook/mms-tts-mar",
    "hi": "facebook/mms-tts-hin",
    "en": "facebook/mms-tts-eng"
}


def generate_fallback_wav_base64(duration_sec: float = 0.5, sample_rate: int = 16000) -> str:
    """Generates a valid, minimal PCM WAV audio file encoded in base64."""
    num_samples = int(sample_rate * duration_sec)
    data_size = num_samples * 2  # 16-bit mono
    header = bytearray()
    header.extend(b'RIFF')
    header.extend(struct.pack('<I', 36 + data_size))
    header.extend(b'WAVEfmt ')
    header.extend(struct.pack('<I', 16))
    header.extend(struct.pack('<H', 1))
    header.extend(struct.pack('<H', 1))
    header.extend(struct.pack('<I', sample_rate))
    header.extend(struct.pack('<I', sample_rate * 2))
    header.extend(struct.pack('<H', 2))
    header.extend(struct.pack('<H', 16))
    header.extend(b'data')
    header.extend(struct.pack('<I', data_size))
    silence = b'\x00' * data_size
    raw_wav = bytes(header + silence)
    return base64.b64encode(raw_wav).decode('utf-8')


def simplify_text_for_speech(text: str, lang: str = "mr") -> str:
    """
    Cleans and summarizes text for natural, conversational speech output.
    Tailored for low-literacy citizens:
    - Strips markdown formatting, links, bullet points.
    - Formats alphanumeric ticket IDs for distinct spoken pronunciation.
    - Limits excessively long responses to 2-3 key sentences.
    """
    if not text:
        return ""

    t = text.strip()

    # Remove URLs
    t = re.sub(r'https?://\S+', '', t)
    # Remove markdown bold/italic asterisks, hash headers, and emojis
    t = re.sub(r'[*#_`~]', '', t)

    # Format ticket numbers for clear acoustic delivery (e.g. WM-20260922-1234 -> WM 2026 0922 1234)
    def format_ticket(match):
        t_str = match.group(0)
        return " ".join(re.findall(r'[A-Za-z]+|\d{1,4}', t_str))

    t = re.sub(r'\b[A-Za-z]{2,3}[-\s]\d{6,8}[-\s]\d{3,4}\b', format_ticket, t)
    t = re.sub(r'\b[A-Z]{3}\d{3}\b', lambda m: " ".join(list(m.group(0))), t)

    # Remove remaining bullets, hyphens, and dashes
    t = re.sub(r'[•\-\–]', ' ', t)

    # Collapse multiple whitespaces and newlines
    t = re.sub(r'\s+', ' ', t).strip()

    # If response is very long (e.g. detailed RAG doc or handbook), extract first 2-3 key sentences
    sentences = re.split(r'(?<=[.?!।])\s+', t)
    if len(sentences) > 3 and len(t) > 250:
        t = " ".join(sentences[:3]).strip()

    return t


class TTSModelManager:
    """
    Thread-safe Singleton Model Manager for local Meta MMS-TTS models.
    Caches loaded VITS models per language (mr, hi, en) in memory.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self.device = (getattr(settings, "TTS_DEVICE", "cpu") or "cpu").lower()
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

    @classmethod
    def get_instance(cls) -> "TTSModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_model_and_tokenizer(self, lang: str = "mr") -> Tuple[Optional[Any], Optional[Any]]:
        """Loads and returns the VitsModel and AutoTokenizer for target language."""
        lang_key = lang if lang in MMS_MODEL_MAP else "mr"
        model_id = MMS_MODEL_MAP.get(lang_key, "facebook/mms-tts-mar")

        with self._lock:
            if lang_key in self._models and lang_key in self._tokenizers:
                return self._models[lang_key], self._tokenizers[lang_key]

            try:
                from transformers import VitsModel, AutoTokenizer
                logger.info(f"[TTSModelManager] Loading Meta MMS-TTS model '{model_id}' on {self.device}...")
                start_t = time.time()

                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = VitsModel.from_pretrained(model_id)
                model.to(self.device)
                model.eval()

                self._models[lang_key] = model
                self._tokenizers[lang_key] = tokenizer
                logger.info(f"[TTSModelManager] Model '{model_id}' loaded in {time.time() - start_t:.2f}s.")
                return model, tokenizer
            except Exception as e:
                logger.warning(f"[TTSModelManager] Could not load MMS-TTS model '{model_id}': {e}")
                return None, None


class LocalTTSService:
    """
    Local Text-to-Speech synthesis service for PCMC Sarathi / WardMitra.
    Synthesizes conversational text into 16kHz mono WAV audio base64.
    """

    def __init__(self):
        self.manager = TTSModelManager.get_instance()
        self.sampling_rate = getattr(settings, "TTS_SAMPLING_RATE", 16000)

    def _normalize_lang(self, lang: Optional[str]) -> str:
        """Normalizes language code for TTS."""
        if not lang:
            return "mr"
        code = lang.strip().lower().split("-")[0].split("_")[0]
        if code in ("marathi", "mr"):
            return "mr"
        elif code in ("hindi", "hi"):
            return "hi"
        elif code in ("english", "en"):
            return "en"
        return "mr"

    def synthesize_speech_sync(
        self,
        text: str,
        language: str = "mr",
        gender: Optional[str] = "female"
    ) -> Dict[str, Any]:
        """
        Synthesizes text into spoken WAV audio encoded in base64.
        Ensures zero-crash resilience: if TTS generation fails, returns
        `audio_available: false` while keeping original text intact.
        """
        lang = self._normalize_lang(language)
        if not text or not text.strip():
            return {
                "success": False,
                "audio_base64": "",
                "audio_format": "wav",
                "language": lang,
                "text": "",
                "audio_available": False,
                "error": "Empty or whitespace text."
            }

        spoken_text = simplify_text_for_speech(text, lang=lang)
        if not spoken_text:
            spoken_text = text.strip()

        # 1. Try local Meta MMS-TTS inference
        model, tokenizer = self.manager.get_model_and_tokenizer(lang)
        if model is not None and tokenizer is not None:
            try:
                start_t = time.time()
                inputs = tokenizer(spoken_text, return_tensors="pt")
                inputs = {k: v.to(self.manager.device) for k, v in inputs.items()}

                with torch.no_grad():
                    output = model(**inputs).waveform

                audio_data = output.squeeze().cpu().numpy()
                model_sr = getattr(model.config, "sampling_rate", self.sampling_rate)

                # Write audio to in-memory WAV buffer
                wav_io = io.BytesIO()
                sf.write(wav_io, audio_data, model_sr, format='WAV', subtype='PCM_16')
                wav_bytes = wav_io.getvalue()
                audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
                proc_time = time.time() - start_t

                logger.info(
                    f"[LocalTTS] Synthesized {len(spoken_text)} chars into {len(wav_bytes)}B WAV "
                    f"in {proc_time:.2f}s (lang={lang})."
                )
                return {
                    "success": True,
                    "audio_base64": audio_b64,
                    "audio_format": "wav",
                    "language": lang,
                    "text": spoken_text,
                    "audio_available": True,
                    "engine": f"Meta MMS-TTS ({MMS_MODEL_MAP.get(lang)})"
                }
            except Exception as inf_err:
                logger.error(f"[LocalTTS] Synthesis inference error: {inf_err}")

        # 2. Resilient Zero-Crash Fallback:
        # Returns mock PCM WAV so browser playback doesn't error, with audio_available flag
        logger.info(f"[LocalTTS] Providing fallback audio response for lang='{lang}'.")
        return {
            "success": True,
            "audio_base64": generate_fallback_wav_base64(duration_sec=0.5),
            "audio_format": "wav",
            "language": lang,
            "text": spoken_text,
            "audio_available": False,
            "engine": "Local TTS Fallback",
            "message": "Local TTS model offline or downloading. Returning text response."
        }

    async def synthesize_speech(
        self,
        text: str,
        language: str = "mr",
        gender: Optional[str] = "female"
    ) -> Dict[str, Any]:
        """Async wrapper running CPU/GPU synthesis in thread executor."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.synthesize_speech_sync,
            text,
            language,
            gender
        )


local_tts_service = LocalTTSService()
