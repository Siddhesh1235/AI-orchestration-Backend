"""
Local Speech-to-Text (STT) Service for WardMitra AI / PCMC Sarathi.
Powered by faster-whisper (CTranslate2-optimized OpenAI Whisper).
100% Free, Local & Self-Hosted. Zero External or Paid APIs.

Supports:
- Marathi (मराठी)
- Hindi (हिंदी)
- English
- Mixed code-switching (Hinglish / Marathish)
- Audio formats: WebM, WAV, MP3, M4A, OGG, Opus
"""

import os
import io
import time
import logging
import tempfile
import threading
from typing import Dict, Any, Optional
import torch

from app.config.settings import settings
from app.services.voice.audio_service import audio_service

logger = logging.getLogger("pcms.stt_service")


class WhisperModelManager:
    """
    Thread-safe Singleton Manager for the local faster-whisper model.
    Loads model weights once in memory and reuses them across requests.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.model = None
        self.active_model_name = None
        self.active_device = None
        self.active_compute_type = None

    @classmethod
    def get_instance(cls) -> "WhisperModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_model(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        download_root: Optional[str] = None
    ):
        """Loads faster-whisper model with hardware auto-detection and fallback."""
        with self._lock:
            target_model = model_name or getattr(settings, "WHISPER_MODEL", "small")
            target_device = (device or getattr(settings, "WHISPER_DEVICE", "cpu")).lower()
            target_compute = (compute_type or getattr(settings, "WHISPER_COMPUTE_TYPE", "int8")).lower()
            target_download_root = download_root or getattr(settings, "WHISPER_DOWNLOAD_ROOT", None)

            # Auto-detect CUDA availability if GPU is configured or requested
            if target_device == "cuda":
                if not torch.cuda.is_available():
                    logger.warning("[WhisperModelManager] CUDA requested but not available. Falling back to CPU.")
                    target_device = "cpu"
                    target_compute = "int8"
                else:
                    if target_compute == "int8":
                        target_compute = "float16"

            if (self.model is not None and
                self.active_model_name == target_model and
                self.active_device == target_device and
                self.active_compute_type == target_compute):
                return self.model

            try:
                from faster_whisper import WhisperModel
                logger.info(
                    f"[WhisperModelManager] Loading faster-whisper model '{target_model}' "
                    f"on {target_device.upper()} (compute_type={target_compute})..."
                )
                start_t = time.time()
                self.model = WhisperModel(
                    target_model,
                    device=target_device,
                    compute_type=target_compute,
                    download_root=target_download_root
                )
                load_time = time.time() - start_t
                self.active_model_name = target_model
                self.active_device = target_device
                self.active_compute_type = target_compute
                logger.info(f"[WhisperModelManager] Model '{target_model}' loaded successfully in {load_time:.2f}s.")
                return self.model
            except Exception as e:
                logger.error(f"[WhisperModelManager] Failed to load Whisper model '{target_model}': {e}")
                # Fallback to tiny on CPU if small fails to load
                if target_model != "tiny" or target_device != "cpu":
                    try:
                        logger.warning("[WhisperModelManager] Attempting fallback to 'tiny' model on CPU...")
                        from faster_whisper import WhisperModel
                        self.model = WhisperModel("tiny", device="cpu", compute_type="int8")
                        self.active_model_name = "tiny"
                        self.active_device = "cpu"
                        self.active_compute_type = "int8"
                        logger.info("[WhisperModelManager] Fallback 'tiny' model loaded successfully.")
                        return self.model
                    except Exception as fb_err:
                        logger.error(f"[WhisperModelManager] Fallback model load failed: {fb_err}")
                raise e


class LocalSTTService:
    """
    Local Speech-to-Text inference service for PCMC Sarathi / WardMitra.
    Transcribes audio into Marathi, Hindi, English, and code-mixed speech.
    """

    def __init__(self):
        self.manager = WhisperModelManager.get_instance()
        self.beam_size = getattr(settings, "WHISPER_BEAM_SIZE", 5)

    def _normalize_lang_code(self, lang: Optional[str]) -> Optional[str]:
        """Normalizes language code for Whisper ('mr', 'hi', 'en', or None for auto-detect)."""
        if not lang or lang in ("auto", "mixed", "unclear"):
            return None
        lang_clean = lang.strip().lower().split("-")[0].split("_")[0]
        if lang_clean in ("marathi", "mr"):
            return "mr"
        elif lang_clean in ("hindi", "hi"):
            return "hi"
        elif lang_clean in ("english", "en"):
            return "en"
        return lang_clean

    def transcribe_audio_sync(
        self,
        audio_bytes: bytes,
        language: Optional[str] = "mr",
        audio_format: str = "webm"
    ) -> Dict[str, Any]:
        """
        Synchronous audio transcription via local faster-whisper.
        Converts audio to 16kHz mono WAV, infers transcript, cleans up temp files.
        """
        is_valid, err_msg = audio_service.validate_audio_bytes(audio_bytes, min_bytes=100)
        lang_hint = self._normalize_lang_code(language)

        if not is_valid:
            return {
                "success": False,
                "text": "",
                "language": lang_hint or "mr",
                "duration_seconds": 0.0,
                "error": err_msg or "Empty or invalid audio data"
            }

        # 1. Normalize audio to 16kHz mono PCM WAV
        wav_bytes = audio_service.convert_to_wav_16k_mono(audio_bytes, input_format=audio_format)
        duration_sec = audio_service.get_audio_duration(wav_bytes)

        # 2. Write to temporary WAV file for faster-whisper CTranslate2 reader
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp_wav.name
        try:
            tmp_wav.write(wav_bytes)
            tmp_wav.close()

            # 3. Ensure model is loaded
            model = self.manager.load_model()

            logger.info(f"[LocalSTT] Transcribing audio ({duration_sec:.1f}s, hint={lang_hint})...")
            start_t = time.time()

            segments, info = model.transcribe(
                tmp_path,
                language=lang_hint,
                beam_size=self.beam_size,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Collect transcribed text
            transcript_parts = [segment.text.strip() for segment in segments]
            full_text = " ".join(transcript_parts).strip()
            detected_lang = info.language if hasattr(info, "language") else (lang_hint or "mr")
            proc_time = time.time() - start_t

            logger.info(f"[LocalSTT] Transcription complete in {proc_time:.2f}s (lang={detected_lang}): '{full_text}'")

            if not full_text:
                return {
                    "success": False,
                    "text": "",
                    "language": detected_lang,
                    "duration_seconds": round(duration_sec, 2),
                    "error": "No audible speech detected."
                }

            return {
                "success": True,
                "text": full_text,
                "language": detected_lang,
                "duration_seconds": round(duration_sec, 2),
                "engine": f"faster-whisper ({self.manager.active_model_name})"
            }
        except Exception as e:
            logger.error(f"[LocalSTT] Transcription error: {e}")
            return {
                "success": False,
                "text": "",
                "language": lang_hint or "mr",
                "duration_seconds": round(duration_sec, 2),
                "error": f"Speech-to-text processing failed: {str(e)}"
            }
        finally:
            audio_service.cleanup_temp_files(tmp_path)

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        language: Optional[str] = "mr",
        audio_format: str = "webm"
    ) -> Dict[str, Any]:
        """Async wrapper running CPU/GPU transcription in thread pool to avoid blocking FastAPI."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.transcribe_audio_sync,
            audio_bytes,
            language,
            audio_format
        )

    async def transcribe_base64(
        self,
        audio_base64: str,
        language: Optional[str] = "mr",
        audio_format: str = "webm"
    ) -> Dict[str, Any]:
        """Decodes base64 string and transcribes spoken audio."""
        import base64
        if not audio_base64:
            return {
                "success": False,
                "text": "",
                "language": self._normalize_lang_code(language) or "mr",
                "duration_seconds": 0.0,
                "error": "Empty audio_base64 string."
            }

        # Strip data URL prefix if present
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",", 1)[1]

        try:
            raw_bytes = base64.b64decode(audio_base64)
            return await self.transcribe_audio(raw_bytes, language=language, audio_format=audio_format)
        except Exception as e:
            logger.error(f"[LocalSTT] Base64 decoding error: {e}")
            return {
                "success": False,
                "text": "",
                "language": self._normalize_lang_code(language) or "mr",
                "duration_seconds": 0.0,
                "error": f"Invalid base64 audio data: {e}"
            }


local_stt_service = LocalSTTService()
