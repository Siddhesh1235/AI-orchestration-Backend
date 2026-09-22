"""
Unified Voice Adapter Service for WardMitra AI / PCMC Sarathi.
Adheres strictly to the architectural golden rule:
WARDMITRA AI DOES NOT CARE WHETHER THE USER TYPED OR SPOKE.

Voice is purely an I/O adapter:
Audio Input -> Local STT -> Text -> Existing Conversation Engine -> Text -> Local TTS -> Audio Output.
"""

import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.services.voice.stt_service import local_stt_service
from app.services.voice.tts_service import local_tts_service
from app.services.conversational_service import conversational_service

logger = logging.getLogger("pcms.voice_service")


class VoiceService:
    """Thin Voice Adapter around WardMitra's conversational intelligence engine."""

    def __init__(self):
        self.stt = local_stt_service
        self.tts = local_tts_service
        self.conversation = conversational_service

    async def process_voice_turn(
        self,
        db: Session,
        audio_bytes: Optional[bytes] = None,
        audio_base64: Optional[str] = None,
        audio_format: str = "webm",
        language: str = "mr",
        session_id: Optional[str] = None,
        citizen_phone: str = "9876543210",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        photo_filename: Optional[str] = None,
        photo_bytes: Optional[bytes] = None,
        action: Optional[str] = None,
        confirm_register: bool = False,
        voice_reply: bool = True
    ) -> Dict[str, Any]:
        """
        Executes a complete voice conversation turn:
        1. Local STT: Converts spoken voice into text.
        2. Core Pipeline: Sends transcribed text into existing conversational_service.handle_chat.
        3. Local TTS: Synthesizes bot reply into spoken audio.
        4. Returns unified response with transcript, text reply, and audio base64.
        """
        transcript = ""
        stt_info = None

        # 1. Speech-to-Text Transcription if audio is present
        if audio_bytes:
            stt_res = await self.stt.transcribe_audio(audio_bytes, language=language, audio_format=audio_format)
            if stt_res.get("success") and stt_res.get("text"):
                transcript = stt_res["text"]
            stt_info = stt_res
        elif audio_base64:
            stt_res = await self.stt.transcribe_base64(audio_base64, language=language, audio_format=audio_format)
            if stt_res.get("success") and stt_res.get("text"):
                transcript = stt_res["text"]
            stt_info = stt_res

        logger.info(f"[VoiceService] Processing turn: session='{session_id}', transcript='{transcript}', action='{action}'")

        # 2. Existing WardMitra Conversational Engine (Zero duplicated business logic)
        chat_result = self.conversation.handle_chat(
            db=db,
            message=transcript if transcript else None,
            category=None,
            latitude=latitude,
            longitude=longitude,
            citizen_phone=citizen_phone,
            photo_filename=photo_filename,
            photo_bytes=photo_bytes,
            video_result=None,
            confirm_register=confirm_register,
            action=action,
            session_id=session_id
        )

        chat_result["transcript"] = transcript
        chat_result["stt_info"] = stt_info

        # 3. Text-to-Speech Synthesis for Bot Reply
        if voice_reply and chat_result.get("reply"):
            reply_lang = chat_result.get("language") or language or "mr"
            try:
                tts_res = await self.tts.synthesize_speech(
                    text=chat_result["reply"],
                    language=reply_lang
                )
                chat_result["audio_base64"] = tts_res.get("audio_base64")
                chat_result["audio_format"] = tts_res.get("audio_format", "wav")
                chat_result["audio_available"] = tts_res.get("audio_available", False)
                chat_result["voice_reply"] = bool(tts_res.get("audio_base64"))
            except Exception as tts_err:
                logger.warning(f"[VoiceService] TTS synthesis warning: {tts_err}")
                chat_result["audio_base64"] = None
                chat_result["audio_format"] = None
                chat_result["audio_available"] = False
                chat_result["voice_reply"] = False
        else:
            chat_result["audio_base64"] = None
            chat_result["audio_format"] = None
            chat_result["audio_available"] = False
            chat_result["voice_reply"] = False

        return chat_result


voice_service = VoiceService()
