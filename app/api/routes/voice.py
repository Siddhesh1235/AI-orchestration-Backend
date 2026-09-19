"""
Voice, STT & TTS API Routes for PCMC Sarathi AI / WardMitra.
Integrates Digital India Bhashini:
1. Speech-to-Text (IndicConformer ASR) for Marathi (mr), Hindi (hi), English (en)
2. Text-to-Speech (Indic TTS) for natural voice responses
"""

from typing import Optional
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Body
from pydantic import BaseModel, Field

from app.services.bhashini_stt_service import bhashini_stt_service
from app.services.bhashini_tts_service import bhashini_tts_service

router = APIRouter(prefix="/voice", tags=["Voice, STT & TTS (Bhashini)"])


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text content to synthesize into spoken audio")
    language: str = Field("mr", description="Target Indic language: mr (Marathi), hi (Hindi), en (English)")
    gender: Optional[str] = Field("female", description="Voice gender: female or male")


@router.post("/stt")
async def bhashini_speech_to_text(
    audio: Optional[UploadFile] = File(None, description="Audio file (WebM / WAV / MP3 / OGG / Opus)"),
    audio_base64: Optional[str] = Form(None, description="Base64 encoded audio string"),
    language: str = Form("mr", description="Source Indic language code: mr (Marathi), en (English), hi (Hindi)"),
    audio_format: str = Form("webm", description="Audio format: webm, wav, mp3, ogg, opus")
):
    """
    Transcribes spoken voice into Marathi, Hindi, or English text using Digital India Bhashini ASR.
    Accepts either multipart audio file (e.g. WhatsApp voice notes OGG/Opus) or base64 audio data.
    """
    if not audio and not audio_base64:
        raise HTTPException(status_code=400, detail="Either 'audio' file or 'audio_base64' string must be provided.")

    if audio and audio.filename:
        audio_bytes = await audio.read()
        ext = audio.filename.split(".")[-1].lower() if "." in audio.filename else audio_format
        result = await bhashini_stt_service.transcribe_audio(
            audio_bytes=audio_bytes,
            language=language,
            audio_format=ext
        )
    else:
        result = await bhashini_stt_service.transcribe_base64(
            audio_base64=audio_base64,
            language=language,
            audio_format=audio_format
        )

    return result


@router.post("/tts")
async def bhashini_text_to_speech(
    payload: TTSRequest
):
    """
    Synthesizes conversational text into spoken audio (WAV, base64) using Digital India Bhashini TTS.
    Request: { "text": str, "language": "mr" | "hi" | "en", "gender": "female" | "male" }
    Response: { "success": bool, "audio_base64": str, "audio_format": "wav", "language": str, "text": str, "engine": str, "source": str }
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="The 'text' field must not be empty.")

    result = await bhashini_tts_service.synthesize_speech(
        text=payload.text,
        language=payload.language,
        gender=payload.gender
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "TTS synthesis failed"))

    return result
