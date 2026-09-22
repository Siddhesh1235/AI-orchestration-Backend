"""
Voice, STT & TTS API Routes for PCMC Sarathi AI / WardMitra.
100% Local, Self-Hosted Voice Interaction Layer:
1. Speech-to-Text via local faster-whisper (Marathi, Hindi, English, Code-mixed)
2. Text-to-Speech via local Meta MMS-TTS (VITS)
3. Full voice conversational turn adapter
"""

import io
from typing import Optional
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Depends, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.services.voice.stt_service import local_stt_service
from app.services.voice.tts_service import local_tts_service
from app.services.voice.voice_service import voice_service

router = APIRouter(prefix="/voice", tags=["Voice Interaction Layer (Local STT & TTS)"])


class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text content to synthesize into spoken audio")
    language: str = Field("mr", description="Target Indic language: mr (Marathi), hi (Hindi), en (English)")
    gender: Optional[str] = Field("female", description="Voice gender: female or male")


class TTSRequest(SynthesizeRequest):
    """Alias for backwards compatibility with previous TTS clients."""
    pass


@router.post("/transcribe")
@router.post("/stt")
async def transcribe_speech(
    audio: Optional[UploadFile] = File(None, description="Audio file (WebM / WAV / MP3 / OGG / Opus)"),
    audio_base64: Optional[str] = Form(None, description="Base64 encoded audio string"),
    language: Optional[str] = Form("mr", description="Source language hint: mr, hi, en, or auto"),
    audio_format: str = Form("webm", description="Audio format: webm, wav, mp3, ogg, opus")
):
    """
    Transcribes spoken voice into text using local faster-whisper model.
    Accepts multipart audio file or base64 audio string.
    Supports Marathi (mr), Hindi (hi), English (en), and code-mixed speech.
    """
    if not audio and not audio_base64:
        raise HTTPException(status_code=400, detail="Either 'audio' file or 'audio_base64' string must be provided.")

    if audio and audio.filename:
        audio_bytes = await audio.read()
        ext = audio.filename.split(".")[-1].lower() if "." in audio.filename else audio_format
        result = await local_stt_service.transcribe_audio(
            audio_bytes=audio_bytes,
            language=language,
            audio_format=ext
        )
    else:
        result = await local_stt_service.transcribe_base64(
            audio_base64=audio_base64,
            language=language,
            audio_format=audio_format
        )

    # Add backwards-compatibility alias 'transcript' matching 'text'
    result["transcript"] = result.get("text", "")
    return result


@router.post("/synthesize")
@router.post("/tts")
async def synthesize_speech(
    payload: SynthesizeRequest
):
    """
    Synthesizes conversational text into spoken audio (16kHz mono WAV, base64) using local Meta MMS-TTS.
    Request: { "text": str, "language": "mr" | "hi" | "en", "gender": "female" | "male" }
    Response: { "success": bool, "audio_base64": str, "audio_format": "wav", "language": str, "text": str, "audio_available": bool }
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="The 'text' field must not be empty.")

    result = await local_tts_service.synthesize_speech(
        text=payload.text,
        language=payload.language,
        gender=payload.gender
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "TTS synthesis failed"))

    return result


@router.post("/chat")
async def voice_chat_turn(
    audio: Optional[UploadFile] = File(None, description="Citizen spoken audio file"),
    audio_base64: Optional[str] = Form(None, description="Base64 encoded audio string"),
    language: str = Form("mr", description="Spoken language: mr, hi, en"),
    audio_format: str = Form("webm", description="Audio format: webm, wav, mp3, ogg, opus"),
    session_id: Optional[str] = Form(None, description="Conversation session ID"),
    citizen_phone: str = Form("9876543210", description="Citizen Phone"),
    latitude: Optional[float] = Form(None, description="GPS Latitude"),
    longitude: Optional[float] = Form(None, description="GPS Longitude"),
    photo: Optional[UploadFile] = File(None, description="Evidence photo"),
    action: Optional[str] = Form(None, description="User action button"),
    confirm_register: bool = Form(False, description="Confirm grievance registration"),
    voice_reply: bool = Form(True, description="Whether to include spoken audio version of bot reply"),
    db: Session = Depends(get_db)
):
    """
    Complete end-to-end voice conversation turn:
    Voice In -> Local STT -> Existing WardMitra Engine -> Text -> Local TTS -> Spoken Voice Out.
    """
    audio_bytes = None
    if audio and audio.filename:
        audio_bytes = await audio.read()
        if "." in audio.filename:
            audio_format = audio.filename.split(".")[-1].lower()

    photo_filename = None
    photo_bytes = None
    if photo and photo.filename:
        photo_filename = photo.filename
        photo_bytes = await photo.read()

    result = await voice_service.process_voice_turn(
        db=db,
        audio_bytes=audio_bytes,
        audio_base64=audio_base64,
        audio_format=audio_format,
        language=language,
        session_id=session_id,
        citizen_phone=citizen_phone,
        latitude=latitude,
        longitude=longitude,
        photo_filename=photo_filename,
        photo_bytes=photo_bytes,
        action=action,
        confirm_register=confirm_register,
        voice_reply=voice_reply
    )

    return result
