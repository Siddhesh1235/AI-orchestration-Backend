"""
Conversation and state flow tests for WardMitra AI Local Voice Interaction Layer.
Verifies:
- Voice complaint initiation -> Category extraction -> Location/evidence prompting.
- Colloquial voice affirmative ("हो", "होय", "बरोबर", "haa", "ho na") handling.
- Colloquial voice negative ("नाही", "नको", "गलत आहे", "wrong") handling.
- Spoken category correction ("नाही माझी तक्रार पाणी गळती बाबत आहे").
- Voice cancellation ("तक्रार रद्द करा").
- Registration rejection ("❌ No / नाही" cancels the complaint draft).
- Voice moderation filter for offensive/abusive audio transcripts.
- Integration of voice_service.process_voice_turn preserving conversational state.
"""

import io
import uuid
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from app.main import app
from app.database.session import SessionLocal
from app.services.voice.voice_service import voice_service
from app.services.conversational_service import conversational_service
from app.agents.nlp_agent import nlp_agent


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _dummy_image_bytes():
    arr = np.random.randint(50, 200, size=(100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ==========================================
# 1. NLP Affirmative & Negative Speech Tests
# ==========================================

def test_nlp_marathi_hindi_affirmative_phrases():
    """Verify nlp_agent recognizes colloquial spoken affirmative tokens across Marathi, Hindi, and English."""
    affirmative_samples = [
        "हो",
        "होय",
        "हो हो",
        "बरोबर",
        "बरोबर आहे",
        "हं",
        "haa",
        "ha",
        "ha bhai",
        "ho na",
        "barobar",
        "yes",
        "yess",
        "correct",
        "thik ahe"
    ]
    for phrase in affirmative_samples:
        assert nlp_agent.is_affirmative(phrase), f"Failed to recognize affirmative phrase: '{phrase}'"


def test_nlp_marathi_hindi_negative_phrases():
    """Verify nlp_agent recognizes colloquial spoken negative tokens across Marathi, Hindi, and English."""
    negative_samples = [
        "नाही",
        "नाही नाही",
        "नको",
        "बरोबर नाही",
        "चूक आहे",
        "nahi",
        "nako",
        "wrong",
        "no",
        "not correct",
        "galat hai",
        "नहीं"
    ]
    for phrase in negative_samples:
        assert nlp_agent.is_negative(phrase), f"Failed to recognize negative phrase: '{phrase}'"


# ==========================================
# 2. Voice Conversation Turns
# ==========================================

@pytest.mark.anyio
async def test_voice_turn_new_complaint_flow(db):
    """
    Citizen speaks: 'माझ्या गल्लीत कचरा साचला आहे खूप दुर्गंधी सुटली आहे'
    Voice service transcribes audio, routes through conversational_service,
    extracts category (garbage), and synthesizes spoken reply.
    """
    session_id = f"voice-test-{uuid.uuid4().hex[:8]}"
    transcript_text = "माझ्या गल्लीत कचरा साचला आहे खूप दुर्गंधी सुटली आहे"

    mock_stt_res = {
        "success": True,
        "text": transcript_text,
        "language": "mr",
        "duration_seconds": 2.4,
        "engine": "faster-whisper (small)"
    }
    mock_tts_res = {
        "success": True,
        "audio_base64": "RIFF_MOCK_WAV_BASE64",
        "audio_format": "wav",
        "audio_available": True
    }

    with patch.object(voice_service.stt, "transcribe_audio", return_value=mock_stt_res), \
         patch.object(voice_service.tts, "synthesize_speech", return_value=mock_tts_res):
        turn_result = await voice_service.process_voice_turn(
            db=db,
            audio_bytes=b"RIFF_FAKE_AUDIO_DATA_FOR_TEST",
            audio_format="wav",
            language="mr",
            session_id=session_id,
            citizen_phone="9876543210"
        )

        assert turn_result["transcript"] == transcript_text
        assert turn_result["reply"] is not None
        assert len(turn_result["reply"]) > 5
        assert turn_result.get("category") in ["garbage", "waste_management", "solid_waste", "health_sanitation"]
        assert turn_result["voice_reply"] is True
        assert turn_result.get("audio_base64") is not None


@pytest.mark.anyio
async def test_voice_category_correction(db):
    """
    Citizen clarifies: 'नाही माझी तक्रार पाणी गळती बाबत आहे'
    AI must switch category to water leakage / supply.
    """
    session_id = f"voice-test-{uuid.uuid4().hex[:8]}"

    conversational_service.handle_chat(
        db=db,
        message="रस्त्यावर खड्डा आहे",
        session_id=session_id
    )

    correction_text = "नाही माझी तक्रार पाणी गळती बाबत आहे पाईप फुटला आहे"
    mock_stt_res = {
        "success": True,
        "text": correction_text,
        "language": "mr",
        "duration_seconds": 2.0
    }
    mock_tts_res = {
        "success": True,
        "audio_base64": "RIFF_MOCK_WAV_BASE64",
        "audio_format": "wav",
        "audio_available": True
    }

    with patch.object(voice_service.stt, "transcribe_audio", return_value=mock_stt_res), \
         patch.object(voice_service.tts, "synthesize_speech", return_value=mock_tts_res):
        turn_result = await voice_service.process_voice_turn(
            db=db,
            audio_bytes=b"RIFF_FAKE_AUDIO_DATA_FOR_TEST",
            audio_format="wav",
            language="mr",
            session_id=session_id
        )

        assert "पाणी" in turn_result["reply"] or turn_result.get("category") in ["water_supply", "pipeline_water_leakage"]
        assert turn_result.get("voice_reply") is True


@pytest.mark.anyio
async def test_voice_registration_rejection_cancels_complaint(db):
    """
    Citizen says 'नाही' / clicks cancel at the confirmation step.
    AI must cancel the complaint draft and confirm cancellation.
    """
    session_id = f"voice-test-reject-{uuid.uuid4().hex[:8]}"

    # Turn 1: problem
    conversational_service.handle_chat(db=db, message="लाईट बंद आहे", session_id=session_id)
    # Turn 2: location
    conversational_service.handle_chat(db=db, message="ढोरे नगर, सांगवी", session_id=session_id)

    # Turn 3: reject confirmation via voice
    mock_stt_res = {"success": True, "text": "नाही", "language": "mr", "duration_seconds": 1.0}
    mock_tts_res = {"success": True, "audio_base64": "RIFF_MOCK_WAV_BASE64", "audio_format": "wav", "audio_available": True}

    with patch.object(voice_service.stt, "transcribe_audio", return_value=mock_stt_res), \
         patch.object(voice_service.tts, "synthesize_speech", return_value=mock_tts_res):
        turn_result = await voice_service.process_voice_turn(
            db=db,
            audio_bytes=b"RIFF_FAKE_AUDIO_DATA",
            action="cancel",
            language="mr",
            session_id=session_id
        )

        assert turn_result.get("ticket_data") is None
        assert "रद्द" in turn_result["reply"]
        assert turn_result.get("intent") == "REGISTRATION_CANCELLED"


@pytest.mark.anyio
async def test_voice_complaint_cancellation(db):
    """
    Citizen says: 'तक्रार रद्द करा'
    AI must handle cancellation protocol asking for confirmation or closing ticket.
    """
    session_id = f"voice-test-{uuid.uuid4().hex[:8]}"

    conversational_service.handle_chat(
        db=db,
        message="पाण्याची लाईन फुटली आहे",
        session_id=session_id
    )

    cancel_text = "तक्रार रद्द करा"
    mock_stt_res = {
        "success": True,
        "text": cancel_text,
        "language": "mr",
        "duration_seconds": 1.2
    }
    mock_tts_res = {
        "success": True,
        "audio_base64": "RIFF_MOCK_WAV_BASE64",
        "audio_format": "wav",
        "audio_available": True
    }

    with patch.object(voice_service.stt, "transcribe_audio", return_value=mock_stt_res), \
         patch.object(voice_service.tts, "synthesize_speech", return_value=mock_tts_res):
        turn_result = await voice_service.process_voice_turn(
            db=db,
            audio_bytes=b"RIFF_FAKE_AUDIO_DATA",
            language="mr",
            session_id=session_id
        )

        assert "रद्द" in turn_result["reply"] or turn_result.get("intent") in ["CANCEL_CONFIRMATION_REQUIRED", "CANCEL_COMPLAINT"]


@pytest.mark.anyio
async def test_voice_abusive_language_moderation(db):
    """
    If citizen utters abusive/offensive language in audio transcript,
    moderation filter flags it appropriately.
    """
    session_id = f"voice-test-{uuid.uuid4().hex[:8]}"
    abusive_text = "तू मूर्ख आहेस काम नीट करत नाहीस बकवास सर्व्हिस"

    mock_stt_res = {
        "success": True,
        "text": abusive_text,
        "language": "mr",
        "duration_seconds": 2.0
    }

    with patch.object(voice_service.stt, "transcribe_audio", return_value=mock_stt_res):
        turn_result = await voice_service.process_voice_turn(
            db=db,
            audio_bytes=b"RIFF_FAKE_AUDIO_DATA",
            language="mr",
            session_id=session_id
        )

        assert turn_result.get("ticket_data") is None
        assert len(turn_result["reply"]) > 0


@pytest.mark.anyio
async def test_voice_turn_with_photo_attachment(db):
    """
    Citizen provides spoken description and uploads photo simultaneously.
    Verifies voice turn handles multipart photo alongside voice transcript.
    """
    session_id = f"voice-test-{uuid.uuid4().hex[:8]}"
    photo_data = _dummy_image_bytes()

    mock_stt_res = {
        "success": True,
        "text": "आमच्या गल्लीतील पथदिवा दोन दिवसांपासून बंद आहे",
        "language": "mr",
        "duration_seconds": 2.5
    }
    mock_tts_res = {
        "success": True,
        "audio_base64": "RIFF_MOCK_WAV_BASE64",
        "audio_format": "wav",
        "audio_available": True
    }

    with patch.object(voice_service.stt, "transcribe_audio", return_value=mock_stt_res), \
         patch.object(voice_service.tts, "synthesize_speech", return_value=mock_tts_res):
        turn_result = await voice_service.process_voice_turn(
            db=db,
            audio_bytes=b"RIFF_FAKE_AUDIO_DATA",
            photo_filename="street_light_issue.jpg",
            photo_bytes=photo_data,
            language="mr",
            session_id=session_id
        )

        assert turn_result["transcript"] == "आमच्या गल्लीतील पथदिवा दोन दिवसांपासून बंद आहे"
        assert turn_result.get("category") in ["streetlight", "electrical", "street_light"]
        assert turn_result.get("audio_base64") is not None
