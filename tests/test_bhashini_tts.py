"""
Automated unit and integration tests for Digital India Bhashini TTS Service and APIs.
Verifies text-to-speech synthesis across Marathi (mr), Hindi (hi), and English (en),
validates the POST /api/v1/voice/tts endpoint, and tests end-to-end voice_reply in chat.
"""

import base64
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.bhashini_tts_service import bhashini_tts_service

client = TestClient(app)


def test_bhashini_tts_empty_text():
    """Verify TTS service gracefully rejects empty text strings."""
    res = bhashini_tts_service.synthesize_speech_sync("", language="mr")
    assert res["success"] is False
    assert "error" in res
    assert res["audio_base64"] == ""


def test_bhashini_tts_marathi_synthesis():
    """Verify Bhashini TTS synthesizes Marathi text and returns valid base64 WAV."""
    sample_text = "नमस्कार! PCMC वॉर्डमित्र मध्ये आपले स्वागत आहे."
    res = bhashini_tts_service.synthesize_speech_sync(sample_text, language="mr")
    assert res["success"] is True
    assert res["language"] == "mr"
    assert res["audio_format"] == "wav"
    assert len(res["audio_base64"]) > 50

    raw_wav = base64.b64decode(res["audio_base64"])
    assert raw_wav.startswith(b"RIFF")
    assert b"WAVE" in raw_wav[:16]


def test_bhashini_tts_hindi_and_english_synthesis():
    """Verify Bhashini TTS synthesizes Hindi and English text."""
    res_hi = bhashini_tts_service.synthesize_speech_sync("पिंपरी चिंचवड नगर निगम में आपका स्वागत है।", language="hi")
    assert res_hi["success"] is True
    assert res_hi["language"] == "hi"
    assert len(res_hi["audio_base64"]) > 50

    res_en = bhashini_tts_service.synthesize_speech_sync("Welcome to PCMC WardMitra grievance redressal platform.", language="en")
    assert res_en["success"] is True
    assert res_en["language"] == "en"
    assert len(res_en["audio_base64"]) > 50


def test_voice_tts_endpoint_valid_json():
    """Verify POST /api/v1/voice/tts accepts JSON payload and returns base64 WAV."""
    payload = {
        "text": "आपली तक्रार यशस्वीरित्या नोंदवली गेली आहे.",
        "language": "mr",
        "gender": "female"
    }
    response = client.post("/api/v1/voice/tts", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["language"] == "mr"
    assert data["audio_format"] == "wav"
    assert len(data["audio_base64"]) > 0
    assert any(eng in data.get("engine", "") for eng in ["Bhashini", "Meta MMS-TTS", "TTS"])


def test_voice_tts_endpoint_empty_text():
    """Verify POST /api/v1/voice/tts rejects empty text with 400 Bad Request."""
    payload = {"text": "   ", "language": "mr"}
    response = client.post("/api/v1/voice/tts", json=payload)
    assert response.status_code == 400
    assert "detail" in response.json()


def test_complaint_chat_with_voice_reply_flag():
    """Verify POST /api/v1/complaints/chat synthesizes spoken voice reply when voice_reply=True."""
    data = {
        "message": "नमस्कार",
        "language": "mr",
        "voice_reply": "true"
    }
    response = client.post("/api/v1/complaints/chat", data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert "reply" in res_json
    assert res_json.get("voice_reply") is True
    assert res_json.get("audio_base64") is not None
    assert len(res_json["audio_base64"]) > 20
    assert res_json.get("audio_format") == "wav"


def test_complaint_chat_without_voice_reply_flag():
    """Verify POST /api/v1/complaints/chat does NOT generate voice audio when voice_reply=False."""
    data = {
        "message": "Hello",
        "language": "en",
        "voice_reply": "false"
    }
    response = client.post("/api/v1/complaints/chat", data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert "reply" in res_json
    assert res_json.get("voice_reply") is not True


def test_chat_complaint_alias_endpoint():
    """Verify POST /api/v1/chat/complaint works as a direct alias for /api/v1/complaints/chat."""
    data = {
        "message": "Streetlight is broken in Wakad",
        "language": "en"
    }
    response = client.post("/api/v1/chat/complaint", data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert "reply" in res_json
    assert "action_prompt" in res_json
