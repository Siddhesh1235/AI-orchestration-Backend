"""
End-to-End API Integration tests for WardMitra AI Local Voice Interaction Layer.
Tests:
- POST /api/v1/voice/transcribe (multipart file upload)
- POST /api/v1/voice/stt (base64 audio)
- POST /api/v1/voice/synthesize (text to 16kHz speech)
- POST /api/v1/voice/tts (backwards-compatibility alias)
- POST /api/v1/voice/chat (complete multi-turn voice grievance workflow)
- POST /api/v1/voice/chat (registration rejection cancellation workflow)
- Error handling on invalid/empty inputs
"""

import io
import base64
import uuid
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.services.voice.voice_service import voice_service


client = TestClient(app)


def _mock_stt_result(text="लाईट बंद आहे", lang="mr"):
    return {
        "success": True,
        "text": text,
        "transcript": text,
        "language": lang,
        "language_probability": 0.98,
        "duration_seconds": 1.5,
        "model_used": "faster-whisper-base-local"
    }


def _mock_tts_result(text="नमस्कार! मी आपली मदत करू शकतो."):
    return {
        "success": True,
        "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
        "audio_format": "wav",
        "language": "mr",
        "text": text,
        "audio_available": True
    }


def test_e2e_voice_transcribe_audio_file():
    """POST /api/v1/voice/transcribe with an uploaded audio file."""
    fake_wav_bytes = b"RIFF" + b"\x00" * 100
    mock_res = _mock_stt_result("रस्त्यावर कचरा साचला आहे", "mr")

    with patch.object(voice_service.stt, "transcribe_audio", return_value=mock_res):
        response = client.post(
            "/api/v1/voice/transcribe",
            files={"audio": ("complaint.wav", io.BytesIO(fake_wav_bytes), "audio/wav")},
            data={"language": "mr", "audio_format": "wav"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["text"] == "रस्त्यावर कचरा साचला आहे"
        assert data["transcript"] == "रस्त्यावर कचरा साचला आहे"


def test_e2e_voice_stt_base64():
    """POST /api/v1/voice/stt with base64 encoded audio."""
    b64_str = base64.b64encode(b"RIFF" + b"\x00" * 100).decode("utf-8")
    mock_res = _mock_stt_result("पाणीपुरवठा बंद आहे", "mr")

    with patch.object(voice_service.stt, "transcribe_base64", return_value=mock_res):
        response = client.post(
            "/api/v1/voice/stt",
            data={"audio_base64": b64_str, "language": "mr", "audio_format": "wav"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "पाणीपुरवठा" in data["text"]


def test_e2e_voice_synthesize():
    """POST /api/v1/voice/synthesize converts text to spoken audio."""
    mock_res = _mock_tts_result("आपली तक्रार नोंदवली आहे.")

    with patch.object(voice_service.tts, "synthesize_speech", return_value=mock_res):
        payload = {
            "text": "आपली तक्रार नोंदवली आहे.",
            "language": "mr",
            "gender": "female"
        }
        response = client.post("/api/v1/voice/synthesize", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["audio_available"] is True
        assert data["audio_format"] == "wav"
        assert len(data["audio_base64"]) > 0


def test_e2e_voice_tts_alias():
    """POST /api/v1/voice/tts backwards compatibility alias."""
    mock_res = _mock_tts_result("धन्यवाद!")

    with patch.object(voice_service.tts, "synthesize_speech", return_value=mock_res):
        payload = {
            "text": "धन्यवाद!",
            "language": "mr"
        }
        response = client.post("/api/v1/voice/tts", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


def test_e2e_voice_chat_turn_workflow():
    """POST /api/v1/voice/chat completes a voice conversation turn."""
    fake_wav_bytes = b"RIFF" + b"\x00" * 100
    mock_stt = _mock_stt_result("सांगवी प्रभाग ३२ चे कार्यालय कुठे आहे?", "mr")
    mock_tts = _mock_tts_result("सांगवी प्रभाग ३२ चे कार्यालय अहिल्यादेवी होळकर मनपा शाळा आवार येथे आहे.")
    session_id = f"voice-e2e-{uuid.uuid4().hex[:8]}"

    with patch.object(voice_service.stt, "transcribe_audio", return_value=mock_stt), \
         patch.object(voice_service.tts, "synthesize_speech", return_value=mock_tts), \
         patch("app.clients.llm_client.llm_client.generate", return_value="सांगवी प्रभाग ३२ चे कार्यालय अहिल्यादेवी होळकर मनपा शाळा आवार येथे आहे."):
        response = client.post(
            "/api/v1/voice/chat",
            files={"audio": ("query.wav", io.BytesIO(fake_wav_bytes), "audio/wav")},
            data={
                "session_id": session_id,
                "language": "mr",
                "voice_reply": "true"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["transcribed_text"] == "सांगवी प्रभाग ३२ चे कार्यालय कुठे आहे?"
        assert "reply" in data
        assert data["audio_available"] is True
        assert len(data["audio_base64"]) > 0


def test_e2e_voice_chat_reject_cancels_complaint():
    """
    POST /api/v1/voice/chat with citizen rejecting registration (action='cancel').
    Must immediately cancel complaint and reset state.
    """
    session_id = f"voice-e2e-reject-{uuid.uuid4().hex[:8]}"

    # Turn 1: text message to set up complaint draft
    client.post("/api/v1/complaints/chat", data={"message": "लाईट बंद आहे", "session_id": session_id})
    # Turn 2: location
    client.post("/api/v1/complaints/chat", data={"message": "ढोरे नगर, सांगवी", "session_id": session_id})

    # Turn 3: reject confirmation via /api/v1/voice/chat
    fake_wav = b"RIFF" + b"\x00" * 50
    mock_stt = _mock_stt_result("नाही", "mr")
    mock_tts = _mock_tts_result("तक्रार नोंदणी रद्द करण्यात आली आहे.")

    with patch.object(voice_service.stt, "transcribe_audio", return_value=mock_stt), \
         patch.object(voice_service.tts, "synthesize_speech", return_value=mock_tts):
        response = client.post(
            "/api/v1/voice/chat",
            files={"audio": ("reject.wav", io.BytesIO(fake_wav), "audio/wav")},
            data={
                "session_id": session_id,
                "action": "cancel",
                "language": "mr"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data.get("ticket_data") is None
        assert "रद्द" in data["reply"]
        assert data.get("intent") == "REGISTRATION_CANCELLED"


def test_e2e_voice_empty_inputs_validation():
    """Verify validation errors for missing or empty inputs."""
    # 1. Transcribe without audio or audio_base64
    res1 = client.post("/api/v1/voice/transcribe", data={"language": "mr"})
    assert res1.status_code == 400
    assert "Either 'audio' file or 'audio_base64'" in res1.json()["detail"]

    # 2. Synthesize with empty text
    res2 = client.post("/api/v1/voice/synthesize", json={"text": "", "language": "mr"})
    assert res2.status_code == 400
    assert "text" in res2.json()["detail"].lower()
