"""
Automated unit and integration tests for Digital India Bhashini STT Service and APIs.
Verifies ASR transcription, endpoint routing, language normalization, and WhatsApp OGG audio conversion.
"""

import io
import base64
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.bhashini_stt_service import bhashini_stt_service

client = TestClient(app)


def test_bhashini_service_empty_audio():
    """Verify Bhashini service handles empty audio gracefully."""
    res = bhashini_stt_service.transcribe_audio_sync(b"", language="mr")
    assert res["success"] is False
    assert "error" in res


def test_bhashini_service_valid_mock_audio():
    """Verify Bhashini service handles audio bytes and returns valid structure."""
    fake_audio = b"RIFF....WAVEfmt ....data...." * 10
    res = bhashini_stt_service.transcribe_audio_sync(fake_audio, language="mr", audio_format="wav")
    assert res["success"] is True
    assert "language" in res
    assert res["language"] == "mr"
    assert "engine" in res
    assert "Bhashini" in res["engine"]


def test_bhashini_service_base64():
    """Verify Bhashini service accepts base64 audio payload."""
    fake_b64 = base64.b64encode(b"FAKE_AUDIO_DATA_FOR_BHASHINI_PIPELINE").decode("utf-8")
    import asyncio
    res = asyncio.run(bhashini_stt_service.transcribe_base64(fake_b64, language="hi"))
    assert res["success"] is True
    assert res["language"] == "hi"
    assert "engine" in res


def test_voice_stt_endpoint_with_audio_file():
    """Verify POST /api/v1/voice/stt with multipart audio file."""
    fake_audio_file = io.BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    files = {"audio": ("sample.wav", fake_audio_file, "audio/wav")}
    data = {"language": "mr", "audio_format": "wav"}

    response = client.post("/api/v1/voice/stt", files=files, data=data)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert json_data["language"] == "mr"
    assert "engine" in json_data
    assert "Bhashini" in json_data["engine"]


def test_voice_stt_endpoint_missing_payload():
    """Verify POST /api/v1/voice/stt rejects requests with no audio data."""
    response = client.post("/api/v1/voice/stt", data={"language": "mr"})
    assert response.status_code == 400


def test_complaint_chat_with_voice_audio():
    """Verify POST /api/v1/complaints/chat accepts voice audio file."""
    fake_voice_file = io.BytesIO(b"MOCK_BHASHINI_VOICE_AUDIO_RECORDING_DATA")
    files = {"voice": ("voice_record.webm", fake_voice_file, "audio/webm")}
    data = {"message": "पिंपरी मध्ये समस्या आहे", "language": "mr"}

    response = client.post("/api/v1/complaints/chat", files=files, data=data)
    assert response.status_code == 200
    json_data = response.json()
    assert "reply" in json_data
    assert len(json_data["reply"]) > 5


def test_bhashini_language_normalization():
    """Verify language codes normalize correctly for mr, hi, and en."""
    from app.services.bhashini_stt_service import normalize_language_code
    assert normalize_language_code("mr-IN") == "mr"
    assert normalize_language_code("MARATHI") == "mr"
    assert normalize_language_code("hi-IN") == "hi"
    assert normalize_language_code("HINDI") == "hi"
    assert normalize_language_code("en-US") == "en"
    assert normalize_language_code("English") == "en"
    assert normalize_language_code("xyz") == "mr"


def test_whatsapp_ogg_audio_conversion():
    """Verify convert_audio_to_wav_16k_mono handles WhatsApp-style audio formats."""
    from app.services.bhashini_stt_service import convert_audio_to_wav_16k_mono
    fake_ogg_bytes = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 40
    converted = convert_audio_to_wav_16k_mono(fake_ogg_bytes, input_format="ogg")
    assert isinstance(converted, bytes)
    assert len(converted) > 0


def test_voice_stt_endpoint_with_ogg_opus_file():
    """Verify POST /api/v1/voice/stt accepts WhatsApp OGG/Opus voice notes."""
    fake_ogg = io.BytesIO(b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 60)
    files = {"audio": ("whatsapp_voice.ogg", fake_ogg, "audio/ogg")}
    data = {"language": "mr-IN", "audio_format": "ogg"}

    response = client.post("/api/v1/voice/stt", files=files, data=data)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert json_data["language"] == "mr"
    assert "engine" in json_data
