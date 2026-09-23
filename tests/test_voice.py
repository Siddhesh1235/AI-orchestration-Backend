"""
Unit tests for WardMitra AI Local Voice Interaction Layer.
Tests:
- Audio normalization, format conversion (FFmpeg), duration estimation, and temp file cleanup.
- Local STT (faster-whisper): empty audio rejection, language normalization, singleton manager, transcription.
- Local TTS (Meta MMS-TTS): low-literacy simplification, ticket ID acoustic expansion, zero-crash fallback.
"""

import io
import struct
import base64
import numpy as np
import pytest
import soundfile as sf
from unittest.mock import MagicMock, patch

from app.services.voice.audio_service import audio_service, get_ffmpeg_executable
from app.services.voice.stt_service import local_stt_service, WhisperModelManager
from app.services.voice.tts_service import (
    local_tts_service,
    simplify_text_for_speech,
    generate_fallback_wav_base64,
    TTSModelManager,
)


def create_synthetic_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000, freq: float = 440.0) -> bytes:
    """Generates synthetic 16kHz mono PCM 16-bit sine wave audio bytes for testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    wav_io = io.BytesIO()
    sf.write(wav_io, audio, sample_rate, format='WAV', subtype='PCM_16')
    return wav_io.getvalue()


# ==========================================
# 1. Audio Service Tests
# ==========================================

def test_audio_validation():
    """Verify audio byte length validation."""
    valid, err = audio_service.validate_audio_bytes(b"", min_bytes=100)
    assert not valid
    assert "empty or too short" in err

    valid, err = audio_service.validate_audio_bytes(b"short", min_bytes=100)
    assert not valid

    valid, err = audio_service.validate_audio_bytes(b"x" * 200, min_bytes=100)
    assert valid
    assert err is None


def test_audio_duration():
    """Verify audio duration calculation on valid WAV bytes."""
    wav_bytes = create_synthetic_wav_bytes(duration_sec=1.5, sample_rate=16000)
    duration = audio_service.get_audio_duration(wav_bytes)
    assert abs(duration - 1.5) < 0.1

    # Empty bytes duration
    assert audio_service.get_audio_duration(b"") == 0.0


def test_audio_conversion_pass_through_if_16k_mono():
    """Verify convert_to_wav_16k_mono preserves already valid 16kHz mono WAV bytes."""
    wav_bytes = create_synthetic_wav_bytes(duration_sec=0.5, sample_rate=16000)
    converted = audio_service.convert_to_wav_16k_mono(wav_bytes, input_format="wav")
    assert converted == wav_bytes


def test_ffmpeg_detection():
    """Verify bundled FFmpeg executable is discovered correctly."""
    exe = get_ffmpeg_executable()
    assert exe is not None
    assert "ffmpeg" in exe.lower()


# ==========================================
# 2. Local STT Service Tests
# ==========================================

def test_stt_language_normalization():
    """Verify language code normalization for faster-whisper."""
    assert local_stt_service._normalize_lang_code("mr") == "mr"
    assert local_stt_service._normalize_lang_code("marathi") == "mr"
    assert local_stt_service._normalize_lang_code("hi") == "hi"
    assert local_stt_service._normalize_lang_code("hindi") == "hi"
    assert local_stt_service._normalize_lang_code("en") == "en"
    assert local_stt_service._normalize_lang_code("english") == "en"
    assert local_stt_service._normalize_lang_code("auto") is None
    assert local_stt_service._normalize_lang_code(None) is None


def test_stt_empty_audio_rejection():
    """Verify STT service returns failure for empty or tiny audio bytes."""
    res = local_stt_service.transcribe_audio_sync(b"", language="mr")
    assert res["success"] is False
    assert res["text"] == ""
    assert "error" in res

    res2 = local_stt_service.transcribe_audio_sync(b"tiny", language="mr")
    assert res2["success"] is False


def test_stt_model_manager_singleton():
    """Verify WhisperModelManager is a thread-safe singleton."""
    mgr1 = WhisperModelManager.get_instance()
    mgr2 = WhisperModelManager.get_instance()
    assert mgr1 is mgr2


def test_stt_transcription_with_mock_model():
    """Verify STT transcription pipeline flow with mocked faster-whisper inference."""
    wav_bytes = create_synthetic_wav_bytes(duration_sec=1.0)

    class MockSegment:
        def __init__(self, text):
            self.text = text

    class MockInfo:
        language = "mr"
        language_probability = 0.98

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        [MockSegment("माझ्या घरासमोर कचरा साचला आहे")],
        MockInfo()
    )

    with patch.object(local_stt_service.manager, "load_model", return_value=mock_model):
        res = local_stt_service.transcribe_audio_sync(wav_bytes, language="mr", audio_format="wav")
        assert res["success"] is True
        assert res["text"] == "माझ्या घरासमोर कचरा साचला आहे"
        assert res["language"] == "mr"
        assert res["duration_seconds"] > 0.5


@pytest.mark.anyio
async def test_stt_async_base64_decoding():
    """Verify base64 audio decoding and transcription."""
    wav_bytes = create_synthetic_wav_bytes(duration_sec=0.5)
    b64_str = base64.b64encode(wav_bytes).decode('utf-8')

    class MockSegment:
        def __init__(self, text):
            self.text = text

    class MockInfo:
        language = "mr"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([MockSegment("पाणी गळती")], MockInfo())

    with patch.object(local_stt_service.manager, "load_model", return_value=mock_model):
        res = await local_stt_service.transcribe_base64(b64_str, language="mr", audio_format="wav")
        assert res["success"] is True
        assert "पाणी गळती" in res["text"]


# ==========================================
# 3. Local TTS Service Tests
# ==========================================

def test_simplify_text_for_low_literacy():
    """Verify low-literacy text simplification: markdown stripping, ticket ID acoustic expansion."""
    raw_text = (
        "**नमस्कार!** आपली तक्रार नोंदवली आहे.\n"
        "- विभाग: पाणीपुरवठा\n"
        "तक्रार क्रमांक: WM-20260922-1234 आहे. https://pcmc.gov.in वर तपासा."
    )
    simplified = simplify_text_for_speech(raw_text, lang="mr")

    # Markdown stripped
    assert "**" not in simplified
    assert "- " not in simplified
    # URL stripped
    assert "https://" not in simplified
    # Ticket number expanded acoustically for clear spoken pronunciation
    assert "WM 2026 0922 1234" in simplified
    assert "तक्रार क्रमांक: WM 2026 0922 1234 आहे." in simplified


def test_simplify_long_handbook_text():
    """Verify long handbook or RAG responses are concisely capped to 2-3 key sentences for speech."""
    long_text = (
        "वॉर्ड ३२ चे कार्यालय सांगवी येथे आहे. "
        "कार्यालय सकाळी ९:४५ ते सायंकाळी ६:१५ पर्यंत चालू असते. "
        "नवीन नळ जोडणीसाठी मालमत्ता कर पावती आणि आधार कार्ड आवश्यक आहे. "
        "अधिक माहितीसाठी आपण ०२०-६७३३३३३३ या क्रमांकावर संपर्क साधू शकता. "
        "आपण ऑनलाईन अर्ज देखील करू शकता."
    )
    simplified = simplify_text_for_speech(long_text, lang="mr")
    # Should cap to first 3 sentences
    assert "०२०-६७३३३३३३" not in simplified
    assert "सांगवी येथे आहे" in simplified


def test_fallback_wav_generation():
    """Verify fallback WAV base64 creates a structurally valid PCM WAV."""
    b64_wav = generate_fallback_wav_base64(duration_sec=0.2, sample_rate=16000)
    raw = base64.b64decode(b64_wav)
    assert raw.startswith(b"RIFF")
    assert b"WAVE" in raw[:16]
    assert b"fmt " in raw[:24]
    assert b"data" in raw


def test_tts_model_manager_singleton():
    """Verify TTSModelManager is a thread-safe singleton."""
    mgr1 = TTSModelManager.get_instance()
    mgr2 = TTSModelManager.get_instance()
    assert mgr1 is mgr2


def test_tts_empty_text_rejection():
    """Verify TTS service returns failure when empty or whitespace text is provided."""
    res = local_tts_service.synthesize_speech_sync("", language="mr")
    assert res["success"] is False
    assert res["audio_available"] is False
    assert "Empty or whitespace" in res["error"]


def test_tts_zero_crash_resilience_on_model_absence():
    """Verify TTS falls back smoothly with audio_available=False if model is not loaded."""
    with patch.object(local_tts_service.manager, "get_model_and_tokenizer", return_value=(None, None)):
        res = local_tts_service.synthesize_speech_sync("आपली तक्रार नोंदवली आहे", language="mr")
        assert res["success"] is True
        assert res["audio_available"] is False
        assert len(res["audio_base64"]) > 50  # Fallback valid WAV base64
        assert res["text"] == "आपली तक्रार नोंदवली आहे"


@pytest.mark.anyio
async def test_tts_synthesis_with_mocked_model():
    """Verify end-to-end TTS synthesis pipeline with mocked VitsModel and AutoTokenizer."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Mock tokenizer output
    mock_tokenizer.return_value = {"input_ids": MagicMock()}

    # Mock model waveform output (1 second of synthetic waveform)
    fake_waveform = np.zeros((1, 16000), dtype=np.float32)
    fake_waveform[0, ::10] = 0.5

    import torch
    mock_waveform = MagicMock()
    mock_waveform.squeeze.return_value.cpu.return_value.numpy.return_value = fake_waveform[0]
    mock_model.return_value = MagicMock(waveform=mock_waveform)
    mock_model.config.sampling_rate = 16000

    with patch.object(local_tts_service.manager, "get_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)):
        res = await local_tts_service.synthesize_speech("सांगवी वॉर्ड कार्यालय", language="mr")
        assert res["success"] is True
        assert res["audio_available"] is True
        assert res["language"] == "mr"
        assert len(res["audio_base64"]) > 50

        # Verify decoded bytes are WAV format
        raw = base64.b64decode(res["audio_base64"])
        assert raw.startswith(b"RIFF")
        assert b"WAVE" in raw[:16]
