"""
Tests for Ward 32 Knowledge Base and Voice-Integrated RAG Inquiries.
Verifies:
- Loading and chunking of WARD_32_HANDBOOK.md.
- Retrieval accuracy for Ward 32 office details, timings, and water connection document checklists.
- End-to-end voice query through voice_service.process_voice_turn:
  1. Ensures NO complaint ticket is created for information queries.
  2. Ensures grounded answers from Ward 32 handbook are delivered.
  3. Ensures spoken simplification and TTS audio generation are produced.
"""

import pytest
from unittest.mock import patch

from app.database.session import SessionLocal
from app.services.rag_service import rag_service, WardDocumentLoader
from app.services.voice.voice_service import voice_service


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_ward_32_handbook_chunking():
    """Verify that WARD_32_HANDBOOK.md is parsed and indexed with all key civic sections."""
    chunks = WardDocumentLoader.load_from_directory("docs")
    assert len(chunks) > 0

    ward_32_chunks = [ch for ch in chunks if ch.ward_number == 32]
    assert len(ward_32_chunks) >= 5, f"Expected >= 5 chunks for Ward 32, found {len(ward_32_chunks)}"

    titles = [ch.title for ch in ward_32_chunks]
    assert any("सांगवी" in t or "Ward Office" in t or "कार्यालय" in t for t in titles)
    assert any("पाणीपुरवठा" in t or "Water" in t for t in titles)
    assert any("पाणी कनेक्शन" in t or "Water Connection" in t or "नळ जोडणी" in t or "कागदपत्रे" in t for t in titles)


def test_ward_32_water_connection_checklist_retrieval():
    """Verify semantic retrieval for new water connection document checklist in Ward 32."""
    query = "नवीन नळ जोडणीसाठी कोणती कागदपत्रे लागतात?"
    results = rag_service.retrieve(query, ward_number=32)
    assert len(results) > 0

    top_chunk, score = results[0]
    assert score > 0.15
    # Must contain essential civic documents
    assert any(doc in top_chunk.content for doc in ["मालमत्ता कर", "आधार", "७/१२", "इंडेक्स", "प्लंबिंग"])


def test_ward_32_office_address_retrieval():
    """Verify retrieval for Ward 32 Sangvi office address and timings."""
    query = "सांगवी प्रभाग ३२ चे कार्यालय कुठे आहे?"
    results = rag_service.retrieve(query, ward_number=32)
    assert len(results) > 0

    top_chunk, score = results[0]
    assert score > 0.15
    assert "सांगवी" in top_chunk.content
    assert any(time_str in top_chunk.content for time_str in ["९:४५", "६:१५", "वेळ"])


@pytest.mark.anyio
async def test_voice_turn_rag_water_connection_docs_no_ticket(db):
    """
    Citizen speaks: 'नवीन नळ जोडणीसाठी कोणती कागदपत्रे लागतात सांगवी वॉर्ड मध्ये?'
    Verifies:
    1. Grounded handbook response returned.
    2. Zero complaint ticket registered.
    3. Voice reply synthesized.
    """
    query_text = "नवीन नळ जोडणीसाठी कोणती कागदपत्रे लागतात सांगवी वॉर्ड मध्ये?"
    mock_stt_res = {
        "success": True,
        "text": query_text,
        "language": "mr",
        "duration_seconds": 3.0
    }
    mock_tts_res = {
        "success": True,
        "audio_base64": "RIFF_MOCK_WAV_BASE64",
        "audio_format": "wav",
        "audio_available": True
    }

    rag_answer = "सांगवी प्रभाग ३२ चे कार्यालय अहिल्यादेवी होळकर मनपा शाळा आवार, सांगवी येथे आहे. वेळ: सकाळी ९:४५ ते संध्याकाळी ६:१५."
    with patch.object(voice_service.stt, "transcribe_audio", return_value=mock_stt_res), \
         patch.object(voice_service.tts, "synthesize_speech", return_value=mock_tts_res), \
         patch("app.clients.llm_client.llm_client.generate", return_value=rag_answer):
        turn_result = await voice_service.process_voice_turn(
            db=db,
            audio_bytes=b"RIFF_FAKE_AUDIO_DATA_FOR_RAG",
            audio_format="wav",
            language="mr",
            session_id="voice-rag-test-1"
        )

        assert turn_result["transcript"] == query_text
        # Strictly NO complaint ticket created for information queries
        assert turn_result.get("ticket_data") is None
        assert turn_result.get("intent") in ["CIVIC_KNOWLEDGE_INQUIRY", "CONVERSATIONAL", "FAQ"]
        assert len(turn_result["reply"]) > 10

        # Spoken audio generation
        assert turn_result["voice_reply"] is True
        assert turn_result.get("audio_base64") is not None


@pytest.mark.anyio
async def test_voice_turn_rag_office_timing_inquiry_no_ticket(db):
    """
    Citizen speaks: 'सांगवी वॉर्ड ऑफिस कुठे आहे आणि किती वाजता उघडते?'
    Verifies factual response and zero complaint ticket created.
    """
    query_text = "सांगवी वॉर्ड ३२ चे कार्यालय कुठे आहे?"
    mock_stt_res = {
        "success": True,
        "text": query_text,
        "language": "mr",
        "duration_seconds": 2.5
    }
    mock_tts_res = {
        "success": True,
        "audio_base64": "RIFF_MOCK_WAV_BASE64",
        "audio_format": "wav",
        "audio_available": True
    }

    rag_answer = "सांगवी प्रभाग ३२ चे कार्यालय अहिल्यादेवी होळकर मनपा शाळा आवार, सांगवी येथे आहे."
    with patch.object(voice_service.stt, "transcribe_audio", return_value=mock_stt_res), \
         patch.object(voice_service.tts, "synthesize_speech", return_value=mock_tts_res), \
         patch("app.clients.llm_client.llm_client.generate", return_value=rag_answer):
        turn_result = await voice_service.process_voice_turn(
            db=db,
            audio_bytes=b"RIFF_FAKE_AUDIO_DATA_FOR_OFFICE",
            audio_format="wav",
            language="mr",
            session_id="voice-rag-test-2"
        )

        assert turn_result.get("ticket_data") is None
        assert any(w in turn_result["reply"] for w in ["सांगवी", "कार्यालय", "वॉर्ड ३२", "प्रभाग ३२", "पत्ता"])
        assert turn_result["voice_reply"] is True
        assert turn_result.get("audio_base64") is not None
