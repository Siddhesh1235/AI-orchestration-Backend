"""
Unit and integration tests for Human Persona Service and PCMC Civic Knowledge Base.
Validates:
1. Warm, humanized conversational tone in Marathi and English.
2. Direct knowledge base resolution (Birth/Death certificates, Property Tax, Waste Schedule, Helplines).
3. Dynamic ward selection and landmark resolution.
"""

import pytest
from app.database.session import SessionLocal
from app.services.human_persona_service import human_persona_service
from app.services.conversational_service import conversational_service


def test_pcmc_knowledge_base_birth_certificate():
    """Test citizen asking how to get a birth/death certificate."""
    ans_mr = human_persona_service.find_knowledge_answer("जन्म दाखला कसा काढायचा?", lang="mr")
    assert ans_mr is not None
    assert "pcmcindia.gov.in" in ans_mr
    assert "दाखला" in ans_mr

    ans_en = human_persona_service.find_knowledge_answer("how to get birth certificate in PCMC?", lang="en")
    assert ans_en is not None
    assert "pcmcindia.gov.in" in ans_en
    assert "Certificate" in ans_en


def test_pcmc_knowledge_base_property_tax():
    """Test citizen asking about paying property tax online."""
    ans_mr = human_persona_service.find_knowledge_answer("घरपट्टी किंवा मालमत्ता कर कसा भरायचा?", lang="mr")
    assert ans_mr is not None
    assert "ptax.pcmcindia.gov.in" in ans_mr

    ans_en = human_persona_service.find_knowledge_answer("how to pay property tax online?", lang="en")
    assert ans_en is not None
    assert "ptax.pcmcindia.gov.in" in ans_en


def test_pcmc_knowledge_base_garbage_schedule():
    """Test citizen asking for garbage van timing."""
    ans_mr = human_persona_service.find_knowledge_answer("कचऱ्याची गाडी कधी येते वेळापत्रक काय आहे?", lang="mr")
    assert ans_mr is not None
    assert "घंटागाडी" in ans_mr or "कचरा संकलन" in ans_mr

    ans_en = human_persona_service.find_knowledge_answer("what is the garbage van timing?", lang="en")
    assert ans_en is not None
    assert "Door-to-Door Waste Collection" in ans_en


def test_pcmc_knowledge_base_helplines():
    """Test citizen asking for helpline numbers."""
    ans_mr = human_persona_service.find_knowledge_answer("महापालिकेचा हेल्पलाईन नंबर काय आहे?", lang="mr")
    assert ans_mr is not None
    assert "020-67333333" in ans_mr

    ans_en = human_persona_service.find_knowledge_answer("what is PCMC helpline contact number?", lang="en")
    assert ans_en is not None
    assert "020-67333333" in ans_en


def test_conversational_service_routes_kb_inquiries():
    """Verify handle_chat seamlessly answers knowledge base questions without registering false complaints."""
    db = SessionLocal()
    try:
        res = conversational_service.handle_chat(db=db, message="जन्म दाखला कसा काढायचा?")
        assert res["intent"] == "CIVIC_KNOWLEDGE_INQUIRY"
        assert res["category"] is None
        assert res["ticket_data"] is None
        assert "pcmcindia.gov.in" in res["reply"]

        res_tax = conversational_service.handle_chat(db=db, message="how to pay house property tax?")
        assert res_tax["intent"] == "CIVIC_KNOWLEDGE_INQUIRY"
        assert res_tax["category"] is None
        assert "ptax.pcmcindia.gov.in" in res_tax["reply"]

        res_garbage = conversational_service.handle_chat(db=db, message="कचऱ्याची गाडी कधी येणार?")
        assert res_garbage["intent"] == "CIVIC_KNOWLEDGE_INQUIRY"
        assert "घंटागाडी" in res_garbage["reply"] or "सकाळी" in res_garbage["reply"]
    finally:
        db.close()


def test_human_persona_fallback_warm_tone():
    """Verify fallback response provides warm, structured municipal assistance."""
    resp_mr = human_persona_service.generate_human_response("माझ्या भागात समस्या आहे मार्गदर्शन करा", lang="mr")
    assert len(resp_mr) > 30
    assert any(w in resp_mr for w in ["नमस्कार", "वॉर्डमित्र", "पिंपरी", "मदत", "समस्या", "सेवा"])

    resp_en = human_persona_service.generate_human_response("How can you help me as a citizen?", lang="en")
    assert len(resp_en) > 30
    assert any(w in resp_en for w in ["WardMitra", "PCMC", "assist", "help", "Hello", "civic"])


def test_dynamic_ward_selection_and_detection():
    """Verify that citizen selects ward or text dynamically resolves ward instead of pre-assigned hardcoding."""
    from app.agents.geo_agent import geo_agent

    w_wakad = geo_agent.detect_ward_from_text("वाकड चौकात खड्डा पडला आहे")
    assert w_wakad is not None
    assert w_wakad["ward_number"] == 25
    assert "Wakad" in w_wakad["ward_name"]

    w_bhosari = geo_agent.detect_ward_from_text("bhosari main road water pipeline defect")
    assert w_bhosari is not None
    assert w_bhosari["ward_number"] == 10

    w_nigdi = geo_agent.detect_ward_from_text("निगडी येथे कचरा साचला आहे")
    assert w_nigdi is not None
    assert w_nigdi["ward_number"] == 3
