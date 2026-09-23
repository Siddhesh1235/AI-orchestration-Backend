"""
Test suite validating resolution of the 4 Citizen Confusion / Mismatch issues:
1. Information Inquiry vs Premature Complaint Intake ('कचरा कुठे टाकू?')
2. 'पाणी न येणे' categorized as Water Supply, NOT Pipeline Leakage ('पिण्याचं पाणी कसं भरायचं?')
3. Unconfirmed default Ward 21 suppressed when citizen provided no location ('रस्त्यावर गटाराचं झाकण तुटलंय')
4. Human-like conversational persona responses, completely free of raw handbook headings ('प्र. ४:', 'उ:')
"""

import pytest
from app.database.session import SessionLocal
from app.services.conversational_service import conversational_service


def test_issue_1_garbage_inquiry_not_premature_complaint():
    """
    Issue 1:
    Citizen asks: 'आमच्या गल्लीत आज कचऱ्याची गाडी आलीच नाही, कचरा कुठे टाकू?'
    Expected:
    - Bot must provide helpful municipal waste guidance.
    - Bot must NOT say 'समजले! मी तक्रार कचरा व घनकचरा मध्ये बदलली आहे'.
    - Intent must be CIVIC_KNOWLEDGE_INQUIRY.
    """
    db = SessionLocal()
    try:
        query = "आमच्या गल्लीत आज कचऱ्याची गाडी आलीच नाही, कचरा कुठे टाकू?"
        res = conversational_service.handle_chat(db=db, message=query)

        assert res["intent"] == "CIVIC_KNOWLEDGE_INQUIRY", f"Expected inquiry intent, got {res['intent']}"
        reply = res["reply"]
        assert "मी तक्रार" not in reply, f"Premature category change triggered: {reply}"
        assert "बदलली आहे" not in reply, f"Premature category change triggered: {reply}"
        assert any(w in reply for w in ["घंटागाडी", "कचरा", "आरोग्य निरीक्षक", "सकाळी", "गाडी"]), f"Missing guidance: {reply}"
        # Human persona check: starts with warm polite greeting
        assert "नमस्कार" in reply
        # Must not contain textbook raw headings
        assert "प्र. " not in reply
    finally:
        db.close()


def test_issue_2_water_supply_inquiry_not_pipeline_leakage():
    """
    Issue 2:
    Citizen asks: 'आज सकाळचं पाणी आलंच नाही, पिण्याचं पाणी कसं भरायचं?'
    Expected:
    - Bot must provide water supply timing / tanker advice.
    - Bot must NOT say 'मी तक्रार पाण्याची पाईपलाईन गळती मध्ये बदलली आहे'.
    - Intent must be CIVIC_KNOWLEDGE_INQUIRY.
    """
    db = SessionLocal()
    try:
        query = "आज सकाळचं पाणी आलंच नाही, पिण्याचं पाणी कसं भरायचं?"
        res = conversational_service.handle_chat(db=db, message=query)

        assert res["intent"] == "CIVIC_KNOWLEDGE_INQUIRY", f"Expected inquiry intent, got {res['intent']}"
        reply = res["reply"]
        assert "पाईपलाईन गळती" not in reply, f"Pipeline leakage falsely applied to water supply: {reply}"
        assert "मी तक्रार" not in reply, f"Premature complaint intake: {reply}"
        assert any(w in reply for w in ["पाणी", "पाणीपुरवठा", "टँकर", "सकाळी", "कुलकर्णी"]), f"Missing water guidance: {reply}"
        assert "नमस्कार" in reply
        assert "प्र. " not in reply
    finally:
        db.close()


def test_issue_3_unconfirmed_default_ward_suppressed():
    """
    Issue 3:
    Citizen reports: 'रस्त्यावर गटाराचं झाकण तुटलंय लहान मुलं पडतील त्वरित माणूस पाठवा'
    Expected:
    - Even if client passes default GPS lat/lng (e.g. 18.6010, 73.7630), bot must NOT announce
      '(प्रभाग 21 - Pimple Nilakh - Vishal Nagar (Zone E))' in greeting because citizen never specified Ward 21.
    - Bot must politely ask for exact street/chowk/landmark.
    - Action prompt should be specify_location_and_details.
    """
    db = SessionLocal()
    try:
        query = "रस्त्यावर गटाराचं झाकण तुटलंय लहान मुलं पडतील त्वरित माणूस पाठवा"
        res = conversational_service.handle_chat(
            db=db,
            message=query,
            latitude=18.6010,
            longitude=73.7630
        )

        reply = res["reply"]
        assert "प्रभाग 21" not in reply, f"Default Ward 21 falsely announced: {reply}"
        assert "Pimple Nilakh" not in reply, f"Default Pimple Nilakh falsely announced: {reply}"
        assert res["category"] in ["drainage", "drainage_failures"], f"Expected drainage, got {res['category']}"
        assert res["action_prompt"] == "specify_location_and_details", f"Expected location prompt, got {res['action_prompt']}"
        assert any(w in reply for w in ["रस्ता", "चौक", "ठिकाण", "कॉलनी"]), f"Expected location request: {reply}"
    finally:
        db.close()


def test_issue_4_human_like_persona_no_raw_handbook_headers():
    """
    Issue 4:
    Citizen asks: 'खांबावरचा दिवा बंद पडलाय रात्री खूप अंधार असतो, काय करू?'
    Expected:
    - Bot provides helpful streetlight resolution advice.
    - Bot does NOT copy-paste raw handbook artifacts like 'प्र. ४:', '(प्र. ४: ...)', or 'उ:'.
    - Response must feel natural and conversational ('नमस्कार! 🙏 ...').
    """
    db = SessionLocal()
    try:
        query = "खांबावरचा दिवा बंद पडलाय रात्री खूप अंधार असतो, काय करू?"
        res = conversational_service.handle_chat(db=db, message=query)

        reply = res["reply"]
        # Must not contain raw handbook markdown question/answer markers
        assert "प्र. " not in reply, f"Raw question label present: {reply}"
        assert "उ:" not in reply, f"Raw answer label present: {reply}"
        assert "**उ:**" not in reply, f"Raw answer label present: {reply}"
        assert "माहिती (प्र." not in reply, f"Raw handbook header present: {reply}"
        # Must contain human greeting and helpful content
        assert "नमस्कार" in reply, f"Missing warm greeting: {reply}"
        assert any(w in reply for w in ["दिवा", "पथदिवा", "विद्युत", "चौधरी", "४८"]), f"Missing streetlight advice: {reply}"
    finally:
        db.close()
