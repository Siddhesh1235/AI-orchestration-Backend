# -*- coding: utf-8 -*-
"""
Test suite validating the 5 Mismatches & Confusion Resolutions for WardMitra AI:
1. "🔍 स्थिती तपासा / Track Status" button handler and fallback ticket tracking.
2. Session Language Locking ('Datta nagar nigdi') and sub-category protection (household waste).
3. Citizen privacy question ("मोबाईल नंबर इतरांना दिसेल का?") returns polite direct negation ("नाही, मुळीच नाही!").
4. "सारथी कॉल सेंटर vs वॉर्डमित्र AI मधील फरक" returns structured comparison, not just helpline list.
5. Ward 15 office + Health Inspector Suresh Waghmare (9822100032) retrieved together.
"""

import pytest
from app.database.session import SessionLocal
from app.database.models import Complaint, ComplaintStatus
from app.services.conversational_service import conversational_service
from app.services.conversation_state_manager import ConversationState


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_track_status_button_duplicate_ticket(db_session):
    """
    Issue 1:
    - Button click with action: 'track_WM-20260923-9561' directly loads status card.
    - Does NOT restart old complaint or ask for location.
    - Fallback from state.last_duplicate_ticket works when citizen types 'Status' or 'स्थिती तपासा'.
    - If no ticket exists anywhere, asks for ticket ID politely without starting intake.
    """
    # 1. Create a dummy complaint in DB
    ticket_id = "WM-20260923-9561"
    existing = db_session.query(Complaint).filter(Complaint.ticket_id == ticket_id).first()
    if not existing:
        complaint = Complaint(
            ticket_id=ticket_id,
            citizen_phone="9876543210",
            detected_category="garbage",
            description="कचऱ्याची समस्या",
            assigned_department="Health / Solid Waste",
            status=ComplaintStatus.ASSIGNED,
            ward_number=15,
            assigned_worker_name="Er. संदीप माने",
            assigned_worker_contact="9822100032",
            sla_hours=24
        )
        db_session.add(complaint)
        db_session.commit()

    # Part A: Test clicking button with action="track_WM-20260923-9561"
    res = conversational_service.handle_chat(
        db=db_session,
        message="",
        action=f"track_{ticket_id}"
    )

    assert res["intent"] == "TRACK_COMPLAINT", f"Expected TRACK_COMPLAINT, got {res['intent']}"
    assert res["ticket_data"] is not None
    assert res["ticket_data"]["ticket_id"] == ticket_id
    reply = res["reply"]
    assert ticket_id in reply
    assert "ठिकाण" not in reply or "तक्रार स्थिती" in reply, f"Intake restarted: {reply}"
    assert "तक्रार निश्चित केली आहे" not in reply, f"Intake confirmation triggered: {reply}"

    # Part B: Test fallback from state.last_duplicate_ticket when user types 'स्थिती तपासा'
    session_id = "test_dup_track_session"
    state = conversational_service.state_manager.get_or_create(session_id)
    state.last_duplicate_ticket = ticket_id
    state.language = "mr"

    res_status = conversational_service.handle_chat(
        db=db_session,
        message="स्थिती तपासा",
        session_id=session_id
    )
    assert res_status["intent"] == "TRACK_COMPLAINT"
    assert res_status["ticket_data"]["ticket_id"] == ticket_id
    assert ticket_id in res_status["reply"]

    # Part C: Test 'Status' with no ticket anywhere -> asks politely for Ticket ID, no intake
    fresh_session = "fresh_empty_session_track"
    res_empty = conversational_service.handle_chat(
        db=db_session,
        message="Status",
        session_id=fresh_session
    )
    assert res_empty["intent"] == "TRACK_COMPLAINT"
    assert res_empty["action_prompt"] == "specify_ticket_id"
    assert "कृपया आपला तक्रार क्रमांक" in res_empty["reply"] or "Please provide your Ticket ID" in res_empty["reply"]


def test_location_and_sub_category_protection(db_session):
    """
    Issue 2:
    - Language locking: 'Datta nagar nigdi' does not reset Marathi to English.
    - Location persistence: photo/text turn records 'Datta nagar nigdi' into state.location.
    - Sub-category protection: 'household waste' goes to issue_type / sub_category, NEVER location.
    - Transitions to registration confirmation when location is already present.
    """
    session_id = "test_lang_subcat_session"
    state = conversational_service.state_manager.get_or_create(session_id)

    # Turn 1: Citizen speaks Marathi about garbage
    res1 = conversational_service.handle_chat(
        db=db_session,
        message="आमच्या इथे खूप कचरा साचला आहे",
        session_id=session_id
    )
    assert res1["language"] == "mr"
    state = conversational_service.state_manager.get(session_id)
    assert state.language == "mr"
    assert state.complaint_category in ["garbage", "illegal_debris_dumping"]

    # Turn 2: Citizen provides location 'Datta nagar nigdi'
    res2 = conversational_service.handle_chat(
        db=db_session,
        message="Datta nagar nigdi",
        session_id=session_id
    )
    # Session Language Locking check: must still be 'mr'
    assert res2["language"] == "mr", f"Language switched to English unexpectedly: {res2['language']}"
    state = conversational_service.state_manager.get(session_id)
    assert state.language == "mr"
    assert state.location is not None
    assert "datta nagar nigdi" in state.location.lower()

    # Turn 3: Citizen replies with garbage sub-category 'household waste'
    res3 = conversational_service.handle_chat(
        db=db_session,
        message="household waste",
        session_id=session_id
    )
    state = conversational_service.state_manager.get(session_id)
    # Sub-category protection check: location must NOT be overwritten with 'household waste'
    assert "household waste" not in (state.location or "").lower(), f"Subcategory was falsely set as location: {state.location}"
    assert "datta nagar nigdi" in (state.location or "").lower()
    assert state.issue_type == "household_waste" or state.sub_category == "household_waste"
    # Action prompt should transition to registration confirmation
    assert res3["action_prompt"] == "confirm_register"


def test_privacy_number_visibility_polite_negation(db_session):
    """
    Issue 3:
    - Citizen asks 'मोबाईल नंबर इतरांना दिसेल का?'
    - Response must start with polite direct negation ('नाही, मुळीच नाही!'), NOT 'होय, पूर्णपणे!'.
    - Checks Marathi, Hindi, and English.
    """
    # Marathi check
    res_mr = conversational_service.handle_chat(
        db=db_session,
        message="मोबाईल नंबर इतरांना दिसेल का?"
    )
    reply_mr = res_mr["reply"]
    assert "नाही, मुळीच नाही!" in reply_mr, f"Missing polite negation in Marathi: {reply_mr}"
    assert "होय, पूर्णपणे!" not in reply_mr, f"Contradictory affirmation present in Marathi: {reply_mr}"
    assert "गोपनीयता" in reply_mr

    # Marathi variant: 'माझा मोबाईल नंबर इतर कोणाला दिसेल का?'
    res_mr2 = conversational_service.handle_chat(
        db=db_session,
        message="माझा मोबाईल नंबर इतर कोणाला दिसेल का?"
    )
    assert "नाही, मुळीच नाही!" in res_mr2["reply"]
    assert "होय, पूर्णपणे!" not in res_mr2["reply"]

    # Hindi check
    res_hi = conversational_service.handle_chat(
        db=db_session,
        message="क्या मेरा मोबाइल नंबर दूसरों को दिखेगा?"
    )
    reply_hi = res_hi["reply"]
    assert "नहीं, बिल्कुल नहीं!" in reply_hi, f"Missing polite negation in Hindi: {reply_hi}"
    assert "हाँ" not in reply_hi

    # English check
    res_en = conversational_service.handle_chat(
        db=db_session,
        message="Will my mobile number be visible to others?"
    )
    reply_en = res_en["reply"]
    assert "No, absolutely not!" in reply_en, f"Missing polite negation in English: {reply_en}"
    assert "Yes" not in reply_en


def test_sarathi_vs_wardmitra_comparison(db_session):
    """
    Issue 4:
    - Citizen asks: 'सारथी कॉल सेंटर vs वॉर्डमित्र AI मधील फरक'
    - Response must be structured comparison between Call Center and WardMitra AI.
    - Must NOT be simply the raw helpline list.
    """
    queries = [
        "सारथी कॉल सेंटर vs वॉर्डमित्र AI मधील फरक",
        "सारथी कॉल सेंटर आणि वॉर्डमित्र AI यात काय फरक आहे",
        "difference between sarathi and wardmitra"
    ]

    for q in queries:
        res = conversational_service.handle_chat(db=db_session, message=q)
        reply = res["reply"]
        assert "सारथी कॉल सेंटर" in reply or "Sarathi Call Center" in reply, f"Missing Sarathi comparison for '{q}': {reply}"
        assert "वॉर्डमित्र AI" in reply or "WardMitra AI" in reply, f"Missing WardMitra comparison for '{q}': {reply}"
        # Must mention Sarathi helpline number and hours
        assert "020-67333333" in reply
        # Must highlight key differences (e.g. 24x7, photo/vision)
        assert any(w in reply for w in ["२४x७", "२४ तास", "24x7", "24 hours", "फोटो", "Photo", "AI"])
        # Must not be the disaster/fire brigade helpline dump
        assert "अग्निशामक दल" not in reply, f"Helpline numbers dumped instead of comparison: {reply}"


def test_ward_15_office_and_health_inspector(db_session):
    """
    Issue 5:
    - Citizen asks about Ward 15 office and Sanitary/Health Inspector.
    - Response must provide both:
      1. Ward 15 Office address (महापालिका भवन / मुंबई-पुणे महामार्ग)
      2. Health Inspector Shri Suresh Waghmare (9822100032)
    """
    query = "प्रभाग १५ चे कार्यालय आणि आरोग्य निरीक्षक कोण आहेत?"
    res = conversational_service.handle_chat(db=db_session, message=query)
    reply = res["reply"]

    # Verify Ward 15 office details
    assert any(w in reply for w in ["महापालिका भवन", "मुंबई-पुणे", "पिंपरी", "क्षेत्रीय कार्यालय", "कार्यालय"]), f"Missing office address: {reply}"

    # Verify Health Inspector Suresh Waghmare and contact
    assert "सुरेश वाघमारे" in reply, f"Missing Health Inspector Suresh Waghmare: {reply}"
    assert "9822100032" in reply, f"Missing Suresh Waghmare contact: {reply}"
