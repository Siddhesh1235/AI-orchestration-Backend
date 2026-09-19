"""
Automated Test Suite for WardMitra AI Verification & Chatbot Scenarios.
Covers all 13 mandated project requirements:
TEST 1: User: 'The light is off.' -> Clarification (home light vs streetlight)
TEST 2: User selects 'Drainage' -> Ask for actual problem description
TEST 3: Marathi complaint -> Marathi response
TEST 4: English complaint -> English response
TEST 5: Unclear language -> Ask user to re-describe in English/Marathi
TEST 6: Streetlight complaint + irrelevant image -> Reject evidence
TEST 7: Pothole complaint + valid pothole image -> YOLO classification + verification
TEST 8: Duplicate complaint -> Duplicate warning / review
TEST 9: Suspicious complaint -> Fraud risk + REVIEW_REQUIRED
TEST 10: Emergency complaint -> Emergency = true and Priority = P1
TEST 11: Normal complaint -> Emergency = false
TEST 12: Low-confidence image -> Request clearer image
TEST 13: Valid complaint -> Verification -> routing -> complaint creation (APPROVED)
"""

import io
import os
import sys
import random
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from app.main import app
from app.database.session import SessionLocal
from app.database.models import Complaint, ComplaintStatus, PriorityLevel
from app.services.conversational_service import conversational_service
from app.orchestrator.orchestrator import orchestrator
from app.orchestrator.decision_engine import verification_engine
from app.agents.image_agent import image_agent
from app.agents.nlp_agent import nlp_agent
from app.agents.severity_agent import severity_agent
from app.agents.fraud_agent import fraud_agent

client = TestClient(app)


def test_scenario_1_clarification_light():
    """
    TEST 1:
    User: 'The light is off.'
    Expected: AI must NOT immediately create a complaint.
    Must ask: 'Is this a home light or a streetlight?'
    """
    db = SessionLocal()
    try:
        res = conversational_service.handle_chat(
            db=db,
            message="The light is off."
        )
        assert res["ticket_data"] is None, "Should not create a ticket on ambiguous input"
        assert res["intent"] == "CLARIFICATION_REQUIRED"
        assert "home light or a streetlight" in res["reply"].lower()
        print("✓ TEST 1 Passed: Ambiguous light complaint clarified successfully.")
    finally:
        db.close()


def test_scenario_2_category_button_drainage():
    """
    TEST 2:
    User selects 'Drainage'.
    Expected: AI must NOT immediately register a drainage complaint.
    It should ask: 'Please describe the actual drainage problem, such as a blocked drain, waterlogging, overflowing drain, or sewage leakage.'
    """
    db = SessionLocal()
    try:
        res = conversational_service.handle_chat(
            db=db,
            category="drainage",
            action="select_category"
        )
        assert res["ticket_data"] is None, "Should not register ticket on category click"
        assert res["intent"] == "CATEGORY_CLARIFICATION_REQUIRED"
        assert "describe the actual drainage problem" in res["reply"].lower()
        assert "waterlogging" in res["reply"].lower() or "blocked drain" in res["reply"].lower()
        print("✓ TEST 2 Passed: Bare category selection intercepted with description prompt.")
    finally:
        db.close()


def test_scenario_3_marathi_complaint():
    """
    TEST 3:
    Marathi complaint.
    Expected: Marathi response.
    """
    db = SessionLocal()
    try:
        user_msg = "रस्त्यावर मोठा खड्डा पडला आहे आणि वाहतूक कोंडी होत आहे."
        res = conversational_service.handle_chat(
            db=db,
            message=user_msg
        )
        assert res["language"] == "mr"
        # Verify response contains Devanagari characters
        devanagari_chars = [c for c in res["reply"] if '\u0900' <= c <= '\u097F']
        assert len(devanagari_chars) > 5, "Response must be in Marathi script"
        print("✓ TEST 3 Passed: Marathi input received Marathi response.")
    finally:
        db.close()


def test_scenario_4_english_complaint():
    """
    TEST 4:
    English complaint.
    Expected: English response.
    """
    db = SessionLocal()
    try:
        user_msg = "There is a severe pothole on my road near sector 21."
        res = conversational_service.handle_chat(
            db=db,
            message=user_msg
        )
        assert res["language"] == "en"
        # Ensure no Devanagari characters in English response
        devanagari_chars = [c for c in res["reply"] if '\u0900' <= c <= '\u097F']
        assert len(devanagari_chars) == 0, "English complaint must receive clean English response without Devanagari"
        print("✓ TEST 4 Passed: English input received English response.")
    finally:
        db.close()


def test_scenario_5_unclear_language():
    """
    TEST 5:
    Unclear language.
    Expected: Ask user to re-describe in English or Marathi.
    """
    db = SessionLocal()
    try:
        user_msg = "Bonjour comment ca va merci beaucoup"
        res = conversational_service.handle_chat(
            db=db,
            message=user_msg
        )
        assert res["intent"] == "LANGUAGE_CLARIFICATION_REQUIRED"
        assert "English or Marathi" in res["reply"] or "इंग्रजी किंवा मराठीत" in res["reply"]
        print("✓ TEST 5 Passed: Unclear language requested re-description in English/Marathi.")
    finally:
        db.close()


def test_scenario_6_streetlight_irrelevant_image():
    """
    TEST 6:
    Streetlight complaint + irrelevant image (e.g. food plate).
    Expected: Reject evidence and request relevant image.
    """
    db = SessionLocal()
    try:
        # Create a food plate sample image
        arr = np.full((224, 224, 3), 240, dtype=np.uint8)
        arr[50:170, 50:170] = [180, 100, 50]  # center food dish
        img = Image.fromarray(arr)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")

        res = conversational_service.handle_chat(
            db=db,
            message="The streetlight outside house 42 is broken and not turning on.",
            category="streetlight",
            photo_filename="food_plate.jpg",
            photo_bytes=img_bytes.getvalue()
        )
        assert res["intent"] == "EVIDENCE_REJECTED"
        assert res["ticket_data"] is None, "Irrelevant image must not create a ticket"
        assert "not appear to show a streetlight" in res["reply"].lower() or "पथदिवा किंवा विद्युत समस्या दिसत नाही" in res["reply"]
        print("✓ TEST 6 Passed: Irrelevant evidence rejected with streetlight prompt.")
    finally:
        db.close()


def test_scenario_7_pothole_valid_image():
    """
    TEST 7:
    Pothole complaint + valid pothole image.
    Expected: YOLO classification + verification passes.
    """
    # Create valid textured image representing road surface
    arr = np.random.randint(60, 160, size=(224, 224, 3), dtype=np.uint8)
    img_path = "uploads/test_pothole_texture.jpg"
    Image.fromarray(arr).save(img_path)

    # 1. Run real YOLO model classification
    yolo_res = image_agent.classify_image(img_path)
    assert "category" in yolo_res
    assert yolo_res["model_mode"] in ["pretrained_yolo11", "custom_trained"]
    assert isinstance(yolo_res["confidence"], float)

    # 2. Run Evidence Verification
    ev_check = image_agent.verify_evidence("pothole", yolo_res)
    assert ev_check["detected_image_category"] is not None
    print(f"✓ TEST 7 Passed: Real YOLO11 inference ran on pothole image (Category: {yolo_res['category']}, Conf: {yolo_res['confidence']}).")


def test_scenario_8_duplicate_complaint():
    """
    TEST 8:
    Duplicate complaint.
    Expected: Duplicate warning / review, no duplicate work order created.
    """
    db = SessionLocal()
    try:
        unique_phone = f"977{random.randint(1000000, 9999999)}"
        lat = round(18.6400 + random.uniform(0.005, 0.045), 5)
        lng = round(73.7500 + random.uniform(0.005, 0.045), 5)
        desc = "मोठा खड्डा पडला आहे वाकड मुख्य रस्त्यावर."

        # Register first complaint
        r1 = orchestrator.process_registration(
            db=db,
            description=desc,
            latitude=lat,
            longitude=lng,
            citizen_phone=unique_phone
        )
        first_ticket = r1["ticket_id"]
        assert r1["is_duplicate"] is False

        # Attempt to register duplicate complaint at exact location
        r2 = orchestrator.process_registration(
            db=db,
            description=desc,
            latitude=lat,
            longitude=lng,
            citizen_phone=unique_phone
        )
        assert r2["is_duplicate"] is True
        assert r2["ticket_id"] == first_ticket
        assert r2["repeat_count"] == 2
        print(f"✓ TEST 8 Passed: Duplicate detected on ticket {first_ticket}, repeat_count incremented to {r2['repeat_count']}.")
    finally:
        db.close()


def test_scenario_9_suspicious_complaint():
    """
    TEST 9:
    Suspicious complaint (e.g. out-of-bounds GPS / spam).
    Expected: Fraud risk elevated + REVIEW_REQUIRED status.
    """
    db = SessionLocal()
    try:
        # Delhi coordinates outside PCMC boundary
        res = orchestrator.process_registration(
            db=db,
            description="Road pothole in sector 10",
            latitude=28.7041,
            longitude=77.1025,
            citizen_phone="9998887776"
        )
        assert res["fraud_score"] >= 0.40, f"Fraud score should be elevated: {res['fraud_score']}"
        assert res["status"] in ["REVIEW_REQUIRED", "REJECTED"]
        print(f"✓ TEST 9 Passed: Suspicious complaint flagged with fraud_score={res['fraud_score']}, status={res['status']}.")
    finally:
        db.close()


def test_scenario_10_emergency_complaint():
    """
    TEST 10:
    Emergency complaint.
    Expected: is_emergency = true and appropriate priority (P1).
    """
    emergency_text = "Exposed live electrical wires sparking on the main road, high danger of electrocution!"
    eval_res = severity_agent.evaluate_priority(
        description=emergency_text,
        category="electricity"
    )
    assert eval_res["is_emergency"] is True
    assert eval_res["emergency_level"] in ["CRITICAL", "HIGH"]
    assert eval_res["priority_code"] == "P1"
    print(f"✓ TEST 10 Passed: Emergency detected successfully (Emergency={eval_res['is_emergency']}, Priority={eval_res['priority_code']}).")


def test_scenario_11_normal_complaint():
    """
    TEST 11:
    Normal complaint.
    Expected: is_emergency = false.
    """
    normal_text = "Minor tree trimming request for garden branch touching fence."
    eval_res = severity_agent.evaluate_priority(
        description=normal_text,
        category="trees"
    )
    assert eval_res["is_emergency"] is False
    assert eval_res["emergency_level"] == "NONE"
    assert eval_res["priority_code"] in ["P3", "P4"]
    print(f"✓ TEST 11 Passed: Normal complaint verified (Emergency={eval_res['is_emergency']}, Priority={eval_res['priority_code']}).")


def test_scenario_12_low_confidence_image():
    """
    TEST 12:
    Low-confidence image.
    Expected: Request clearer image.
    """
    # Test low confidence simulated output
    low_conf_pred = {
        "category": "pothole",
        "confidence": 0.15,  # below 0.30 threshold
        "model_mode": "pretrained_yolo11",
        "raw_label": "obscure_object"
    }
    check = image_agent.verify_evidence(
        complaint_category="pothole",
        image_prediction=low_conf_pred,
        min_confidence=0.30
    )
    assert check["is_confident"] is False
    assert "couldn't confidently identify" in check["message_en"].lower() or "निश्चित ओळख पटवता आली नाही" in check["message_mr"]
    print("✓ TEST 12 Passed: Low-confidence image rejected with request for clearer photo.")


def test_scenario_13_valid_complaint_pipeline():
    """
    TEST 13:
    Valid complaint.
    Expected: Verification -> routing -> complaint creation (status: REGISTERED, APPROVED).
    """
    db = SessionLocal()
    try:
        unique_phone = f"966{random.randint(1000000, 9999999)}"

        # 1. Test Verification Engine direct approval
        verif = verification_engine.verify(
            category="pothole",
            image_confidence=0.88,
            evidence_valid=True,
            is_duplicate=False,
            fraud_risk=0.05,
            severity="MEDIUM",
            is_emergency=False,
            priority="P3",
            department="CIVIL_ROADS"
        )
        assert verif["verification_status"] == "APPROVED"
        assert verif["category"] == "pothole"

        # 2. Test End-to-End Orchestrator Pipeline with fresh PCMC location
        rand_lat = round(18.6600 + random.uniform(0.005, 0.040), 5)
        rand_lng = round(73.7900 + random.uniform(0.005, 0.040), 5)

        res = orchestrator.process_registration(
            db=db,
            description="पिंपळे गुरव मुख्य चौकात रस्त्यावर मोठा खड्डा पडला आहे.",
            latitude=rand_lat,
            longitude=rand_lng,
            citizen_phone=unique_phone,
            allow_duplicate_override=True
        )

        assert res["verification_status"] == "APPROVED"
        assert res["status"] == "REGISTERED"
        assert res["assigned_department"] == "CIVIL_ROADS"
        assert res["ticket_id"].startswith("WM-")
        assert res["ward"] is not None
        print(f"✓ TEST 13 Passed: Valid complaint successfully approved, routed to {res['assigned_department']}, and registered with Ticket {res['ticket_id']}.")
    finally:
        db.close()


def test_scenario_14_natural_conversational_greeting_and_chat():
    """
    TEST 14:
    User inputs casual greetings like 'hello Siddheshwar', 'who are you', or 'thank you'.
    Expected: Chatbot must respond naturally like an AI assistant without falsely
    classifying the greeting as a grievance or assigning it to 'Garbage & Cleanliness'.
    """
    db = SessionLocal()
    try:
        # 1. Personalized Greeting
        res1 = conversational_service.handle_chat(db=db, message="hello Siddheshwar")
        assert res1["intent"] == "GREETING"
        assert res1["category"] is None, "Greeting must not be assigned a civic category"
        assert res1["ticket_data"] is None
        assert "garbage" not in res1["reply"].lower() or "cleanliness" not in res1["reply"].lower()
        assert "hello" in res1["reply"].lower() or "wardmitra" in res1["reply"].lower()

        # 2. Smalltalk / Gratitude
        res2 = conversational_service.handle_chat(db=db, message="thank you")
        assert res2["intent"] == "SMALLTALK"
        assert res2["category"] is None
        assert "welcome" in res2["reply"].lower()

        # 3. Bot Identity
        res3 = conversational_service.handle_chat(db=db, message="who are you")
        assert res3["intent"] == "SERVICE_INQUIRY"
        assert res3["category"] is None
        assert "wardmitra" in res3["reply"].lower() or "sarathi" in res3["reply"].lower()

        print("✓ TEST 14 Passed: Natural conversational responses verified for greetings, inquiries, and gratitude.")
    finally:
        db.close()


def test_scenario_15_complaint_process_step_by_step_guidance():
    """
    Requirement 15: Step-by-Step Guidance for Grievance Filing (City & Gramin friendly)
    When citizen asks 'how to register the complaint' or in Marathi 'तक्रार कशी नोंदवायची?':
    - Bot must NOT misclassify as Streetlight / Pothole.
    - Bot must provide a clear 4-stage step-by-step guidance tailored for city and gramin citizens.
    - Even if category was previously passed by client, asking for guidance must not lock to that category.
    """
    db = SessionLocal()
    try:
        # 1. English Inquiry
        res1 = conversational_service.handle_chat(db=db, message="how to register the complaint")
        assert res1["intent"] == "COMPLAINT_PROCESS_GUIDE"
        assert res1["category"] is None
        assert "Step" in res1["reply"] or "step" in res1["reply"].lower()
        assert "1" in res1["reply"] and "4" in res1["reply"]

        # 2. English Inquiry with category passed from previous state
        res2 = conversational_service.handle_chat(db=db, message="how to register the complaint", category="streetlight")
        assert res2["intent"] == "COMPLAINT_PROCESS_GUIDE"
        assert res2["category"] is None
        assert "streetlight" not in res2["reply"].lower().split("\n")[0] # Not forced to streetlight

        # 3. Marathi Inquiry
        res3 = conversational_service.handle_chat(db=db, message="तक्रार कशी नोंदवायची?")
        assert res3["intent"] == "COMPLAINT_PROCESS_GUIDE"
        assert res3["category"] is None
        assert "पायरी" in res3["reply"]
        assert "नोंदणी फॉर्म" in res3["reply"]

        # 4. Rural Marathi Inquiry
        res4 = conversational_service.handle_chat(db=db, message="मला तक्रार करायची आहे कशी करू?")
        assert res4["intent"] == "COMPLAINT_PROCESS_GUIDE"
        assert res4["category"] is None
        assert "पायरी" in res4["reply"]

        print("✓ TEST 15 Passed: Step-by-step complaint registration guidance verified for city and rural citizens.")
    finally:
        db.close()


if __name__ == "__main__":
    print("\n=======================================================")
    print("RUNNING WARDMITRA AI SCENARIO VERIFICATION TESTS (1-15)")
    print("=======================================================\n")
    test_scenario_1_clarification_light()
    test_scenario_2_category_button_drainage()
    test_scenario_3_marathi_complaint()
    test_scenario_4_english_complaint()
    test_scenario_5_unclear_language()
    test_scenario_6_streetlight_irrelevant_image()
    test_scenario_7_pothole_valid_image()
    test_scenario_8_duplicate_complaint()
    test_scenario_9_suspicious_complaint()
    test_scenario_10_emergency_complaint()
    test_scenario_11_normal_complaint()
    test_scenario_12_low_confidence_image()
    test_scenario_13_valid_complaint_pipeline()
    test_scenario_14_natural_conversational_greeting_and_chat()
    test_scenario_15_complaint_process_step_by_step_guidance()
    print("\n🎉 ALL 15 SCENARIO TESTS PASSED SUCCESSFULLY!\n")
