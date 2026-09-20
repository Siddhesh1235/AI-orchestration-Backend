"""
Automated Test Suite for Image Understanding & Context-Aware Follow-Up Questions in WardMitra AI.

Verifies the 8 mandated test scenarios:
TEST 1: User uploads streetlight image -> Identifies streetlight & asks relevant follow-up.
TEST 2: User says 'streetlight is OFF' + uploads streetlight image -> Does not ask ON/OFF again.
TEST 3: User uploads pothole image -> Identifies pothole/road issue & asks next relevant question.
TEST 4: User uploads garbage image -> Identifies garbage issue & continues complaint flow.
TEST 5: User uploads an unrelated image (food plate / pet) -> Politely asks for civic photo.
TEST 6: User uploads a blurry / low-texture image -> Rejects with request for clearer photo.
TEST 7: Image contains multiple possible civic issues -> Asks citizen which issue to report.
TEST 8: User provides complete information before photo -> Uses photo as evidence without duplicate questions.
"""

import io
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.session import SessionLocal
from app.services.conversational_service import conversational_service
from app.services.image_understanding_service import image_understanding_service, ImageAnalysisResult


def _create_test_image_bytes(pattern: str = "random", width: int = 224, height: int = 224) -> bytes:
    """Helper to generate in-memory synthetic images for testing."""
    if pattern == "uniform_blur":
        # Flat gray image with zero edge gradients (variance < 35.0)
        arr = np.full((height, width, 3), 128, dtype=np.uint8)
    elif pattern == "food_plate":
        # White background with round dish center
        arr = np.full((height, width, 3), 240, dtype=np.uint8)
        arr[50:170, 50:170] = [180, 100, 50]
    elif pattern == "night_streetlight_on":
        # Dark night background with bright illuminated lamp in top half
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[20:45, 100:125] = [255, 255, 255]  # bright hotspot
        arr[45:180, 110:115] = [120, 120, 120]  # pole
    elif pattern == "streetlight_off":
        # Shaded dusk scene with dark unlit lamp fixture
        arr = np.random.randint(25, 60, size=(height, width, 3), dtype=np.uint8)
        arr[45:180, 105:120] = np.random.randint(90, 150, size=(135, 15, 3), dtype=np.uint8)  # dark pole
        arr[20:45, 95:130] = np.random.randint(50, 80, size=(25, 35, 3), dtype=np.uint8)   # dark unlit lamp
    elif pattern == "pothole_texture":
        # Asphalt/road texture
        arr = np.random.randint(60, 160, size=(height, width, 3), dtype=np.uint8)
        arr[80:150, 80:150] = np.random.randint(20, 50, size=(70, 70, 3), dtype=np.uint8)  # dark depression
    else:
        # Standard civic texture
        arr = np.random.randint(50, 200, size=(height, width, 3), dtype=np.uint8)

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_scenario_1_streetlight_image_followup():
    """
    TEST 1:
    User uploads streetlight image without prior context.
    Expected: Bot identifies streetlight and asks a relevant follow-up question.
    """
    db = SessionLocal()
    try:
        img_bytes = _create_test_image_bytes(pattern="streetlight_off")
        res = conversational_service.handle_chat(
            db=db,
            message=None,
            photo_filename="streetlight_pole_sample.jpg",
            photo_bytes=img_bytes
        )
        assert res["intent"] == "IMAGE_FOLLOWUP_REQUIRED"
        assert res["category"] == "streetlight"
        assert res["ticket_data"] is None, "Should not prematurely register ticket"
        assert "image_analysis" in res
        # Must ask relevant streetlight follow-up
        reply_lower = res["reply"].lower()
        assert any(w in reply_lower for w in ["streetlight", "light", "on", "off", "location", "landmark", "pole", "पथदिवा", "दिवा", "चालू", "बंद"])
        print("✓ TEST 1 Passed: Streetlight photo identified, asked context-aware follow-up.")
    finally:
        db.close()


def test_scenario_2_streetlight_already_stated_off():
    """
    TEST 2:
    User says 'streetlight is OFF' and uploads streetlight image.
    Expected: Bot does NOT ask whether it is ON/OFF again. Asks for next missing detail (pole/location).
    """
    db = SessionLocal()
    try:
        img_bytes = _create_test_image_bytes(pattern="streetlight_off")
        res = conversational_service.handle_chat(
            db=db,
            message="The streetlight is OFF outside my building.",
            photo_filename="streetlight_damaged.jpg",
            photo_bytes=img_bytes
        )
        assert res["intent"] == "IMAGE_FOLLOWUP_REQUIRED"
        assert res["category"] == "streetlight"
        assert res["ticket_data"] is None
        # Bot must NOT ask "Is it ON or OFF?"
        reply_lower = res["reply"].lower()
        assert "is the streetlight currently on or off" not in reply_lower
        assert "चालू आहे की बंद" not in reply_lower
        # Instead asks for landmark/pole or damages
        assert any(w in reply_lower for w in ["pole", "location", "landmark", "wiring", "damage", "खूण", "नंबर", "तारा"])
        print("✓ TEST 2 Passed: Suppressed redundant ON/OFF question when user already stated light is OFF.")
    finally:
        db.close()


def test_scenario_3_pothole_image():
    """
    TEST 3:
    User uploads pothole image.
    Expected: Bot identifies pothole/road issue and asks the next relevant question.
    """
    db = SessionLocal()
    try:
        img_bytes = _create_test_image_bytes(pattern="pothole_texture")
        res = conversational_service.handle_chat(
            db=db,
            message=None,
            photo_filename="road_pothole_crack.jpg",
            photo_bytes=img_bytes
        )
        assert res["intent"] == "IMAGE_FOLLOWUP_REQUIRED"
        assert res["category"] == "pothole"
        assert res["ticket_data"] is None
        reply_lower = res["reply"].lower()
        assert any(w in reply_lower for w in ["pothole", "road", "location", "landmark", "खड्डा", "रस्ता", "ठिकाण"])
        print("✓ TEST 3 Passed: Pothole photo identified and prompted for road/location details.")
    finally:
        db.close()


def test_scenario_4_garbage_image():
    """
    TEST 4:
    User uploads garbage image.
    Expected: Bot identifies garbage-related issue and continues the complaint flow.
    """
    db = SessionLocal()
    try:
        img_bytes = _create_test_image_bytes(pattern="random")
        res = conversational_service.handle_chat(
            db=db,
            message=None,
            photo_filename="garbage_waste_dump.jpg",
            photo_bytes=img_bytes
        )
        assert res["intent"] == "IMAGE_FOLLOWUP_REQUIRED"
        assert res["category"] == "garbage"
        assert res["ticket_data"] is None
        reply_lower = res["reply"].lower()
        assert any(w in reply_lower for w in ["garbage", "waste", "located", "कचरा", "ठिकाण", "साचला"])
        print("✓ TEST 4 Passed: Garbage photo identified and prompted for waste location/duration.")
    finally:
        db.close()


def test_scenario_5_unrelated_image():
    """
    TEST 5:
    User uploads an unrelated image (e.g. food plate, pet, tableware).
    Expected: Bot politely asks for a relevant civic-issue image.
    """
    db = SessionLocal()
    try:
        img_bytes = _create_test_image_bytes(pattern="food_plate")
        res = conversational_service.handle_chat(
            db=db,
            message=None,
            photo_filename="food_plate_lunch.jpg",
            photo_bytes=img_bytes
        )
        assert res["intent"] == "EVIDENCE_REJECTED"
        assert res["ticket_data"] is None, "Irrelevant image must not create a ticket"
        reply_lower = res["reply"].lower()
        assert any(w in reply_lower for w in ["couldn't identify", "not appear", "civic issue", "नागरी समस्या", "स्पष्टपणे आढळली नाही"])
        print("✓ TEST 5 Passed: Unrelated food plate photo politely rejected.")
    finally:
        db.close()


def test_scenario_6_blurry_image():
    """
    TEST 6:
    User uploads a blurry image (very low Laplacian variance).
    Expected: Bot indicates that the issue cannot be confidently identified and asks for a clearer image.
    """
    db = SessionLocal()
    try:
        img_bytes = _create_test_image_bytes(pattern="uniform_blur")
        res = conversational_service.handle_chat(
            db=db,
            message=None,
            photo_filename="blurry_pothole.jpg",
            photo_bytes=img_bytes
        )
        assert res["intent"] == "LOW_CONFIDENCE_IMAGE"
        assert res["ticket_data"] is None
        reply_lower = res["reply"].lower()
        assert any(w in reply_lower for w in ["blurry", "confidently identify", "clearer photo", "अस्पष्ट", "स्पष्ट छायाचित्र"])
        print("✓ TEST 6 Passed: Blurry image flagged and clearer photo requested.")
    finally:
        db.close()


def test_scenario_7_multiple_issues_image():
    """
    TEST 7:
    Image contains multiple possible civic issues (e.g. multi pothole + garbage).
    Expected: Bot asks the user which issue they want to report.
    """
    db = SessionLocal()
    try:
        img_bytes = _create_test_image_bytes(pattern="random")
        res = conversational_service.handle_chat(
            db=db,
            message=None,
            photo_filename="multi_pothole_and_garbage.jpg",
            photo_bytes=img_bytes
        )
        assert res["intent"] == "MULTIPLE_ISSUES_DETECTED"
        assert res["ticket_data"] is None
        assert res["image_analysis"]["has_multiple_issues"] is True
        reply_lower = res["reply"].lower()
        assert any(w in reply_lower for w in ["more than one", "which issue", "एकापेक्षा जास्त", "कोणती समस्या"])
        print("✓ TEST 7 Passed: Multi-issue scene detected and user asked to select primary grievance.")
    finally:
        db.close()


def test_scenario_8_complete_info_before_photo_registers_complaint():
    """
    TEST 8:
    User provides all required information (category, issue, location/ward) before uploading image.
    Expected: Bot uses the image as evidence, does not repeat already answered questions,
              and advances directly to ticket registration.
    """
    db = SessionLocal()
    try:
        img_bytes = _create_test_image_bytes(pattern="pothole_texture")
        import random
        unique_phone = f"944{random.randint(1000000, 9999999)}"
        lat = round(18.6100 + random.uniform(0.01, 0.05), 5)
        lng = round(73.7500 + random.uniform(0.01, 0.05), 5)
        complete_text = f"There is a severe deep pothole on Wakad road near Akurdi chowk section {random.randint(1, 999)} causing traffic hazards."
        res = conversational_service.handle_chat(
            db=db,
            message=complete_text,
            latitude=lat,
            longitude=lng,
            citizen_phone=unique_phone,
            action="register_new",
            ward_number=25,
            photo_filename="pothole_evidence.jpg",
            photo_bytes=img_bytes
        )
        assert res["intent"] == "COMPLAINT_REGISTERED"
        assert res["ticket_data"] is not None
        assert "ticket_id" in res["ticket_data"]
        assert res["ticket_data"]["status"] == "REGISTERED"
        assert any(w in res["reply"] for w in ["WM-", "नोंदवली गेली आहे", "successfully registered"])
        print(f"✓ TEST 8 Passed: Complete info smoothly registered ticket {res['ticket_data']['ticket_id']} with verified photo.")
    finally:
        db.close()


def test_unit_image_understanding_service_marathi_flow():
    """
    Unit test: Validates Marathi localized follow-up generation for drainage and water leakage.
    """
    # 1. Drainage analysis
    analysis_drainage = ImageAnalysisResult(
        category="drainage",
        confidence=0.92,
        issue="blocked_drain",
        image_quality="good"
    )
    followup_drainage = image_understanding_service.generate_followup(
        analysis=analysis_drainage,
        user_message="गटार तुंबले आहे",
        context=None,
        lang="mr"
    )
    assert followup_drainage["all_info_known"] is False
    assert "ड्रेनेज" in followup_drainage["followup_question"] or "गटर" in followup_drainage["followup_question"]

    # 2. Water leakage analysis
    analysis_water = ImageAnalysisResult(
        category="pipeline_water_leakage",
        confidence=0.94,
        issue="pipeline_leak",
        image_quality="good"
    )
    followup_water = image_understanding_service.generate_followup(
        analysis=analysis_water,
        user_message="पाणी वाहत आहे",
        context={"latitude": 18.62, "longitude": 73.78, "ward_number": 12},
        lang="mr"
    )
    assert followup_water["all_info_known"] is False
    assert "पाणी" in followup_water["followup_question"] or "गळती" in followup_water["followup_question"]
    print("✓ Unit Test Passed: Marathi localized follow-ups generated correctly for drainage & water supply.")
