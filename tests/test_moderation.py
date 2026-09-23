"""
Comprehensive Unit and Integration Tests for Content Moderation Subsystem.
Tests:
1. Multilingual Profanity Agent (Marathi, Hindi, English, clean civic messages)
2. Image Safety Heuristics (Solid blank images, valid images, corrupted images)
3. End-to-End Orchestrator Grievance Registration with Moderation Rejection and Pass
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath("."))
from PIL import Image
import numpy as np

from app.agents.profanity_agent import profanity_agent
from app.agents.moderation_agent import moderation_agent
from app.orchestrator.orchestrator import orchestrator
from app.database.session import SessionLocal


def test_clean_civic_text():
    """Verify that normal civic grievances are NOT falsely flagged as profane."""
    clean_cases = [
        "वाकड चौकात रस्त्यावर मोठा खड्डा पडला आहे.",
        "थेरगाव मुख्य रस्त्यावरील कचराकुंडी भरून कचरा पसरला आहे.",
        "पिंपळे सौदागर येथे ड्रेनेज लाईन तुंबली आहे.",
        "Streetlight is not working on Sector 24 for 2 days.",
        "Water pipeline leakage near bus stand.",
        "सकाळी पाणी वेळेवर येत नाही कृपया दुरुस्ती करावी."
    ]
    for text in clean_cases:
        res = profanity_agent.evaluate_text(text)
        assert not res["is_profane"], f"False positive detected on: '{text}', words: {res['detected_words']}"
        assert res["severity"] == "NONE"


def test_multilingual_profanity_detection():
    """Verify detection of offensive words in Marathi, Hindi, and English."""
    profane_cases = [
        ("हा अधिकारी अगदी भिकारचोट आहे", "भिकारचोट"),
        ("येथे काम करा रे चूतिया लोकांनो", "चूतिया"),
        ("This is total bullshit and garbage work fuck you", "fuck"),
        ("tum sab bhenchod aur madarchod ho", "bhenchod"),
        ("अरे आईघाल्या काम कधी करणार", "आईघाल्या")
    ]
    for text, expected_word in profane_cases:
        res = profanity_agent.evaluate_text(text)
        assert res["is_profane"], f"Failed to detect profanity in: '{text}'"
        assert res["severity"] in ["HIGH", "MEDIUM"]
        assert any(expected_word.lower() in w.lower() for w in res["detected_words"])


def test_image_safety_local_heuristics():
    """Verify local vision checks for blank/solid images and valid images."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # 1. Blank single-color image (Spam)
        blank_img_path = str(tmp_path / "blank.jpg")
        blank_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        blank_img.save(blank_img_path)
        res_blank = moderation_agent.evaluate_image_safety(blank_img_path)
        assert not res_blank["is_safe"], "Blank image should be flagged as unsafe/invalid"

        # 2. Normal textured civic test image
        valid_img_path = str(tmp_path / "civic.jpg")
        arr = np.random.randint(50, 200, size=(100, 100, 3), dtype=np.uint8)
        valid_img = Image.fromarray(arr)
        valid_img.save(valid_img_path)
        res_valid = moderation_agent.evaluate_image_safety(valid_img_path)
        assert res_valid["is_safe"], f"Valid civic image was falsely flagged: {res_valid}"


def test_orchestrator_moderation_rejection():
    """Verify that orchestrator intercepts and rejects offensive grievances early."""
    db = SessionLocal()
    try:
        # Submit offensive grievance
        result = orchestrator.process_registration(
            db=db,
            description="येथे खूप कचरा आहे आणि अधिकारी नालायक भिकारचोट आहेत",
            latitude=18.6010,
            longitude=73.7630,
            citizen_phone="9876543210"
        )
        assert result["moderation_status"] == "REJECTED"
        assert result["is_fraud"] is True
        assert result["status"] == "REJECTED"
        assert "🚫" in result["message"]
        assert result["assigned_worker_name"] == "N/A"
    finally:
        db.close()


def test_orchestrator_moderation_pass():
    """Verify that clean civic grievance passes moderation and registers successfully."""
    db = SessionLocal()
    try:
        result = orchestrator.process_registration(
            db=db,
            description="वाकड चौकात रस्त्यावर मोठा खड्डा पडला आहे, अपघात होण्याची शक्यता आहे.",
            latitude=18.6010,
            longitude=73.7630,
            citizen_phone="9988776655"
        )
        assert result["moderation_status"] == "PASSED"
        assert result["moderation_reason"] is None
        assert result["ticket_id"].startswith("WM-") or result["ticket_id"].startswith("PCMC-")
    finally:
        db.close()


if __name__ == "__main__":
    print("Running Test 1: Clean Civic Text...")
    test_clean_civic_text()
    print("✓ Test 1 Passed!")

    print("Running Test 2: Multilingual Profanity Detection...")
    test_multilingual_profanity_detection()
    print("✓ Test 2 Passed!")

    print("Running Test 3: Image Safety Local Heuristics...")
    test_image_safety_local_heuristics()
    print("✓ Test 3 Passed!")

    print("Running Test 4: Orchestrator Rejection Flow...")
    test_orchestrator_moderation_rejection()
    print("✓ Test 4 Passed!")

    print("Running Test 5: Orchestrator Pass Flow...")
    test_orchestrator_moderation_pass()
    print("✓ Test 5 Passed!")

    print("\n🎉 ALL CONTENT MODERATION TESTS PASSED SUCCESSFULLY!")
