"""
End-to-End Automated Verification Test Suite for PCMC Sarathi AI (Section 3.4.2).
Tests Registration, Image & Text Classification, Tracking, Status Updates, and Citizen Feedback.
"""

import io
import sys
import random
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify health telemetry."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "Ward Mitra" in data["service"]
    print("✓ Health Check Passed:", data)


def test_full_complaint_lifecycle():
    """Verify end-to-end Grievance Registration, Tracking, Update, and Closure."""
    # 1. Create a dummy image in-memory
    img = Image.new("RGB", (100, 100), color="gray")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # 2. Register complaint with photo and Wakad coordinates
    files = {"photo": ("pothole_sample.jpg", img_bytes.getvalue(), "image/jpeg")}
    unique_phone = f"987{random.randint(1000000, 9999999)}"
    data = {
        "description": "वाकड चौकात रस्त्यावर मोठा खड्डा पडला आहे आणि वाहतूक कोंडी होत आहे.",
        "latitude": 18.6010,
        "longitude": 73.7630,
        "citizen_phone": unique_phone
    }

    reg_resp = client.post("/api/v1/complaints/register", data=data, files=files)
    assert reg_resp.status_code == 201
    reg_data = reg_resp.json()
    ticket_id = reg_data["ticket_id"]
    print(f"✓ Registered Grievance: Ticket ID = {ticket_id}")
    assert reg_data["status"] == "REGISTERED"
    assert reg_data["ward"] is not None
    assert reg_data["assigned_department"] is not None

    # 3. Check status
    status_resp = client.get(f"/api/v1/complaints/{ticket_id}/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] == "REGISTERED"
    assert len(status_data["timeline"]) == 5
    print("✓ Grievance Tracking Timeline Verified.")

    # 4. Officer updates status to IN_PROGRESS
    update1 = client.patch(
        f"/api/v1/complaints/{ticket_id}/status",
        data={"new_status": "IN_PROGRESS", "officer_name": "Er. P. Jadhav", "remarks": "Team on site."}
    )
    assert update1.status_code == 200
    assert update1.json()["status"] == "IN_PROGRESS"
    print("✓ Officer Status Update -> IN_PROGRESS Verified.")

    # 5. Officer marks RESOLVED with proof photo
    res_img = Image.new("RGB", (100, 100), color="green")
    res_bytes = io.BytesIO()
    res_img.save(res_bytes, format="JPEG")
    res_bytes.seek(0)
    files_res = {"resolution_photo": ("resolved_proof.jpg", res_bytes.getvalue(), "image/jpeg")}

    update2 = client.patch(
        f"/api/v1/complaints/{ticket_id}/status",
        data={"new_status": "RESOLVED", "officer_name": "Er. P. Jadhav", "remarks": "Road repaired with asphalt."},
        files=files_res
    )
    assert update2.status_code == 200
    assert update2.json()["status"] == "RESOLVED"
    print("✓ Officer Status Update -> RESOLVED with Photo Proof Verified.")

    # 6. Citizen Submits 5-Star Feedback and Confirms Resolution
    feedback_payload = {
        "rating": 5,
        "comments": "खूप छान आणि जलद काम केले. धन्यवाद महापालिका!",
        "confirmed_resolved": True
    }
    fb_resp = client.post(f"/api/v1/complaints/{ticket_id}/feedback", json=feedback_payload)
    assert fb_resp.status_code == 200
    assert fb_resp.json()["status"] == "CLOSED"
    print(f"✓ Citizen Feedback & Closure Verified: Status = {fb_resp.json()['status']}")


def test_priority_and_3level_escalation():
    """
    Verify Priority Grading (Low/Medium/High) and 3-Level Municipal Escalation:
    Level 1 (Ward Worker) -> Level 2 (Supervisor) -> Level 3 (HOD)
    """
    print("\n--- Testing Priority & 3-Level Municipal Escalation Hierarchy ---")
    
    # 1. Register a HIGH-priority grievance (exposed wire/danger in PCMC)
    unique_phone = f"982{random.randint(1000000, 9999999)}"
    lat_val = round(18.6300 + random.uniform(0.005, 0.040), 4)
    lng_val = round(73.7800 + random.uniform(0.005, 0.040), 4)
    data = {
        "description": "वाकड चौकात उघडी डीपी आणि विजेची तार लटकत आहे, अतिशय धोकादायक आहे.",
        "latitude": lat_val,
        "longitude": lng_val,
        "citizen_phone": unique_phone,
        "priority": "HIGH"
    }
    reg_resp = client.post("/api/v1/complaints/register", data=data)
    assert reg_resp.status_code == 201
    reg_data = reg_resp.json()
    ticket_id = reg_data["ticket_id"]
    
    print(f"✓ Registered Complaint {ticket_id} with Priority: {reg_data['priority']}")
    assert reg_data["priority"] == "HIGH"
    assert reg_data["escalation_level"] == "LEVEL_1_WORKER"
    assert reg_data["assigned_worker_name"] is not None
    print(f"✓ Level 1 (Ward Worker) Initially Assigned: {reg_data['assigned_worker_name']}")

    # 2. Check Status and Escalation Chain
    status_resp = client.get(f"/api/v1/complaints/{ticket_id}/status")
    assert status_resp.status_code == 200
    st_data = status_resp.json()
    assert st_data["escalation_level"] == "LEVEL_1_WORKER"
    assert "level_1_worker" in st_data["escalation_chain"]
    assert "level_2_supervisor" in st_data["escalation_chain"]
    assert "level_3_hod" in st_data["escalation_chain"]
    print("✓ Escalation Chain Profile retrieved:", {
        "Level 1": st_data["escalation_chain"]["level_1_worker"]["name"],
        "Level 2": st_data["escalation_chain"]["level_2_supervisor"]["name"],
        "Level 3": st_data["escalation_chain"]["level_3_hod"]["name"]
    })

    # 3. Simulate Level 1 Worker did not take action -> Escalate to Level 2 (Supervisor)
    esc1_resp = client.post(
        f"/api/v1/complaints/{ticket_id}/escalate",
        json={"reason": "क्षेत्रीय कामगाराने ४ तासांत दखल न घेतल्याने वर्ग केले."}
    )
    assert esc1_resp.status_code == 200
    esc1_data = esc1_resp.json()
    assert esc1_data["escalated"] is True
    assert esc1_data["to_level"] == "LEVEL_2_SUPERVISOR"
    print(f"✓ Escalated to Level 2 (Supervisor): {esc1_data['current_officer']} ({esc1_data['current_contact']})")

    # Verify status reflects Level 2
    st2_resp = client.get(f"/api/v1/complaints/{ticket_id}/status")
    assert st2_resp.json()["escalation_level"] == "LEVEL_2_SUPERVISOR"

    # 4. Simulate Level 2 Supervisor did not take action -> Escalate to Level 3 (HOD)
    esc2_resp = client.post(
        f"/api/v1/complaints/{ticket_id}/escalate",
        json={"reason": "प्रभाग अधिकाऱ्यांनी वेळेत कार्यवाही न केल्याने थेट HOD कडे वर्ग केले."}
    )
    assert esc2_resp.status_code == 200
    esc2_data = esc2_resp.json()
    assert esc2_data["escalated"] is True
    assert esc2_data["to_level"] == "LEVEL_3_HOD"
    print(f"✓ Escalated to Level 3 (HOD): {esc2_data['current_officer']} ({esc2_data['current_contact']})")

    # Verify status reflects Level 3
    st3_resp = client.get(f"/api/v1/complaints/{ticket_id}/status")
    assert st3_resp.json()["escalation_level"] == "LEVEL_3_HOD"

    # 5. Check automated batch escalation checker endpoint
    check_resp = client.post("/api/v1/complaints/check-escalations")
    assert check_resp.status_code == 200
    print(f"✓ Automated SLA Escalation Checker Passed: Checked {check_resp.json()['checked_count']} complaints.")

    # 6. Close complaint cleanly
    client.patch(f"/api/v1/complaints/{ticket_id}/status", data={"new_status": "RESOLVED", "officer_name": "Er. HOD", "remarks": "Resolved."})
    client.post(f"/api/v1/complaints/{ticket_id}/feedback", json={"rating": 5, "confirmed_resolved": True})


def test_fraud_and_spam_detection():
    """
    Verify Fake / Fraud / Spam Complaint Detection (Gibberish and Out-of-bounds GPS).
    """
    print("\n--- Testing Fake / Fraud Complaint Detection ---")

    # 1. Gibberish text
    data_spam = {
        "description": "asdfghjk",
        "latitude": 18.6010,
        "longitude": 73.7630,
        "citizen_phone": "9999911111"
    }
    resp1 = client.post("/api/v1/complaints/register", data=data_spam)
    assert resp1.status_code == 201
    d1 = resp1.json()
    assert d1["is_fraud"] is True
    print(f"✓ Gibberish Text detected as fraud: {d1['fraud_reason']}")

    # 2. Out of bounds GPS (Delhi coordinates: 28.7041, 77.1025)
    data_oob = {
        "description": "रस्त्यावर मोठा खड्डा पडला आहे, कृपया दुरुस्त करा.",
        "latitude": 28.7041,
        "longitude": 77.1025,
        "citizen_phone": "9999922222"
    }
    resp2 = client.post("/api/v1/complaints/register", data=data_oob)
    assert resp2.status_code == 201
    d2 = resp2.json()
    assert d2["is_fraud"] is True
    print(f"✓ Out-of-bounds GPS detected as fraud: {d2['fraud_reason']}")


def test_duplicate_and_repeat_count_increment():
    """
    Verify User Requirement:
    If a complaint already exists (or was resolved and re-reported),
    do NOT create a new work order; simply increment the repeat_count in the database!
    """
    print("\n--- Testing Repeating Complaint Count Increment ---")

    # 1. Register initial drainage complaint in Ward 25 (Wakad)
    unique_phone = f"989{random.randint(1000000, 9999999)}"
    lat_offset = round(18.6050 + random.uniform(0.001, 0.009), 4)
    lng_offset = round(73.7650 + random.uniform(0.001, 0.009), 4)
    data = {
        "description": "वाकड चौकात गटार तुंबले असून सांडपाणी रस्त्यावर पसरले आहे.",
        "latitude": lat_offset,
        "longitude": lng_offset,
        "citizen_phone": unique_phone
    }
    resp1 = client.post("/api/v1/complaints/register", data=data)
    assert resp1.status_code == 201
    d1 = resp1.json()
    original_ticket = d1["ticket_id"]
    assert d1["repeat_count"] == 1
    assert d1["is_duplicate"] is False
    print(f"✓ Initial Complaint Registered: Ticket = {original_ticket} (Count = {d1['repeat_count']})")

    # 2. Citizen (or neighbor) submits the same complaint again at same location
    resp2 = client.post("/api/v1/complaints/register", data=data)
    assert resp2.status_code == 201
    d2 = resp2.json()
    
    # Must return the SAME ticket ID and increment count!
    assert d2["ticket_id"] == original_ticket
    assert d2["is_duplicate"] is True
    assert d2["repeat_count"] == 2
    print(f"✓ Repeat Complaint Detected: Same Ticket = {d2['ticket_id']} -> Count incremented to {d2['repeat_count']} (No duplicate created)")

    # 3. Mark the complaint as RESOLVED by officer
    client.patch(
        f"/api/v1/complaints/{original_ticket}/status",
        data={"new_status": "RESOLVED", "officer_name": "Er. D. Jagtap", "remarks": "Drainage unblocked."}
    )
    st_res = client.get(f"/api/v1/complaints/{original_ticket}/status").json()
    assert st_res["status"] == "RESOLVED"
    print("✓ Officer marked complaint as RESOLVED.")

    # 4. User reports again that issue is still there/recurring
    resp3 = client.post("/api/v1/complaints/register", data=data)
    assert resp3.status_code == 201
    d3 = resp3.json()
    assert d3["ticket_id"] == original_ticket
    assert d3["repeat_count"] == 3
    assert d3["status"] == "IN_PROGRESS"  # Re-opened
    print(f"✓ Re-reported Resolved Complaint -> Count incremented to {d3['repeat_count']} and re-opened to {d3['status']}")


def test_apscheduler_cycle():
    """Verify APScheduler background execution cycle."""
    print("\n--- Testing APScheduler Background Cycle ---")
    from app.services.scheduler_service import run_auto_escalation_cycle
    result = run_auto_escalation_cycle()
    assert "scanned" in result
    assert "escalated" in result
    print(f"✓ APScheduler execution cycle verified: Scanned {result['scanned']} tickets, auto-escalated {result['escalated']}.")


if __name__ == "__main__":
    print("\n--- Running PCMC Sarathi AI Verification Suite ---")
    test_health_endpoint()
    test_full_complaint_lifecycle()
    test_priority_and_3level_escalation()
    test_fraud_and_spam_detection()
    test_duplicate_and_repeat_count_increment()
    test_apscheduler_cycle()
    print("\n--- ALL TESTS PASSED SUCCESSFULLY! ---\n")
