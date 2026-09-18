"""
Deterministic Priority Rules Engine for WardMitra AI.
Maps Severity, Emergency, and Category signals directly to P1, P2, P3, P4.
"""

from typing import Optional, Dict, Any
from app.database.models import PriorityLevel


def calculate_priority_level(
    severity: str,
    is_emergency: bool,
    category: Optional[str] = None,
    requested_priority: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deterministically computes complaint priority (P1 - P4).
    Rules:
      - P1 (Immediate Action Required): Any emergency situation OR CRITICAL severity.
      - P2 (High Priority): HIGH severity issues posing public inconvenience or safety hazard.
      - P3 (Standard Priority): MEDIUM severity civic grievances (standard potholes, garbage, streetlights).
      - P4 (Low Priority): LOW severity cosmetic or routine requests (beautification, non-blocking flex, inquiries).
    """
    sev_upper = (severity or "MEDIUM").upper()

    # Admin / Citizen override check (if valid P1-P4 or LOW/MEDIUM/HIGH)
    if requested_priority:
        req = requested_priority.strip().upper()
        if req in ["P1", "CRITICAL"]:
            return {
                "priority": "P1",
                "legacy_enum": PriorityLevel.HIGH,
                "reason": "Requested P1 Priority",
                "is_automated": False
            }
        elif req in ["P2", "HIGH"]:
            return {
                "priority": "P2",
                "legacy_enum": PriorityLevel.HIGH,
                "reason": "Requested P2 Priority",
                "is_automated": False
            }
        elif req in ["P3", "MEDIUM"]:
            return {
                "priority": "P3",
                "legacy_enum": PriorityLevel.MEDIUM,
                "reason": "Requested P3 Priority",
                "is_automated": False
            }
        elif req in ["P4", "LOW"]:
            return {
                "priority": "P4",
                "legacy_enum": PriorityLevel.LOW,
                "reason": "Requested P4 Priority",
                "is_automated": False
            }

    # Deterministic Rule Evaluation:
    # 1. Any confirmed emergency or critical severity is P1
    if is_emergency or sev_upper == "CRITICAL":
        return {
            "priority": "P1",
            "legacy_enum": PriorityLevel.HIGH,
            "reason": "Immediate emergency or critical public safety hazard",
            "is_automated": True
        }

    # 2. HIGH severity is P2
    if sev_upper == "HIGH":
        return {
            "priority": "P2",
            "legacy_enum": PriorityLevel.HIGH,
            "reason": "High severity civic disruption without active emergency",
            "is_automated": True
        }

    # 3. LOW severity is P4
    if sev_upper == "LOW":
        return {
            "priority": "P4",
            "legacy_enum": PriorityLevel.LOW,
            "reason": "Low severity or routine civic maintenance",
            "is_automated": True
        }

    # 4. Standard default is P3
    return {
        "priority": "P3",
        "legacy_enum": PriorityLevel.MEDIUM,
        "reason": "Standard civic grievance priority",
        "is_automated": True
    }
