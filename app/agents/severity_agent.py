"""
Severity & Priority Grading Agent for WardMitra AI / PCMC Sarathi AI.
Assigns Severity (LOW / MEDIUM / HIGH / CRITICAL), Emergency Status,
and Deterministic Priority (P1 / P2 / P3 / P4).
Calculates dynamic SLA and 3-level escalation timelines.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone

from app.database.models import PriorityLevel
from app.rules.severity_rules import evaluate_emergency_status, evaluate_severity_level
from app.rules.priority_rules import calculate_priority_level

logger = logging.getLogger("pcms.severity_agent")


class SeverityAgent:
    def detect_emergency(self, description: str, category: str) -> Dict[str, Any]:
        """
        Requirement 9: Determines whether the complaint is potentially an emergency.
        Returns:
            is_emergency: bool
            emergency_level: "CRITICAL" | "HIGH" | "NONE"
            reason: str
        """
        return evaluate_emergency_status(description, category)

    def evaluate_severity(
        self,
        description: str,
        category: str,
        is_emergency: bool = False
    ) -> Dict[str, Any]:
        """
        Requirement 8: Calculates complaint severity (LOW, MEDIUM, HIGH, CRITICAL).
        """
        return evaluate_severity_level(description, category, is_emergency)

    def evaluate_priority(
        self,
        description: str,
        category: str,
        requested_priority: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Requirement 10: Calculates complaint priority (P1-P4) separately from emergency status,
        while maintaining backward compatibility with legacy PriorityLevel enum.
        """
        emergency_eval = self.detect_emergency(description, category)
        is_emergency = emergency_eval["is_emergency"]
        emergency_level = emergency_eval["emergency_level"]

        severity_eval = self.evaluate_severity(description, category, is_emergency)
        severity = severity_eval["severity"]

        priority_eval = calculate_priority_level(
            severity=severity,
            is_emergency=is_emergency,
            category=category,
            requested_priority=requested_priority
        )

        return {
            "priority": priority_eval["legacy_enum"],
            "priority_code": priority_eval["priority"],  # "P1", "P2", "P3", "P4"
            "severity": severity,
            "is_emergency": is_emergency,
            "emergency_level": emergency_level,
            "reason": priority_eval["reason"],
            "emergency_reason": emergency_eval["reason"],
            "is_automated": priority_eval["is_automated"]
        }

    def get_escalation_schedule(
        self,
        priority: Any,
        base_sla_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Calculates dynamic SLA and thresholds for 3-Level Escalation:
        Level 1 (Worker) -> Level 2 (Supervisor) -> Level 3 (HOD).
        Accepts PriorityLevel enum or string ("P1", "P2", "P3", "P4", "HIGH", etc.).
        """
        p_val = priority.value if hasattr(priority, "value") else str(priority)

        if p_val in ["P1", "HIGH", "CRITICAL"]:
            sla_hours = min(base_sla_hours, 12)
            worker_hours = 4
            supervisor_hours = 4
        elif p_val in ["P4", "LOW"]:
            sla_hours = max(base_sla_hours, 48)
            worker_hours = 24
            supervisor_hours = 24
        else:  # P2, P3, MEDIUM
            sla_hours = base_sla_hours
            worker_hours = max(4, int(sla_hours * 0.5))
            supervisor_hours = max(4, int(sla_hours * 0.5))

        now = datetime.now(timezone.utc)
        return {
            "sla_hours": sla_hours,
            "sla_deadline": now + timedelta(hours=sla_hours),
            "worker_threshold_hours": worker_hours,
            "supervisor_threshold_hours": supervisor_hours
        }


severity_agent = SeverityAgent()
