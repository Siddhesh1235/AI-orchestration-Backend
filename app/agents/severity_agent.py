"""
Severity & Priority Grading Agent for PCMC Sarathi AI.
Assigns Priority (LOW / MEDIUM / HIGH) based on citizen input,
civic hazard keywords (Marathi & English), and category severity.
Calculates dynamic SLA and 3-level escalation timelines.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone

from app.database.models import PriorityLevel

logger = logging.getLogger("pcms.severity_agent")

HIGH_PRIORITY_KEYWORDS = [
    # English
    "danger", "urgent", "emergency", "sparking", "shock", "open wire",
    "manhole open", "burst", "cave in", "ambulance", "hospital",
    "short circuit", "fatal", "hazard", "blast", "flood",
    # Marathi
    "धोकादायक", "धोका", "तातडीने", "तातडीचे", "अतितातडी", "उघडी तार",
    "शॉर्ट सर्किट", "स्पार्किंग", "करंट", "आग", "गंभीर", "मॅनहोल उघडे",
    "पाईप फुटला", "पूर", "रुग्णवाहिका", "रुग्णालय", "दवाखाना", "रस्ता खचला"
]

LOW_PRIORITY_KEYWORDS = [
    # English
    "request", "trimming", "suggestion", "beautification", "minor", "inquiry",
    # Marathi
    "फांद्या छाटणे", "सूचना", "सुशोभीकरण", "साधी विनंती", "माहिती", "चौकशी"
]


class SeverityAgent:
    def evaluate_priority(
        self,
        description: str,
        category: str,
        requested_priority: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Determines priority level (LOW, MEDIUM, HIGH) and returns rationale.
        """
        # 1. Citizen / Admin explicitly specified priority
        if requested_priority:
            req_upper = requested_priority.strip().upper()
            if req_upper in PriorityLevel.__members__:
                logger.info(f"[SeverityAgent] Using requested priority: {req_upper}")
                return {
                    "priority": PriorityLevel[req_upper],
                    "reason": f"Explicitly set to {req_upper}",
                    "is_automated": False
                }

        desc_lower = (description or "").lower()

        # 2. Check High-Priority Hazard Keywords
        for kw in HIGH_PRIORITY_KEYWORDS:
            if kw.lower() in desc_lower:
                logger.info(f"[SeverityAgent] High hazard keyword matched: '{kw}'")
                return {
                    "priority": PriorityLevel.HIGH,
                    "reason": f"Hazard keyword detected: '{kw}'",
                    "is_automated": True
                }

        # 3. Category-specific baseline rules
        if category == "electricity":
            # Live wire and transformer hazards default to HIGH
            return {
                "priority": PriorityLevel.HIGH,
                "reason": "Electrical hazards pose life-safety risk",
                "is_automated": True
            }

        if category == "unauthorized_banner_flex":
            return {
                "priority": PriorityLevel.LOW,
                "reason": "Non-hazardous advertising flex",
                "is_automated": True
            }

        # 4. Check Low-Priority Keywords
        for kw in LOW_PRIORITY_KEYWORDS:
            if kw.lower() in desc_lower:
                return {
                    "priority": PriorityLevel.LOW,
                    "reason": f"Low urgency keyword matched: '{kw}'",
                    "is_automated": True
                }

        # 5. Default standard priority
        return {
            "priority": PriorityLevel.MEDIUM,
            "reason": "Standard civic redressal priority",
            "is_automated": True
        }

    def get_escalation_schedule(
        self,
        priority: PriorityLevel,
        base_sla_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Calculates dynamic SLA and thresholds for 3-Level Escalation:
        Level 1 (Worker) -> Level 2 (Supervisor) -> Level 3 (HOD)
        """
        if priority == PriorityLevel.HIGH:
            sla_hours = min(base_sla_hours, 12)
            worker_hours = 4
            supervisor_hours = 4
        elif priority == PriorityLevel.LOW:
            sla_hours = max(base_sla_hours, 48)
            worker_hours = 24
            supervisor_hours = 24
        else:  # MEDIUM
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
