"""
Deterministic Severity & Emergency Detection Rules for WardMitra AI.
Evaluates description, category, and public safety impact to assign
Severity (LOW, MEDIUM, HIGH, CRITICAL) and Emergency status.
"""

from typing import Dict, Any, Optional

EMERGENCY_SIGNALS = [
    # English
    "live wire", "exposed wire", "open wire", "sparking transformer", "short circuit",
    "electrocuted", "electrocution", "electric shock", "pipeline burst", "pipe burst",
    "massive flood", "dangerous flooding", "road cave in", "sinkhole", "fatal", "fire hazard",
    "severe road obstruction", "explosion", "immediate danger", "life threatening",
    # Marathi
    "उघडी तार", "जिवंत तार", "शॉर्ट सर्किट", "स्पार्किंग", "करंट", "विद्युत धक्का",
    "पाईप फुटला", "मोठी पाणी गळती", "पूर", "रस्ता खचला", "मोठा खड्डा पडून रस्ता बंद",
    "जीवितहानी", "आग", "स्फोट", "तातडीने जीव धोक्यात"
]

HIGH_SEVERITY_SIGNALS = [
    # English
    "manhole open", "open manhole", "deep pothole", "traffic blocked", "fallen tree",
    "hospital entrance blocked", "school bus stuck", "contaminated water", "sewage overflowing",
    "major accident", "sparking", "dark black spot",
    # Marathi
    "मॅनहोल उघडे", "उघडे गटर", "खूप मोठा खड्डा", "वाहतूक ठप्प", "झाड पडले",
    "सांडपाणी घरात शिरले", "गढूळ पाणी", "अपघात"
]

LOW_SEVERITY_SIGNALS = [
    # English
    "trimming", "suggestion", "beautification", "minor", "inquiry", "request",
    "banner", "flex", "poster", "routine",
    # Marathi
    "फांद्या छाटणे", "सूचना", "सुशोभीकरण", "साधी विनंती", "माहिती", "चौकशी",
    "फ्लेक्स", "बॅनर"
]


def evaluate_emergency_status(description: str, category: str) -> Dict[str, Any]:
    """
    Evaluates whether the complaint constitutes a civic/life-safety emergency.
    """
    desc_clean = (description or "").lower()

    for signal in EMERGENCY_SIGNALS:
        if signal.lower() in desc_clean:
            level = "CRITICAL" if any(w in signal.lower() for w in ["live wire", "उघडी तार", "electrocution", "burst", "cave in", "sinkhole"]) else "HIGH"
            return {
                "is_emergency": True,
                "emergency_level": level,
                "reason": f"Active emergency hazard detected: '{signal}'"
            }

    # Category specific emergency defaults
    if category in ["electricity", "damaged_streetlights"] and any(w in desc_clean for w in ["shock", "spark", "करंट", "स्पार्क"]):
        return {
            "is_emergency": True,
            "emergency_level": "HIGH",
            "reason": "Electrical sparking or current leakage reported"
        }

    return {
        "is_emergency": False,
        "emergency_level": "NONE",
        "reason": "No emergency hazard identified"
    }


def evaluate_severity_level(description: str, category: str, is_emergency: bool = False) -> Dict[str, Any]:
    """
    Deterministically computes complaint severity:
    CRITICAL, HIGH, MEDIUM, LOW.
    """
    if is_emergency:
        return {
            "severity": "CRITICAL",
            "reason": "Emergency safety hazard",
            "safety_impact": "CRITICAL"
        }

    desc_clean = (description or "").lower()

    # 1. Check High Severity Signals
    for signal in HIGH_SEVERITY_SIGNALS:
        if signal.lower() in desc_clean:
            return {
                "severity": "HIGH",
                "reason": f"High severity condition matched: '{signal}'",
                "safety_impact": "HIGH"
            }

    # 2. Category baseline rules
    if category in ["electricity", "traffic_jams"]:
        return {
            "severity": "HIGH",
            "reason": f"Category '{category}' has elevated baseline severity",
            "safety_impact": "MODERATE_HIGH"
        }

    # 3. Check Low Severity Signals
    for signal in LOW_SEVERITY_SIGNALS:
        if signal.lower() in desc_clean:
            return {
                "severity": "LOW",
                "reason": f"Low priority/routine issue matched: '{signal}'",
                "safety_impact": "MINIMAL"
            }

    if category in ["unauthorized_banner_flex", "noise_pollution"]:
        return {
            "severity": "LOW",
            "reason": f"Category '{category}' is non-hazardous routine maintenance",
            "safety_impact": "LOW"
        }

    # 4. Standard Default
    return {
        "severity": "MEDIUM",
        "reason": "Standard civic grievance severity",
        "safety_impact": "MODERATE"
    }
