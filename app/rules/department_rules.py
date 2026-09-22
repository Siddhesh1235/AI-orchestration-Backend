"""
Configurable Department Routing Rules for WardMitra AI.
Maps civic categories to responsible municipal departments.
"""

from typing import Dict, Any

DEPARTMENT_MAPPINGS: Dict[str, Dict[str, Any]] = {
    # Roads & Infrastructure
    "pothole": {
        "department_code": "CIVIL_ROADS",
        "department_name": "Roads",
        "department_name_mr": "रस्ते व स्थापत्य विभाग",
        "sla_hours": 48
    },
    "potholes": {
        "department_code": "CIVIL_ROADS",
        "department_name": "Roads",
        "department_name_mr": "रस्ते व स्थापत्य विभाग",
        "sla_hours": 48
    },
    "road_damage": {
        "department_code": "CIVIL_ROADS",
        "department_name": "Roads",
        "department_name_mr": "रस्ते व स्थापत्य विभाग",
        "sla_hours": 48
    },

    # Electrical & Lighting
    "streetlight": {
        "department_code": "ELECTRICAL",
        "department_name": "Electrical",
        "department_name_mr": "विद्युत विभाग",
        "sla_hours": 24
    },
    "damaged_streetlights": {
        "department_code": "ELECTRICAL",
        "department_name": "Electrical",
        "department_name_mr": "विद्युत विभाग",
        "sla_hours": 24
    },
    "electricity": {
        "department_code": "ELECTRICAL",
        "department_name": "Electrical",
        "department_name_mr": "विद्युत विभाग / MSEDCL समन्वय",
        "sla_hours": 12
    },

    # Sanitation & Solid Waste
    "garbage": {
        "department_code": "HEALTH_SWM",
        "department_name": "Sanitation",
        "department_name_mr": "आरोग्य व घनकचरा व्यवस्थापन विभाग",
        "sla_hours": 24
    },
    "overflowing_garbage": {
        "department_code": "HEALTH_SWM",
        "department_name": "Sanitation",
        "department_name_mr": "आरोग्य व घनकचरा व्यवस्थापन विभाग",
        "sla_hours": 24
    },
    "illegal_debris_dumping": {
        "department_code": "HEALTH_SWM",
        "department_name": "Sanitation / Municipal Enforcement",
        "department_name_mr": "घनकचरा व अतिक्रमण निर्मूलन विभाग",
        "sla_hours": 24
    },

    # Drainage & Sewerage
    "drainage": {
        "department_code": "DRAINAGE_DEPT",
        "department_name": "Drainage",
        "department_name_mr": "जलनिस्सारण व ड्रेनेज विभाग",
        "sla_hours": 24
    },
    "drainage_failures": {
        "department_code": "DRAINAGE_DEPT",
        "department_name": "Drainage",
        "department_name_mr": "जलनिस्सारण व ड्रेनेज विभाग",
        "sla_hours": 24
    },

    # Water Supply
    "pipeline_water_leakage": {
        "department_code": "WATER_SUPPLY",
        "department_name": "Water Supply",
        "department_name_mr": "पाणीपुरवठा विभाग",
        "sla_hours": 24
    },
    "water_pipeline_leakages": {
        "department_code": "WATER_SUPPLY",
        "department_name": "Water Supply",
        "department_name_mr": "पाणीपुरवठा विभाग",
        "sla_hours": 24
    },

    # Traffic
    "traffic_jams": {
        "department_code": "TRAFFIC_CELL",
        "department_name": "Traffic Cell",
        "department_name_mr": "वाहतूक नियोजन कक्ष",
        "sla_hours": 4
    },

    # Trees & Gardens
    "trees": {
        "department_code": "GARDEN_TREE",
        "department_name": "Garden & Trees",
        "department_name_mr": "उद्यान व वृक्ष प्राधिकरण विभाग",
        "sla_hours": 48
    },

    # Anti-Encroachment
    "encroachment": {
        "department_code": "ANTI_ENCROACHMENT",
        "department_name": "Anti-Encroachment",
        "department_name_mr": "अतिक्रमण निर्मूलन विभाग",
        "sla_hours": 72
    },

    # Pollution
    "noise_pollution": {
        "department_code": "POLLUTION_CONTROL",
        "department_name": "Pollution Control",
        "department_name_mr": "पर्यावरण व प्रदूषण नियंत्रण विभाग",
        "sla_hours": 4
    },

    # Sky Signs
    "unauthorized_banner_flex": {
        "department_code": "SKY_SIGNS_LICENSE",
        "department_name": "Sky Signs & Licensing",
        "department_name_mr": "आकाशचिन्ह व परवाना विभाग",
        "sla_hours": 24
    },

    # Health & Public Sanitation
    "health_sanitation": {
        "department_code": "HEALTH_SWM",
        "department_name": "Health & Sanitation",
        "department_name_mr": "आरोग्य व स्वच्छता विभाग",
        "sla_hours": 24
    },
    "sanitation": {
        "department_code": "HEALTH_SWM",
        "department_name": "Health & Sanitation",
        "department_name_mr": "आरोग्य व स्वच्छता विभाग",
        "sla_hours": 24
    },

    # Default
    "other": {
        "department_code": "GENERAL_ADMIN",
        "department_name": "General Administration",
        "department_name_mr": "सामान्य प्रशासन विभाग",
        "sla_hours": 48
    }
}

try:
    from app.config.category_registry import normalize_category_key, get_category_info
except ImportError:
    normalize_category_key = None
    get_category_info = None


def get_department_routing(category: str) -> Dict[str, Any]:
    """
    Returns department routing info for the given civic category.
    """
    cat_key = (category or "").lower().strip()
    if cat_key in DEPARTMENT_MAPPINGS:
        return DEPARTMENT_MAPPINGS[cat_key]
    if normalize_category_key and get_category_info:
        norm = normalize_category_key(cat_key)
        info = get_category_info(norm)
        if info and "department_code" in info:
            return {
                "department_code": info.get("department_code", "GENERAL_ADMIN"),
                "department_name": info.get("department_name", "General Administration"),
                "department_name_mr": info.get("department_name_mr", "सामान्य प्रशासन विभाग"),
                "sla_hours": info.get("sla_hours", 48)
            }
    return DEPARTMENT_MAPPINGS.get("other")

