"""
Centralized Category Registry for WardMitra AI.
Single Source of Truth for all 12 Dataset Categories across:
- YOLO11 Image Classification (best.pt)
- NLP Intent & Keyword Detection
- Conversational Chatbot Flow & State Management
- Department Routing & SLA Calculation
- Frontend Dashboard & Analytics
"""

from typing import Dict, Any, List, Optional

CIVIC_12_CATEGORIES: List[Dict[str, Any]] = [
    {
        "id": 1,
        "key": "streetlight",
        "model_class": "StreetLight",
        "aliases": ["street_light", "damaged_streetlights", "light", "lamp", "पथदिवा", "स्ट्रीट लाइट"],
        "name_en": "Street Light Defect",
        "name_mr": "पथदिवा समस्या",
        "name_hi": "स्ट्रीट लाइट की समस्या",
        "department_code": "ELECTRICAL",
        "department_name": "Electrical Department",
        "department_name_mr": "विद्युत विभाग",
        "sla_hours": 24,
        "icon": "fa-lightbulb",
        "required_fields": ["location", "description"],
        "clarification_prompt_en": "Is the streetlight completely OFF, flickering, or is the pole/wiring physically damaged?",
        "clarification_prompt_mr": "पथदिवा पूर्णपणे बंद आहे, लुकलुकत आहे की खांब किंवा वायर नादुरुस्त आहे?",
        "clarification_prompt_hi": "क्या स्ट्रीट लाइट पूरी तरह बंद है, टिमटिमा रही है या खंभा/तार क्षतिग्रस्त है?",
        "keywords": [
            "streetlight", "street light", "light pole", "lamp", "streetlamp",
            "पथदिवा", "दिवा", "लाईट", "अंधार", "स्ट्रीटलाईट", "स्ट्रीट लाइट",
            "लाइट", "खंभा", "बत्ती", "पोल"
        ]
    },
    {
        "id": 2,
        "key": "pothole",
        "model_class": "PotHoles",
        "aliases": ["potholes", "road_damage", "road_incidents", "खड्डा", "गड्ढा"],
        "name_en": "Potholes & Road Damage",
        "name_mr": "रस्त्यावरील खड्डे",
        "name_hi": "सड़क के गड्ढे और खराबी",
        "department_code": "CIVIL_ROADS",
        "department_name": "Roads & Civil Infrastructure",
        "department_name_mr": "रस्ते व स्थापत्य विभाग",
        "sla_hours": 48,
        "icon": "fa-road",
        "required_fields": ["location", "description"],
        "clarification_prompt_en": "Please describe the road damage/pothole location and share a photo of the road surface.",
        "clarification_prompt_mr": "कृपया रस्त्यावरील खड्ड्याचे नेमके ठिकाण सांगा आणि खड्ड्याचा फोटो जोडा.",
        "clarification_prompt_hi": "कृपया सड़क के गड्ढे का सटीक स्थान बताएं और सड़क की तस्वीर साझा करें।",
        "keywords": [
            "pothole", "potholes", "road damage", "crater", "tarmac", "asphalt", "broken road",
            "खड्डा", "खड्डे", "खराब रस्ता", "खड्डेमय", "डामर", "गड्ढा", "गड्ढे", "सड़क खराब"
        ]
    },
    {
        "id": 3,
        "key": "garbage",
        "model_class": "Garbage",
        "aliases": ["overflowing_garbage", "waste", "trash", "solid_waste", "कचरा"],
        "name_en": "Garbage & Waste Accumulation",
        "name_mr": "कचरा व घनकचरा",
        "name_hi": "कचरा और गंदगी का ढेर",
        "department_code": "HEALTH_SWM",
        "department_name": "Health & Solid Waste Management",
        "department_name_mr": "आरोग्य व घनकचरा व्यवस्थापन विभाग",
        "sla_hours": 24,
        "icon": "fa-trash-can",
        "required_fields": ["location", "description"],
        "clarification_prompt_en": "Is the garbage bin overflowing or is it an open dumping spot near a street/residential area?",
        "clarification_prompt_mr": "कचराकुंडी भरून वाहत आहे की उघड्या जागेवर कचरा साचला आहे?",
        "clarification_prompt_hi": "क्या कचरा पेटी भर गई है या खुले में कचरे का ढेर लगा है?",
        "keywords": [
            "garbage", "trash", "waste", "dumping", "dustbin", "litter", "debris",
            "कचरा", "घाण", "कचराकुंडी", "उकिरडा", "कूड़ा", "कचरे का ढेर",
            "कचरा साचला", "साचला", "कचऱ्याचा ढीग", "दुर्गंधी", "घाण वास", "बदबू"
        ]
    },
    {
        "id": 4,
        "key": "drainage",
        "model_class": "Drainage",
        "aliases": ["drainage_failures", "gutter", "sewage", "manhole", "ड्रेनेज", "गटर"],
        "name_en": "Drainage & Open Manhole",
        "name_mr": "ड्रेनेज व उघडे मॅनहोल",
        "name_hi": "ड्रेनेज और खुला मैनहोल",
        "department_code": "DRAINAGE_DEPT",
        "department_name": "Drainage & Sewerage Department",
        "department_name_mr": "जलनिस्सारण व ड्रेनेज विभाग",
        "sla_hours": 24,
        "icon": "fa-water",
        "required_fields": ["location", "description"],
        "clarification_prompt_en": "Is the drain blocked, sewage overflowing onto the street, or is a manhole cover broken/missing?",
        "clarification_prompt_mr": "गटार तुंबले आहे, सांडपाणी रस्त्यावर येत आहे की मॅनहोलचे झाकण तुटले/गायब आहे?",
        "clarification_prompt_hi": "क्या नाली जाम है, सीवर का गंदा पानी सड़क पर आ रहा है या मैनहोल का ढक्कन टूटा/गायब है?",
        "keywords": [
            "drainage", "drain", "gutter", "manhole", "sewage", "overflowing drain",
            "सांडपाणी", "ड्रेनेज", "गटर", "गटार", "नाले", "तुंबले", "नाली", "सीवर", "मैनहोल"
        ]
    },
    {
        "id": 5,
        "key": "pipeline_water_leakage",
        "model_class": "PipelineDefects",
        "aliases": ["water_leakage", "water_pipeline_leakages", "water_supply", "pipe_burst", "पाणी गळती"],
        "name_en": "Water Supply & Pipeline Leakage",
        "name_mr": "पाणीपुरवठा व पाईपलाईन गळती",
        "name_hi": "जलापूर्ति और पाइपलाइन लीकेज",
        "department_code": "WATER_SUPPLY",
        "department_name": "Water Supply Department",
        "department_name_mr": "पाणीपुरवठा विभाग",
        "sla_hours": 24,
        "icon": "fa-faucet-drip",
        "required_fields": ["location", "description"],
        "clarification_prompt_en": "Is drinking water gushing from a burst pipe, leaking underground, or contaminated water coming?",
        "clarification_prompt_mr": "पाईप फुटून पिण्याचे पाणी वाहत आहे, गळती आहे की दूषित पाणी येत आहे?",
        "clarification_prompt_hi": "क्या पाइप फूटने से पीने का पानी बह रहा है, लीकेज है या गंदा पानी आ रहा है?",
        "keywords": [
            "water", "pipeline", "leakage", "tap", "pipe burst", "drinking water",
            "पाणी", "गळती", "पाईपलाईन", "नळ", "जलपुरवठा", "पानी", "लीकेज", "पाइपलाइन"
        ]
    },
    {
        "id": 6,
        "key": "health_sanitation",
        "model_class": "Health_Sanitation",
        "aliases": ["sanitation", "public_toilet", "health", "hygiene", "रोगराई", "शौचालय"],
        "name_en": "Public Health & Sanitation",
        "name_mr": "सार्वजनिक आरोग्य व स्वच्छता",
        "name_hi": "सार्वजनिक स्वास्थ्य और स्वच्छता",
        "department_code": "HEALTH_SWM",
        "department_name": "Health & Sanitation Department",
        "department_name_mr": "आरोग्य विभाग",
        "sla_hours": 24,
        "icon": "fa-pump-soap",
        "required_fields": ["location", "description"],
        "clarification_prompt_en": "Is this regarding an unclean public toilet, open defecation, or pest/mosquito breeding?",
        "clarification_prompt_mr": "ही तक्रार अस्वच्छ सार्वजनिक शौचालय, डासांची उत्पत्ती किंवा दुर्गंधीबाबत आहे का?",
        "clarification_prompt_hi": "क्या यह शिकायत गंदे सार्वजनिक शौचालय, मच्छर पनपने या दुर्गंध के बारे में है?",
        "keywords": [
            "sanitation", "toilet", "public toilet", "urinal", "hygiene", "mosquito", "pest", "stagnant water",
            "शौचालय", "सार्वजनिक स्वच्छता", "आरोग्य", "डास", "दुर्गंधी", "सफाई", "मच्छर"
        ]
    },
    {
        "id": 7,
        "key": "trees",
        "model_class": "Trees",
        "aliases": ["tree", "fallen_tree", "tree_branch", "garden_tree", "झाड"],
        "name_en": "Fallen Trees & Overgrowth",
        "name_mr": "झाड पडणे व धोकादायक फांद्या",
        "name_hi": "पेड़ गिरना और खतरनाक डालियां",
        "department_code": "GARDEN_TREE",
        "department_name": "Garden & Tree Authority",
        "department_name_mr": "उद्यान व वृक्ष प्राधिकरण विभाग",
        "sla_hours": 48,
        "icon": "fa-tree",
        "required_fields": ["location", "description"],
        "clarification_prompt_en": "Has a tree fallen blocking the street or are dangerously overgrown branches threatening wires?",
        "clarification_prompt_mr": "झाड पडून रस्ता अडवला आहे की धोकादायक फांद्या विद्युत तारांवर झुकल्या आहेत?",
        "clarification_prompt_hi": "क्या पेड़ गिरने से रास्ता बंद है या खतरनाक शाखाएं तारों पर लटकी हैं?",
        "keywords": [
            "tree", "branch", "fallen tree", "garden", "tree trimming", "uprooted tree",
            "झाड", "फांदी", "झाडे", "वृक्ष", "उद्यान", "झाड पडले", "पेड़", "डाल", "शाखा"
        ]
    },
    {
        "id": 8,
        "key": "traffic_jams",
        "model_class": "Road_Incidents_Traffic",
        "aliases": ["traffic", "traffic_jam", "road_incidents_traffic", "road_incident", "traffic_signal", "वाहतूक कोंडी"],
        "name_en": "Traffic Congestion & Road Incidents",
        "name_mr": "वाहतूक कोंडी व अपघात",
        "name_hi": "ट्रैफिक जाम और सड़क घटनाएं",
        "department_code": "TRAFFIC_CELL",
        "department_name": "Traffic Cell & Police Coordination",
        "department_name_mr": "वाहतूक नियोजन कक्ष",
        "sla_hours": 4,
        "icon": "fa-traffic-light",
        "required_fields": ["location", "description"],
        "clarification_prompt_en": "What is the exact junction or road affected, and is a non-functioning traffic signal causing it?",
        "clarification_prompt_mr": "नेमका कोणता चौक किंवा रस्ता प्रभावित झाला आहे आणि सिग्नल बंद असल्यामुळे कोंडी झाली आहे का?",
        "clarification_prompt_hi": "सटीक कौन सा चौक या सड़क प्रभावित है और क्या ट्रैफिक सिग्नल बंद होने से जाम लगा है?",
        "keywords": [
            "traffic", "jam", "congestion", "signal", "traffic light", "accident", "gridlock",
            "वाहतूक", "ट्रॅफिक", "कोंडी", "सिग्नल", "जाम", "चौक", "गाड़ियाँ", "ट्रैफिक जाम"
        ]
    },
    {
        "id": 9,
        "key": "electricity",
        "model_class": "Electricity",
        "aliases": ["open_wires", "transformer", "power_hazard", "current", "विद्युत धोका"],
        "name_en": "Electrical Hazards & Exposed Wires",
        "name_mr": "उघड्या विद्युत तारा व डीपी धोका",
        "name_hi": "खुले बिजली के तार और ट्रांसफार्मर खतरा",
        "department_code": "ELECTRICAL",
        "department_name": "Electrical & MSEDCL Coordination",
        "department_name_mr": "विद्युत विभाग / MSEDCL समन्वय",
        "sla_hours": 12,
        "icon": "fa-bolt",
        "required_fields": ["location", "description"],
        "clarification_prompt_en": "Are live wires dangling, is a transformer (DP) sparking, or is an electrical box open?",
        "clarification_prompt_mr": "जिवंत तारा लटकत आहेत, ट्रान्सफॉर्मर (डीपी) मधून स्पार्किंग होत आहे की विद्युत पेटी उघडी आहे?",
        "clarification_prompt_hi": "क्या खुले तार लटक रहे हैं, ट्रांसफार्मर (डीपी) से चिंगारी निकल रही है या बॉक्स खुला है?",
        "keywords": [
            "wire", "sparking", "transformer", "shock", "current", "live wire", "feeder box", "electric pole",
            "विद्युत", "उघडी तार", "शॉर्ट सर्किट", "डीपी", "करंट", "बिजली", "तार", "ट्रांसफार्मर", "शॉर्ट सर्किट"
        ]
    },
    {
        "id": 10,
        "key": "encroachment",
        "model_class": "Encroachment",
        "aliases": ["illegal_hawkers", "footpath_encroachment", "illegal_stalls", "अतिक्रमण"],
        "name_en": "Footpath Encroachment & Illegal Stalls",
        "name_mr": "पदपथ अतिक्रमण व अनधिकृत स्टॉल",
        "name_hi": "फुटपाथ अतिक्रमण और अवैध दुकानें",
        "department_code": "ANTI_ENCROACHMENT",
        "department_name": "Anti-Encroachment Department",
        "department_name_mr": "अतिक्रमण निर्मूलन विभाग",
        "sla_hours": 72,
        "icon": "fa-store-slash",
        "required_fields": ["location", "description"],
        "clarification_prompt_en": "Are illegal stalls/hawkers obstructing the pedestrian footpath or commercial vehicles blocking the carriageway?",
        "clarification_prompt_mr": "अनधिकृत फेरीवाले/टपऱ्या पदपथ अडवत आहेत की रस्त्यावर बेकायदेशीर अतिक्रमण झाले आहे?",
        "clarification_prompt_hi": "क्या अवैध दुकानें/फेरीवाले फुटपाथ रोक रहे हैं या सड़क पर अतिक्रमण हुआ है?",
        "keywords": [
            "encroachment", "hawker", "illegal stall", "footpath", "pedestrian", "unauthorized construction",
            "अतिक्रमण", "फेरीवाले", "फूटपाथ", "हॉकर्स", "बेकायदेशीर", "कब्जा", "अवैध दुकान"
        ]
    },
    {
        "id": 11,
        "key": "unauthorized_banner_flex",
        "model_class": "Banners_Flex",
        "aliases": ["banners_flex", "banner", "flex", "hoarding", "illegal_poster", "बॅनर", "फ्लेक्स"],
        "name_en": "Unauthorized Banners & Flex Hoardings",
        "name_mr": "अनधिकृत बॅनर व फ्लेक्स होर्डिंग",
        "name_hi": "अवैध बैनर और फ्लेक्स होर्डिंग",
        "department_code": "SKY_SIGNS_LICENSE",
        "department_name": "Sky Signs & Licensing Department",
        "department_name_mr": "आकाशचिन्ह व परवाना विभाग",
        "sla_hours": 24,
        "icon": "fa-rectangle-ad",
        "required_fields": ["location", "description"],
        "clarification_prompt_en": "Is the banner blocking traffic vision or tied illegally across electric poles and footpaths?",
        "clarification_prompt_mr": "बॅनरमुळे वाहतुकीस अडथळा होत आहे का किंवा विद्युत खांब/पदपथावर अनधिकृतपणे लावले आहे का?",
        "clarification_prompt_hi": "क्या बैनर से यातायात बाधित हो रहा है या बिजली के खंभे/फुटपाथ पर अवैध रूप से बंधा है?",
        "keywords": [
            "banner", "flex", "hoarding", "poster", "illegal board", "cutout",
            "बॅनर", "फ्लेक्स", "होर्डिंग", "पोस्टर", "जाहिरात", "बैनर", "होर्डिंग", "पोस्टर"
        ]
    },
    {
        "id": 12,
        "key": "noise_pollution",
        "model_class": "Noise_Pollution",
        "aliases": ["loudspeaker", "dj_noise", "industrial_noise", "ध्वनी प्रदूषण"],
        "name_en": "Noise Pollution & Loudspeakers",
        "name_mr": "ध्वनी प्रदूषण व लाऊडस्पीकर",
        "name_hi": "ध्वनि प्रदूषण और लाउडस्पीकर",
        "department_code": "POLLUTION_CONTROL",
        "department_name": "Environment & Pollution Control",
        "department_name_mr": "पर्यावरण व प्रदूषण नियंत्रण विभाग",
        "sla_hours": 4,
        "icon": "fa-volume-high",
        "required_fields": ["location", "description"],
        "clarification_prompt_en": "Is an unauthorized DJ/loudspeaker operating after permitted hours or high-decibel industrial machinery in a residential zone?",
        "clarification_prompt_mr": "रात्री उशिरापर्यंत लाऊडस्पीकर/डीजे चालू आहे की रहिवासी भागात प्रचंड आवाजाचे यंत्र चालू आहे?",
        "clarification_prompt_hi": "क्या देर रात तक लाउडस्पीकर/डीजे बज रहा है या रिहायशी इलाके में तेज आवाज वाली मशीन चल रही है?",
        "keywords": [
            "noise", "loudspeaker", "sound", "dj", "amplifier", "decibel", "noise pollution",
            "ध्वनी", "आवाज", "लाउडस्पीकर", "गोंगाट", "प्रदूषण", "शोर", "ध्वनि प्रदूषण", "डीजे"
        ]
    }
]

# Quick Lookups & Sets
CATEGORIES = [cat["key"] for cat in CIVIC_12_CATEGORIES]
CATEGORY_KEYS = CATEGORIES

CATEGORY_LOOKUP: Dict[str, Dict[str, Any]] = {}
for cat in CIVIC_12_CATEGORIES:
    CATEGORY_LOOKUP[cat["key"].lower()] = cat
    CATEGORY_LOOKUP[cat["model_class"].lower()] = cat
    for alias in cat["aliases"]:
        CATEGORY_LOOKUP[alias.lower()] = cat


def normalize_category_key(raw: Optional[str]) -> str:
    """
    Normalizes any raw input (YOLO class, user text, alias, database value)
    to one of the 12 canonical category keys, or 'other' if unknown.
    """
    if not raw:
        return "other"
    cleaned = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned in CATEGORY_LOOKUP:
        return CATEGORY_LOOKUP[cleaned]["key"]
    # Partial match checking
    for key, item in CATEGORY_LOOKUP.items():
        if key in cleaned or cleaned in key:
            return item["key"]
    return "other"


def get_category_info(category_key: str) -> Dict[str, Any]:
    """
    Returns full category metadata dict for given key/alias.
    """
    normalized = normalize_category_key(category_key)
    if normalized in CATEGORY_LOOKUP:
        return CATEGORY_LOOKUP[normalized]
    # Return a sensible fallback if 'other'
    return {
        "id": 99,
        "key": "other",
        "model_class": "Other",
        "aliases": ["general", "other"],
        "name_en": "General Civic Issue",
        "name_mr": "इतर नागरी समस्या",
        "name_hi": "अन्य नागरिक समस्या",
        "department_code": "GENERAL_ADMIN",
        "department_name": "General Administration",
        "department_name_mr": "सामान्य प्रशासन विभाग",
        "sla_hours": 48,
        "icon": "fa-circle-info",
        "required_fields": ["location", "description"],
        "clarification_prompt_en": "Please provide more details regarding the civic issue and its location.",
        "clarification_prompt_mr": "कृपया नागरी समस्येचे स्वरूप आणि ठिकाण सांगा.",
        "clarification_prompt_hi": "कृपया समस्या का विवरण और स्थान बताएं।",
        "keywords": []
    }


def get_all_categories() -> List[Dict[str, Any]]:
    """Returns list of all 12 categories."""
    return CIVIC_12_CATEGORIES


def is_valid_category(category_key: str) -> bool:
    """Returns True if category is one of the 12 valid dataset categories."""
    normalized = normalize_category_key(category_key)
    return normalized in CATEGORY_KEYS
