"""
Geospatial & Ward Mapping Agent for PCMC Sarathi AI.
Maps GPS coordinates (latitude, longitude) to one of PCMC's 32 administrative Prabhags/Wards.
Calculates nearest centroid and administrative Zone (A to H).
"""

import math
from typing import Dict, Any, Optional

# PCMC Prabhag/Ward Centroids (Key Municipal Locations across 32 Wards)
PCMC_WARD_CENTROIDS = [
    {"ward": 1, "name": "Chikhali - Talwade", "lat": 18.7012, "lng": 73.8150, "zone": "Zone A"},
    {"ward": 2, "name": "Rupeenagar - Sambhajinagar", "lat": 18.6870, "lng": 73.7950, "zone": "Zone A"},
    {"ward": 3, "name": "Yamunanagar - Nigdi", "lat": 18.6650, "lng": 73.7740, "zone": "Zone A"},
    {"ward": 4, "name": "Pradhikaran - Sector 24", "lat": 18.6530, "lng": 73.7660, "zone": "Zone B"},
    {"ward": 5, "name": "Akurdi Gaonthan - Mohan Nagar", "lat": 18.6480, "lng": 73.7850, "zone": "Zone B"},
    {"ward": 6, "name": "Shahunagar - Masulkar Colony", "lat": 18.6420, "lng": 73.8050, "zone": "Zone B"},
    {"ward": 7, "name": "Moshi Gaon - Borhadewadi", "lat": 18.6830, "lng": 73.8420, "zone": "Zone C"},
    {"ward": 8, "name": "Dudulgaon - Charholi", "lat": 18.6710, "lng": 73.8750, "zone": "Zone C"},
    {"ward": 9, "name": "Dighi - Magazine Corner", "lat": 18.6250, "lng": 73.8710, "zone": "Zone C"},
    {"ward": 10, "name": "Bhosari Gaonthan - Dhavade Vasti", "lat": 18.6280, "lng": 73.8440, "zone": "Zone C"},
    {"ward": 11, "name": "Indrayaninagar - Landewadi", "lat": 18.6360, "lng": 73.8320, "zone": "Zone C"},
    {"ward": 12, "name": "MIDC Bhosari - Telco Road", "lat": 18.6420, "lng": 73.8190, "zone": "Zone C"},
    {"ward": 13, "name": "Nehrunagar - Vallabhnagar", "lat": 18.6240, "lng": 73.8220, "zone": "Zone D"},
    {"ward": 14, "name": "Sant Tukaram Nagar - YCM Hospital", "lat": 18.6180, "lng": 73.8110, "zone": "Zone D"},
    {"ward": 15, "name": "Pimpri Gaon - Morwadi", "lat": 18.6270, "lng": 73.7990, "zone": "Zone D"},
    {"ward": 16, "name": "Kasarwadi - Kundan Nagar", "lat": 18.6010, "lng": 73.8210, "zone": "Zone D"},
    {"ward": 17, "name": "Phugewadi - Dapodi", "lat": 18.5830, "lng": 73.8290, "zone": "Zone D"},
    {"ward": 18, "name": "Pimple Gurav - Katepuram", "lat": 18.5910, "lng": 73.8080, "zone": "Zone E"},
    {"ward": 19, "name": "Old Sangvi - Shitole Nagar", "lat": 18.5770, "lng": 73.8110, "zone": "Zone E"},
    {"ward": 20, "name": "New Sangvi - Famous Chowk", "lat": 18.5840, "lng": 73.7990, "zone": "Zone E"},
    {"ward": 21, "name": "Pimple Nilakh - Vishal Nagar", "lat": 18.5900, "lng": 73.7820, "zone": "Zone E"},
    {"ward": 22, "name": "Pimple Saudagar - Roseland", "lat": 18.6020, "lng": 73.7890, "zone": "Zone F"},
    {"ward": 23, "name": "Kokane Nagar - Kunal Icon", "lat": 18.6060, "lng": 73.7960, "zone": "Zone F"},
    {"ward": 24, "name": "Rahatani - Ram Nagar", "lat": 18.6150, "lng": 73.7830, "zone": "Zone F"},
    {"ward": 25, "name": "Wakad - Dutta Mandir", "lat": 18.6010, "lng": 73.7630, "zone": "Zone G"},
    {"ward": 26, "name": "Kaspate Vasti - Hinjawadi Road", "lat": 18.5920, "lng": 73.7550, "zone": "Zone G"},
    {"ward": 27, "name": "Thergaon - Dange Chowk", "lat": 18.6190, "lng": 73.7710, "zone": "Zone G"},
    {"ward": 28, "name": "Kalewadi - Vijay Nagar", "lat": 18.6250, "lng": 73.7860, "zone": "Zone G"},
    {"ward": 29, "name": "Chinchwad Station - Anandnagar", "lat": 18.6360, "lng": 73.7890, "zone": "Zone H"},
    {"ward": 30, "name": "Chinchwadgaon - Keshavnagar", "lat": 18.6290, "lng": 73.7730, "zone": "Zone H"},
    {"ward": 31, "name": "Walhekarwadi - Gurav Pimple", "lat": 18.6380, "lng": 73.7540, "zone": "Zone H"},
    {"ward": 32, "name": "Ravet - Kiwale - Shinde Vasti", "lat": 18.6520, "lng": 73.7440, "zone": "Zone H"}
]


class GeoAgent:
    def map_coordinates_to_ward(self, lat: Optional[float], lng: Optional[float]) -> Dict[str, Any]:
        """
        Maps latitude and longitude to PCMC Ward Number (1 to 32) and Ward Name.
        Uses Euclidean minimum-distance algorithm against PCMC municipal ward centroids.
        """
        if lat is None or lng is None or (lat == 0.0 and lng == 0.0):
            # Default to central ward (Pimpri Ward 15)
            default_ward = PCMC_WARD_CENTROIDS[14]
            return {
                "ward_number": default_ward["ward"],
                "ward_name": default_ward["name"],
                "zone": default_ward["zone"],
                "is_fallback": True
            }

        best_ward = None
        min_dist = float("inf")

        for ward in PCMC_WARD_CENTROIDS:
            # Distance formula (d = sqrt((lat2 - lat1)^2 + (lng2 - lng1)^2))
            dist = math.sqrt((lat - ward["lat"]) ** 2 + (lng - ward["lng"]) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_ward = ward

        return {
            "ward_number": best_ward["ward"],
            "ward_name": best_ward["name"],
            "zone": best_ward["zone"],
            "is_fallback": False
        }

    def get_ward_by_number(self, ward_num: int) -> Optional[Dict[str, Any]]:
        """Returns ward details for a specific ward number (1 to 32)."""
        for ward in PCMC_WARD_CENTROIDS:
            if ward["ward"] == ward_num:
                return {
                    "ward_number": ward["ward"],
                    "ward_name": ward["name"],
                    "zone": ward["zone"],
                    "lat": ward["lat"],
                    "lng": ward["lng"],
                    "is_fallback": False
                }
        return None

    def detect_ward_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Scans complaint text for PCMC landmarks or area names to dynamically resolve ward."""
        if not text:
            return None
        t_lower = text.lower()

        area_keywords = {
            "चिखली": 1, "तळवडे": 1, "chikhali": 1, "talwade": 1,
            "रुपीनगर": 2, "संभाजीनगर": 2, "rupinagar": 2, "sambhajinagar": 2,
            "यमुनानगर": 3, "निगडी": 3, "yamunanagar": 3, "nigdi": 3,
            "प्राधिकरण": 4, "pradhikaran": 4, "sector 24": 4, "सेक्टर २४": 4,
            "आकुर्डी": 5, "मोहन नगर": 5, "akurdi": 5, "mohan nagar": 5,
            "शाहूनगर": 6, "मसुळकर कॉलनी": 6, "shahunagar": 6, "masulkar": 6,
            "मोशी": 7, "बोरहाडेवाडी": 7, "moshi": 7, "borhadewadi": 7,
            "डुडुळगाव": 8, "चऱ्होली": 8, "dudulgaon": 8, "charholi": 8,
            "दिघी": 9, "मॅगझिन कॉर्नर": 9, "dighi": 9, "magazine": 9,
            "भोसरी": 10, "धावडे वस्ती": 10, "bhosari": 10, "dhavade": 10,
            "इंद्रायणी नगर": 11, "लांडेवाडी": 11, "indrayani": 11, "landewadi": 11,
            "एमआयडीसी": 12, "टेल्को": 12, "telco": 12, "midc bhosari": 12,
            "नेहरूनगर": 13, "वल्लभनगर": 13, "nehrunagar": 13, "vallabhnagar": 13,
            "संत तुकाराम नगर": 14, "वायसीएम": 14, "ycm": 14, "sant tukaram": 14,
            "पिंपरी": 15, "मोरवाडी": 15, "pimpri": 15, "morwadi": 15,
            "कासारवाडी": 16, "कुंदन नगर": 16, "kasarwadi": 16, "kundan nagar": 16,
            "फुगेवाडी": 17, "दापोडी": 17, "phugewadi": 17, "dapodi": 17,
            "पिंपळे गुरव": 18, "काटेपुरम": 18, "pimple gurav": 18, "katepuram": 18,
            "नवी सांगवी": 20, "फेमस चौक": 20, "new sangvi": 20, "famous chowk": 20,
            "जुनी सांगवी": 19, "शितोळे नगर": 19, "old sangvi": 19, "sangvi": 19, "सांगवी": 19,
            "पिंपळे निलख": 21, "विशाल नगर": 21, "pimple nilakh": 21, "vishal nagar": 21,
            "पिंपळे सौदागर": 22, "रोजलँड": 22, "pimple saudagar": 22, "roseland": 22,
            "कोकाणे नगर": 23, "कुणाल आयकॉन": 23, "kokane nagar": 23, "kunal icon": 23,
            "रहाटणी": 24, "राम नगर": 24, "rahatani": 24, "ram nagar": 24,
            "वाकड": 25, "दत्त मंदिर": 25, "wakad": 25, "dutta mandir": 25,
            "कस्पटे वस्ती": 26, "हिंजवडी": 26, "kaspate": 26, "hinjawadi": 26, "hinjewadi": 26,
            "थेरगाव": 27, "डांगे चौक": 27, "thergaon": 27, "dange chowk": 27,
            "काळेवाडी": 28, "विजय नगर": 28, "kalewadi": 28, "vijay nagar": 28,
            "चिंचवड स्टेशन": 29, "आनंदनगर": 29, "chinchwad station": 29, "anandnagar": 29,
            "चिंचवड": 30, "केशवनगर": 30, "chinchwad": 30, "chinchwadgaon": 30,
            "वाल्हेकरवाडी": 31, "गुरव पिंपळे": 31, "walhekarwadi": 31,
            "रावेत": 32, "किवळे": 32, "शिंदे वस्ती": 32, "ravet": 32, "kiwale": 32
        }

        for kw, wnum in area_keywords.items():
            if kw in t_lower:
                return self.get_ward_by_number(wnum)

        return None


geo_agent = GeoAgent()

