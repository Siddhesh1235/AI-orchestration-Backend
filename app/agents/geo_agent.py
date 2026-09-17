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


geo_agent = GeoAgent()
