"""
Department Routing & SLA Management Agent for PCMC Sarathi AI.
Assigns tickets to responsible PCMC departments, calculates SLA deadlines,
and generates unique standardized PCMC Ticket IDs.
"""

import yaml
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from app.config.settings import settings


class RoutingAgent:
    def __init__(self):
        self.categories_config = {}
        self.departments_config = {}
        self._load_config()

    def _load_config(self):
        try:
            with open(settings.MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.categories_config = data.get("categories", {})
                self.departments_config = data.get("departments", {})
        except Exception as e:
            print(f"[RoutingAgent] Warning: Could not load {settings.MODEL_CONFIG_PATH}: {e}")

    def route_complaint(self, category: str) -> Dict[str, Any]:
        """
        Determines responsible department, SLA hours, deadline, and localized details.
        """
        cat_info = self.categories_config.get(category, {})
        dept_code = cat_info.get("department", "HEALTH_SWM")
        sla_hours = cat_info.get("sla_hours", 24)
        
        dept_info = self.departments_config.get(dept_code, {})
        deadline = datetime.now(timezone.utc) + timedelta(hours=sla_hours)

        return {
            "department": dept_code,
            "department_name_mr": cat_info.get("department_name_mr", "आरोग्य व घनकचरा व्यवस्थापन विभाग"),
            "department_name_en": cat_info.get("department_name_en", "Health & Solid Waste Management"),
            "category_name_mr": cat_info.get("department_name_mr", category),
            "sla_hours": sla_hours,
            "sla_deadline": deadline,
            "head_officer": dept_info.get("head", "Concerned Ward Officer"),
            "officer_contact": dept_info.get("contact", "020-67333333")
        }

    def generate_ticket_id(self) -> str:
        """
        Generates Ward Mitra Standard Ticket ID: WM-YYYYMMDD-XXXX
        """
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        random_suffix = f"{random.randint(1001, 9999)}"
        return f"WM-{today_str}-{random_suffix}"


routing_agent = RoutingAgent()
