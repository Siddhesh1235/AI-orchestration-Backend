"""
Department Routing & SLA Management Agent for WardMitra AI / PCMC Sarathi.
Assigns tickets to responsible PCMC departments, calculates SLA deadlines,
and generates unique standardized PCMC Ticket IDs.
"""

import yaml
import random
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from app.config.settings import settings
from app.rules.department_rules import get_department_routing

logger = logging.getLogger("pcms.routing_agent")


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
            logger.warning(f"[RoutingAgent] Could not load {settings.MODEL_CONFIG_PATH}: {e}")

    def route_complaint(self, category: str) -> Dict[str, Any]:
        """
        Determines responsible department, SLA hours, deadline, and localized details.
        Uses configurable department rules (Requirement 11).
        """
        rule_info = get_department_routing(category)
        dept_code = rule_info["department_code"]
        sla_hours = rule_info["sla_hours"]
        dept_name = rule_info["department_name"]
        dept_name_mr = rule_info["department_name_mr"]

        cat_info = self.categories_config.get(category, {})
        dept_info = self.departments_config.get(dept_code, {})
        deadline = datetime.now(timezone.utc) + timedelta(hours=sla_hours)

        return {
            "department": dept_code,
            "department_name": dept_name,
            "department_name_mr": dept_name_mr,
            "department_name_en": cat_info.get("department_name_en", dept_name),
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
