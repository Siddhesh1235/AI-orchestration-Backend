"""
Auto-Escalation Scheduler Service for PCMC Sarathi AI.
Uses APScheduler (AsyncIOScheduler) to automatically monitor active grievances.
When SLA threshold expires without field action, it escalates sequentially:
  Level 1 (Ward Worker) -> Level 2 (Supervisor) -> Level 3 (HOD).
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.database.session import SessionLocal
from app.database.models import Complaint, ComplaintStatus, EscalationLevel
from app.services.ward_service import ward_service
from app.agents.severity_agent import severity_agent

logger = logging.getLogger("pcms.scheduler")

scheduler: AsyncIOScheduler = AsyncIOScheduler()


def run_auto_escalation_cycle() -> Dict[str, Any]:
    """
    Executes a single scan of unresolved grievances and auto-escalates SLA-breached tickets.
    """
    db = SessionLocal()
    try:
        active_complaints = db.query(Complaint).filter(
            Complaint.status.in_([ComplaintStatus.REGISTERED, ComplaintStatus.ASSIGNED]),
            Complaint.escalation_level != EscalationLevel.LEVEL_3_HOD,
            Complaint.is_fraud == False
        ).all()

        now = datetime.now(timezone.utc)
        escalated_list: List[Dict[str, Any]] = []

        for comp in active_complaints:
            sched = severity_agent.get_escalation_schedule(comp.priority, comp.sla_hours)
            created_time = comp.created_at
            if created_time.tzinfo is None:
                created_time = created_time.replace(tzinfo=timezone.utc)

            elapsed_hours = (now - created_time).total_seconds() / 3600.0

            should_escalate = False
            reason = ""

            if comp.escalation_level == EscalationLevel.LEVEL_1_WORKER and elapsed_hours >= sched["worker_threshold_hours"]:
                should_escalate = True
                reason = f"क्षेत्रीय कामगाराने विहित मुदतीत ({sched['worker_threshold_hours']} तास) कार्यवाही न केल्याने आपोआप वर्ग झाले."
            elif comp.escalation_level == EscalationLevel.LEVEL_2_SUPERVISOR and elapsed_hours >= (sched["worker_threshold_hours"] + sched["supervisor_threshold_hours"]):
                should_escalate = True
                reason = f"पर्यवेक्षक स्तरावर विहित मुदतीत कार्यवाही न झाल्याने आपोआप HOD कडे वर्ग झाले."

            if should_escalate:
                res = ward_service.escalate_complaint(db, comp, reason=reason)
                escalated_list.append(res)
                logger.info(f"[Scheduler] Auto-escalated ticket {comp.ticket_id} to {res.get('to_level')}")

        return {
            "timestamp": now.isoformat(),
            "scanned": len(active_complaints),
            "escalated": len(escalated_list),
            "details": escalated_list
        }

    except Exception as e:
        logger.error(f"[Scheduler] Error during auto-escalation cycle: {e}")
        return {"error": str(e), "scanned": 0, "escalated": 0}
    finally:
        db.close()


async def scheduled_escalation_job():
    """Async wrapper for the scheduled background escalation job."""
    run_auto_escalation_cycle()


def start_auto_escalation_scheduler(interval_seconds: int = 60):
    """
    Starts the APScheduler background daemon running every interval_seconds.
    """
    if not scheduler.running:
        scheduler.add_job(
            scheduled_escalation_job,
            trigger=IntervalTrigger(seconds=interval_seconds),
            id="pcmc_auto_escalation_job",
            name="PCMC Auto-Escalation SLA Monitor",
            replace_existing=True
        )
        scheduler.start()
        logger.info(f"[Scheduler] APScheduler started successfully (Interval: {interval_seconds}s).")


def stop_auto_escalation_scheduler():
    """
    Cleanly shuts down the scheduler.
    """
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("[Scheduler] APScheduler stopped.")
