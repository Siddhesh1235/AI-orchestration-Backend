"""
Main Application Entrypoint for PCMC Sarathi AI Orchestrator.
Configures FastAPI, SQLite ORM auto-migrations, Static Frontend mounts, and CORS.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.database.session import engine, Base
from app.database import models
from app.api.routes import complaint, health, voice, rag
from app.services.scheduler_service import start_auto_escalation_scheduler, stop_auto_escalation_scheduler

# 1. Initialize SQLite Database Tables on startup
Base.metadata.create_all(bind=engine)

def _safe_migrate_db():
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            res = conn.execute(text("PRAGMA table_info(complaints)")).fetchall()
            cols = {r[1] for r in res}
            new_cols = [
                ("severity", "VARCHAR(32) DEFAULT 'MEDIUM'"),
                ("is_emergency", "BOOLEAN DEFAULT 0"),
                ("emergency_level", "VARCHAR(32) DEFAULT 'NONE'"),
                ("verification_status", "VARCHAR(32) DEFAULT 'APPROVED'"),
                ("fraud_score", "FLOAT DEFAULT 0.0"),
                ("evidence_valid", "BOOLEAN DEFAULT 1"),
                ("video_path", "VARCHAR(255)"),
                ("reopen_reason", "TEXT"),
            ]
            for col_name, col_type in new_cols:
                if col_name not in cols:
                    conn.execute(text(f"ALTER TABLE complaints ADD COLUMN {col_name} {col_type}"))
            conn.commit()
    except Exception as e:
        import logging
        logging.getLogger("pcms.main").warning(f"DB auto-migration notice: {e}")

_safe_migrate_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and clean background daemon shutdown."""
    # Startup: launch APScheduler for automated SLA escalation
    start_auto_escalation_scheduler(interval_seconds=60)
    yield
    # Shutdown: terminate scheduler cleanly
    stop_auto_escalation_scheduler()


# 2. FastAPI Application Instance
app = FastAPI(
    title="Ward Mitra - Grievance Orchestrator",
    description="Backend AI Orchestrator powering Ward Mitra Grievance Platform (Ward-Centric Architecture)",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 3. CORS Middleware (Permit browser testing from any port/client)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to prevent browser caching of static files (ensures fresh JS/CSS)
@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# 4. Mount Uploads and Static Directories
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

upload_dir = Path(settings.UPLOAD_DIR)
upload_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(upload_dir)), name="uploads")

# 5. Include API Routers
app.include_router(complaint.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")
app.include_router(voice.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")

# Direct alias for /api/v1/chat/complaint
app.add_api_route("/api/v1/chat/complaint", complaint.chat_with_bot, methods=["POST"], tags=["Grievance Redressal"])


# 6. Root Route -> Redirect to Citizen Mobile Chatbot UI
@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirects default root URL directly to the Citizen Mobile Chatbot UI."""
    return RedirectResponse(url="/static/chat.html")


@app.get("/chat", include_in_schema=False)
def chat_redirect():
    """Direct URL to the Citizen Mobile Chatbot UI."""
    return RedirectResponse(url="/static/chat.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
