"""
Health check endpoints.
"""

from datetime import datetime
from pathlib import Path
import json

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db_session

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Basic health check."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.get("/health/ready")
async def readiness_check(
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """
    Readiness check including database connectivity.
    """
    checks = {
        "database": False,
    }
    
    # Check database
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as e:
        checks["database_error"] = str(e)
    
    all_healthy = all(v is True for v in checks.values() if isinstance(v, bool))
    
    return {
        "status": "ready" if all_healthy else "degraded",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/gepa")
async def gepa_health_check() -> dict:
    """
    GEPA automation health check.
    
    Returns status of the optimization system.
    """
    result = {
        "status": "unknown",
        "automation_running": False,
        "traces_collected": 0,
        "optimizations_completed": 0,
        "pending_escalations": 0,
        "last_optimization": None,
        "agent_scores": {},
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    try:
        # Check state file
        state_file = Path("./data/gepa/auto_state.json")
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            
            result["automation_running"] = True
            result["traces_collected"] = sum(state.get("trace_counts", {}).values())
            result["optimizations_completed"] = len(state.get("last_optimization", {}))
            result["pending_escalations"] = len(state.get("pending_escalations", []))
            result["agent_scores"] = state.get("current_scores", {})
            
            # Find most recent optimization
            last_opts = state.get("last_optimization", {})
            if last_opts:
                result["last_optimization"] = max(last_opts.values())
        
        # Check traces directory
        traces_dir = Path("./data/gepa/traces")
        if traces_dir.exists():
            result["traces_collected"] = len(list(traces_dir.glob("*.json")))
        
        # Check for pending notifications
        notifications_dir = Path("./data/gepa/notifications")
        urgent_marker = notifications_dir / "URGENT_NOTIFICATION_PENDING"
        if urgent_marker.exists():
            result["pending_escalations"] += 1
            result["status"] = "needs_attention"
        elif result["automation_running"]:
            result["status"] = "healthy"
        else:
            result["status"] = "not_running"
            
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result
