"""
GEPA Trace Collection Middleware.

Automatically collects execution traces from all agent task completions.
This enables fully automated GEPA optimization without manual intervention.
"""

import asyncio
from datetime import datetime
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger()


class GEPATraceMiddleware(BaseHTTPMiddleware):
    """
    Middleware that hooks into task completions to collect traces.
    
    This runs after every task completion and saves the trace
    for GEPA optimization.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        response = await call_next(request)
        
        # Only process task-related endpoints
        if "/api/v1/tasks" not in request.url.path:
            return response
        
        # Don't process GET requests (just queries)
        if request.method == "GET":
            return response
        
        # Schedule trace collection (don't block response)
        asyncio.create_task(self._collect_trace_background())
        
        return response
    
    async def _collect_trace_background(self) -> None:
        """Background task to collect traces."""
        try:
            from src.core.gepa.auto_loop import GEPAAutomationLoop
            
            loop = GEPAAutomationLoop()
            await loop._collect_traces()
        except Exception as e:
            logger.warning("trace_collection_background_failed", error=str(e))


class GEPAPromptLoader:
    """
    Loads optimized prompts into agents at startup.
    
    Call this during application startup to ensure agents
    use the latest optimized prompts.
    """
    
    @staticmethod
    async def load_all() -> dict[str, bool]:
        """
        Load optimized prompts for all registered agents.
        
        Returns dict of agent_type -> success status.
        """
        from src.agents import AGENT_REGISTRY
        from src.core.gepa import OptimizationStore
        from pathlib import Path
        import json
        
        results = {}
        store = OptimizationStore("./data/gepa")
        
        # Load active prompts configuration
        active_prompts_file = store.storage_dir / "active_prompts.json"
        
        if not active_prompts_file.exists():
            logger.info("no_active_prompts_file")
            return {agent_type: False for agent_type in AGENT_REGISTRY}
        
        with open(active_prompts_file) as f:
            active = json.load(f)
        
        for agent_type, agent_class in AGENT_REGISTRY.items():
            try:
                if agent_type in active:
                    agent_config = active[agent_type]
                    
                    # Store prompts in a way agents can access
                    # We'll use a module-level cache
                    _OPTIMIZED_PROMPTS[agent_type] = agent_config["prompts"]
                    
                    logger.info(
                        "loaded_optimized_prompts",
                        agent_type=agent_type,
                        score=agent_config.get("score"),
                        updated_at=agent_config.get("updated_at"),
                    )
                    results[agent_type] = True
                else:
                    results[agent_type] = False
                    
            except Exception as e:
                logger.warning(
                    "prompt_load_failed",
                    agent_type=agent_type,
                    error=str(e),
                )
                results[agent_type] = False
        
        return results


# Module-level cache for optimized prompts
_OPTIMIZED_PROMPTS: dict[str, dict[str, str]] = {}


def get_optimized_prompt(agent_type: str, component: str = "system_prompt") -> str | None:
    """
    Get the optimized prompt for an agent.
    
    Called by BaseAgent.system_prompt to get optimized version.
    """
    if agent_type in _OPTIMIZED_PROMPTS:
        return _OPTIMIZED_PROMPTS[agent_type].get(component)
    return None


async def start_gepa_automation_background() -> None:
    """
    Start the GEPA automation loop in the background.
    
    Call this from the FastAPI lifespan handler.
    """
    import asyncio
    from src.core.gepa.auto_loop import GEPAAutomationLoop
    
    async def run_loop():
        try:
            loop = GEPAAutomationLoop()
            await loop.start()
        except Exception as e:
            logger.error("gepa_automation_crashed", error=str(e))
    
    # Start in background
    asyncio.create_task(run_loop())
    logger.info("gepa_automation_started_background")
