"""
Task execution service.

Handles running agent tasks and persisting results.
Automatically collects GEPA traces for continuous optimization.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import select
import structlog

from src.agents import get_agent_class
from src.core.base_agent import AgentContext
from src.core.llm_client import LLMClient
from src.core.schemas import TaskConfig, TaskResponse, AgentType
from src.db.models import Task, TaskMessage
from src.db.session import get_session

logger = structlog.get_logger()


async def execute_task_async(task_id: str) -> TaskResponse:
    """
    Execute a task by ID.
    
    Loads task from database, runs the appropriate agent,
    and persists results. Automatically collects GEPA traces.
    """
    logger.info("executing_task", task_id=task_id)
    
    async with get_session() as db:
        # Load task
        result = await db.execute(
            select(Task).where(Task.id == task_id)
        )
        task = result.scalar_one_or_none()
        
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        # Update status to running
        task.status = "running"
        task.started_at = datetime.utcnow()
        await db.commit()
        
        try:
            # Get agent class and load optimized prompts
            agent_class = get_agent_class(task.agent_type)
            agent = agent_class()
            
            # Load GEPA-optimized prompts if available
            await _load_optimized_prompts(agent, task.agent_type)
            
            # Create LLM client
            config = TaskConfig(**task.config)
            llm_client = LLMClient(
                provider=config.llm_provider,
                model=config.llm_model,
            )
            
            # Create execution context
            context = AgentContext(
                task_id=task_id,
                instructions=task.instructions,
                config=config,
                llm_client=llm_client,
                context_data=task.context_data or {},
            )
            
            # Run agent
            response = await agent.run(context)
            
            # Persist messages
            for msg in response.messages:
                db_msg = TaskMessage(
                    task_id=task_id,
                    role=msg.role.value,
                    content=msg.content,
                    tool_calls=[tc.model_dump() for tc in (msg.tool_calls or [])],
                    tool_results=[tr.model_dump() for tr in (msg.tool_results or [])],
                    token_count=msg.token_count,
                    created_at=msg.timestamp,
                )
                db.add(db_msg)
            
            # Update task with results
            task.status = response.status.value
            task.result = response.result
            task.error = response.error
            task.artifacts = [a.model_dump() for a in response.artifacts]
            task.llm_calls = response.llm_calls
            task.total_tokens = response.total_tokens
            task.cost_cents = response.cost_cents
            task.completed_at = datetime.utcnow()
            
            # Store system prompt used (for GEPA trace)
            if not task.config:
                task.config = {}
            task.config["system_prompt_used"] = agent.system_prompt
            
            await db.commit()
            
            logger.info(
                "task_completed",
                task_id=task_id,
                status=response.status.value,
                llm_calls=response.llm_calls,
            )
            
            # Collect GEPA trace asynchronously (don't block response)
            await _collect_gepa_trace(task, response, agent, context)
            
            return response
            
        except Exception as e:
            logger.exception("task_failed", task_id=task_id, error=str(e))
            
            # Update task with error
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            await db.commit()
            
            # Still collect trace for failed tasks (important for GEPA learning)
            try:
                await _collect_gepa_trace_failure(task, e)
            except Exception as trace_err:
                logger.warning("gepa_trace_collection_failed", error=str(trace_err))
            
            raise


async def _load_optimized_prompts(agent: Any, agent_type: str) -> None:
    """Load GEPA-optimized prompts into the agent."""
    try:
        from src.core.gepa.middleware import get_optimized_prompt
        
        # Try to get optimized system prompt
        optimized = get_optimized_prompt(agent_type, "system_prompt")
        if optimized:
            agent.load_optimized_prompts({"system_prompt": optimized})
            logger.debug("loaded_optimized_prompt", agent_type=agent_type)
    except Exception as e:
        # Non-fatal - use default prompts
        logger.debug("optimized_prompt_not_available", agent_type=agent_type, error=str(e))


async def _collect_gepa_trace(
    task: Task,
    response: TaskResponse,
    agent: Any,
    context: AgentContext,
) -> None:
    """Collect execution trace for GEPA optimization."""
    try:
        from src.core.gepa import ExecutionTrace, OptimizationStore
        
        store = OptimizationStore("./data/gepa")
        
        # Build step list from messages
        steps = []
        for msg in response.messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    steps.append({
                        "step_num": len(steps) + 1,
                        "type": tc.name,
                        "input": tc.arguments,
                        "output": None,  # Will be filled from tool_results
                        "success": True,
                        "error": None,
                        "metadata": {"tool_id": tc.id},
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    })
            
            if msg.tool_results:
                for tr in msg.tool_results:
                    # Find matching step and update
                    for step in steps:
                        if step["metadata"].get("tool_id") == tr.tool_use_id:
                            step["output"] = tr.content[:1000] if tr.content else None
                            step["success"] = tr.is_success
                            if not tr.is_success:
                                step["error"] = tr.content[:500] if tr.content else "Unknown error"
                            break
        
        # Create trace
        trace = ExecutionTrace(
            task_id=task.id,
            agent_type=task.agent_type,
            instructions=task.instructions,
            system_prompt=agent.system_prompt,
            steps=steps,
            success=response.status.value == "completed",
            result=response.result,
            error=response.error,
            llm_calls=response.llm_calls,
            total_tokens=response.total_tokens,
            started_at=task.started_at or task.created_at,
            completed_at=task.completed_at or datetime.utcnow(),
        )
        
        if trace.completed_at and trace.started_at:
            trace.duration_seconds = (trace.completed_at - trace.started_at).total_seconds()
        
        # Save trace
        store.save_trace(trace)
        
        # Update GEPA state counter
        try:
            from src.core.gepa.automation import AutoGEPAConfig, StateManager
            config = AutoGEPAConfig.from_env()
            state_manager = StateManager(config)
            count = state_manager.record_trace(task.agent_type)
            logger.debug("gepa_trace_recorded", task_id=task.id, trace_count=count)
        except Exception as e:
            logger.debug("gepa_state_update_skipped", error=str(e))
        
    except Exception as e:
        logger.warning("gepa_trace_collection_error", task_id=task.id, error=str(e))


async def _collect_gepa_trace_failure(task: Task, error: Exception) -> None:
    """Collect trace for a failed task."""
    try:
        from src.core.gepa import ExecutionTrace, OptimizationStore
        
        store = OptimizationStore("./data/gepa")
        
        trace = ExecutionTrace(
            task_id=task.id,
            agent_type=task.agent_type,
            instructions=task.instructions,
            system_prompt=task.config.get("system_prompt_used", "") if task.config else "",
            steps=[{
                "step_num": 1,
                "type": "execution_error",
                "input": task.instructions,
                "output": None,
                "success": False,
                "error": str(error),
                "metadata": {"exception_type": type(error).__name__},
                "timestamp": datetime.utcnow().isoformat(),
            }],
            success=False,
            result=None,
            error=str(error),
            llm_calls=0,
            total_tokens=0,
            started_at=task.started_at or task.created_at,
            completed_at=datetime.utcnow(),
        )
        
        store.save_trace(trace)
        
    except Exception as e:
        logger.warning("gepa_failure_trace_error", task_id=task.id, error=str(e))


async def update_task_progress(task_id: str, progress: str) -> None:
    """Update task progress message."""
    async with get_session() as db:
        result = await db.execute(
            select(Task).where(Task.id == task_id)
        )
        task = result.scalar_one_or_none()
        
        if task:
            task.progress = progress
            await db.commit()
