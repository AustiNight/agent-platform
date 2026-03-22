"""
Task management endpoints.

Handles creating, querying, and managing agent tasks.
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from src.core.schemas import (
    CreateTaskRequest,
    TaskResponse,
    TaskStatus,
    TaskStatusResponse,
    TaskConfig,
    AgentType,
)
from src.db.models import Task, TaskMessage
from src.db.session import get_db_session
from src.gateway.services.task_executor import execute_task_async

logger = structlog.get_logger()
router = APIRouter()


@router.post("", response_model=TaskStatusResponse)
async def create_task(
    request: CreateTaskRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
) -> TaskStatusResponse:
    """
    Create a new task.
    
    If `wait=true`, blocks until task completes.
    Otherwise, returns immediately with task ID for polling.
    """
    task_id = str(uuid4())
    
    # Create task record
    task = Task(
        id=task_id,
        agent_type=request.agent_type.value,
        instructions=request.instructions,
        config=request.config.model_dump(),
        context_data=request.context,
        parent_task_id=str(request.parent_task_id) if request.parent_task_id else None,
        status="pending",
    )
    
    db.add(task)
    await db.commit()
    await db.refresh(task)
    
    logger.info(
        "task_created",
        task_id=task_id,
        agent_type=request.agent_type.value,
    )
    
    if request.wait:
        # Execute synchronously
        result = await execute_task_async(task_id)
        
        # Refresh task from DB
        await db.refresh(task)
        
        return TaskStatusResponse(
            task_id=UUID(task.id),
            status=TaskStatus(task.status),
            agent_type=AgentType(task.agent_type),
            created_at=task.created_at,
            updated_at=task.updated_at,
            result=task.result,
            error=task.error,
        )
    else:
        # Execute in background
        background_tasks.add_task(execute_task_async, task_id)
        
        return TaskStatusResponse(
            task_id=UUID(task.id),
            status=TaskStatus(task.status),
            agent_type=AgentType(task.agent_type),
            created_at=task.created_at,
            updated_at=task.updated_at,
        )


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task(
    task_id: UUID,
    db: AsyncSession = Depends(get_db_session),
) -> TaskStatusResponse:
    """Get task status and results."""
    result = await db.execute(
        select(Task).where(Task.id == str(task_id))
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskStatusResponse(
        task_id=UUID(task.id),
        status=TaskStatus(task.status),
        agent_type=AgentType(task.agent_type),
        created_at=task.created_at,
        updated_at=task.updated_at,
        result=task.result,
        error=task.error,
        progress=task.progress,
    )


@router.get("/{task_id}/full", response_model=TaskResponse)
async def get_task_full(
    task_id: UUID,
    db: AsyncSession = Depends(get_db_session),
) -> TaskResponse:
    """Get full task details including message history and artifacts."""
    result = await db.execute(
        select(Task).where(Task.id == str(task_id))
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Get messages
    msg_result = await db.execute(
        select(TaskMessage)
        .where(TaskMessage.task_id == str(task_id))
        .order_by(TaskMessage.created_at)
    )
    messages = msg_result.scalars().all()
    
    # Convert to response format
    from src.core.schemas import Message, MessageRole, ToolCall, ToolResult
    
    message_list = []
    for msg in messages:
        message_list.append(Message(
            role=MessageRole(msg.role),
            content=msg.content,
            tool_calls=[ToolCall(**tc) for tc in (msg.tool_calls or [])],
            tool_results=[ToolResult(**tr) for tr in (msg.tool_results or [])],
            timestamp=msg.created_at,
            token_count=msg.token_count,
        ))
    
    return TaskResponse(
        task_id=UUID(task.id),
        agent_type=AgentType(task.agent_type),
        status=TaskStatus(task.status),
        instructions=task.instructions,
        result=task.result,
        error=task.error,
        messages=message_list,
        artifacts=task.artifacts or [],
        started_at=task.started_at,
        completed_at=task.completed_at,
        llm_calls=task.llm_calls,
        total_tokens=task.total_tokens,
        cost_cents=task.cost_cents,
    )


@router.get("")
async def list_tasks(
    status: TaskStatus | None = None,
    agent_type: AgentType | None = None,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """List tasks with optional filtering."""
    query = select(Task)
    count_query = select(func.count(Task.id))
    
    # Apply filters
    if status:
        query = query.where(Task.status == status.value)
        count_query = count_query.where(Task.status == status.value)
    if agent_type:
        query = query.where(Task.agent_type == agent_type.value)
        count_query = count_query.where(Task.agent_type == agent_type.value)
    
    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination
    query = query.order_by(Task.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    tasks = result.scalars().all()
    
    return {
        "items": [
            TaskStatusResponse(
                task_id=UUID(t.id),
                status=TaskStatus(t.status),
                agent_type=AgentType(t.agent_type),
                created_at=t.created_at,
                updated_at=t.updated_at,
                result=t.result,
                error=t.error,
                progress=t.progress,
            )
            for t in tasks
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
    }


@router.post("/{task_id}/cancel")
async def cancel_task(
    task_id: UUID,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Cancel a pending or running task."""
    result = await db.execute(
        select(Task).where(Task.id == str(task_id))
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status not in ["pending", "running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task with status: {task.status}",
        )
    
    task.status = "cancelled"
    task.completed_at = datetime.utcnow()
    await db.commit()
    
    logger.info("task_cancelled", task_id=str(task_id))
    
    return {"status": "cancelled", "task_id": str(task_id)}


@router.delete("/{task_id}")
async def delete_task(
    task_id: UUID,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Delete a task and its history."""
    result = await db.execute(
        select(Task).where(Task.id == str(task_id))
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    await db.delete(task)
    await db.commit()
    
    logger.info("task_deleted", task_id=str(task_id))
    
    return {"status": "deleted", "task_id": str(task_id)}
