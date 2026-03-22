"""
Database models for task persistence.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Task(Base):
    """
    Persistent task record.
    
    Stores task requests, status, results, and metrics.
    """
    
    __tablename__ = "tasks"

    # Use String for SQLite compatibility, UUID for Postgres
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Task definition
    agent_type = Column(String(50), nullable=False, index=True)
    instructions = Column(Text, nullable=False)
    config = Column(JSON, nullable=False, default=dict)
    context_data = Column(JSON, nullable=False, default=dict)
    
    # Status tracking
    status = Column(
        String(20),
        nullable=False,
        default="pending",
        index=True,
    )
    progress = Column(String(255), nullable=True)
    
    # Results
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    artifacts = Column(JSON, nullable=False, default=list)
    
    # Relationships
    parent_task_id = Column(String(36), ForeignKey("tasks.id"), nullable=True)
    parent_task = relationship("Task", remote_side=[id], backref="subtasks")
    messages = relationship("TaskMessage", back_populates="task", cascade="all, delete-orphan")
    
    # Metrics
    llm_calls = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    cost_cents = Column(Float, nullable=False, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def __repr__(self) -> str:
        return f"<Task {self.id[:8]} {self.agent_type} {self.status}>"

    @property
    def duration_seconds(self) -> float | None:
        """Calculate task duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class TaskMessage(Base):
    """
    Individual message in a task's conversation history.
    """
    
    __tablename__ = "task_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False, index=True)
    
    # Message content
    role = Column(String(20), nullable=False)  # system, user, assistant, tool
    content = Column(Text, nullable=False, default="")
    
    # Tool-related fields
    tool_calls = Column(JSON, nullable=True)  # For assistant messages with tool calls
    tool_results = Column(JSON, nullable=True)  # For tool result messages
    
    # Metadata
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    task = relationship("Task", back_populates="messages")

    def __repr__(self) -> str:
        return f"<TaskMessage {self.id} {self.role}>"
