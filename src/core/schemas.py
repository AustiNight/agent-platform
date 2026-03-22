"""
Shared schemas for the agent platform.

These define the common interface that all agents and the gateway use.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class AgentType(str, Enum):
    """Available agent types."""

    BROWSER = "browser"
    RESEARCH = "research"
    WRITER = "writer"
    SECURITY = "security"
    SCRAPER = "scraper"
    FIXER = "fixer"


class TaskStatus(str, Enum):
    """Task lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_HUMAN = "waiting_human"  # Paused for human approval
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageRole(str, Enum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# =============================================================================
# Core Message Types
# =============================================================================


class ToolCall(BaseModel):
    """A tool invocation by the agent."""

    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of a tool execution."""

    tool_call_id: str
    success: bool
    result: Any = None
    error: str | None = None
    screenshot_path: str | None = None  # For browser agent


class Message(BaseModel):
    """A single message in the agent conversation."""

    role: MessageRole
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_count: int | None = None


# =============================================================================
# Task Request/Response
# =============================================================================


class TaskConfig(BaseModel):
    """Agent-specific configuration passed with a task."""

    # Common settings
    timeout_seconds: int = Field(default=300, ge=10, le=3600)
    max_llm_calls: int = Field(default=20, ge=1, le=100)

    # LLM settings (override defaults)
    llm_provider: str | None = None
    llm_model: str | None = None

    # Browser-specific
    headless: bool | None = None
    viewport_width: int = 1280
    viewport_height: int = 720

    # Extensible for other agents
    extra: dict[str, Any] = Field(default_factory=dict)


class TaskRequest(BaseModel):
    """Request to execute an agent task."""

    agent_type: AgentType
    instructions: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Natural language instructions for the agent",
    )
    config: TaskConfig = Field(default_factory=TaskConfig)

    # Optional context from previous tasks
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (e.g., data from previous tasks)",
    )

    # For chained/orchestrated tasks
    parent_task_id: UUID | None = None


class TaskArtifact(BaseModel):
    """An artifact produced by a task (screenshot, file, etc.)."""

    type: str  # screenshot, file, data
    path: str | None = None  # For files
    data: Any | None = None  # For inline data
    description: str | None = None


class TaskResponse(BaseModel):
    """Response from a completed task."""

    task_id: UUID = Field(default_factory=uuid4)
    agent_type: AgentType
    status: TaskStatus
    instructions: str

    # Results
    result: Any | None = None
    error: str | None = None

    # Conversation history
    messages: list[Message] = Field(default_factory=list)

    # Artifacts (screenshots, extracted data, etc.)
    artifacts: list[TaskArtifact] = Field(default_factory=list)

    # Metrics
    started_at: datetime | None = None
    completed_at: datetime | None = None
    llm_calls: int = 0
    total_tokens: int = 0
    cost_cents: float = 0.0

    @property
    def duration_seconds(self) -> float | None:
        """Calculate task duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


# =============================================================================
# API Request/Response Wrappers
# =============================================================================


class CreateTaskRequest(BaseModel):
    """API request to create a new task."""

    agent_type: AgentType
    instructions: str
    config: TaskConfig = Field(default_factory=TaskConfig)
    context: dict[str, Any] = Field(default_factory=dict)
    parent_task_id: UUID | None = None
    
    # Whether to wait for completion or return immediately
    wait: bool = Field(
        default=False,
        description="If true, wait for task completion before returning",
    )


class TaskStatusResponse(BaseModel):
    """API response for task status queries."""

    task_id: UUID
    status: TaskStatus
    agent_type: AgentType
    created_at: datetime
    updated_at: datetime
    result: Any | None = None
    error: str | None = None
    progress: str | None = None  # Human-readable progress update


class PaginatedResponse(BaseModel):
    """Paginated list response."""

    items: list[Any]
    total: int
    page: int
    page_size: int
    pages: int
