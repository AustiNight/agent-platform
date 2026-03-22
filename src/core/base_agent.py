"""
Base agent interface that all specialist agents must implement.

This defines the common contract for agent development.
All agents automatically support GEPA optimization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from src.core.schemas import (
    Message,
    MessageRole,
    TaskConfig,
    TaskResponse,
    TaskStatus,
    TaskArtifact,
    ToolCall,
    ToolResult,
    AgentType,
)
from src.core.llm_client import LLMClient

logger = structlog.get_logger()


class AgentCapability(str, Enum):
    """Capabilities that agents can declare."""

    WEB_BROWSING = "web_browsing"
    FILE_ACCESS = "file_access"
    CODE_EXECUTION = "code_execution"
    WEB_SEARCH = "web_search"
    API_ACCESS = "api_access"
    HUMAN_INTERACTION = "human_interaction"


@dataclass
class AgentContext:
    """
    Runtime context passed to agents during execution.
    
    Contains everything an agent needs to do its work.
    """

    task_id: str
    instructions: str
    config: TaskConfig
    llm_client: LLMClient
    context_data: dict[str, Any] = field(default_factory=dict)

    # Runtime state
    messages: list[Message] = field(default_factory=list)
    artifacts: list[TaskArtifact] = field(default_factory=list)
    llm_calls: int = 0
    total_tokens: int = 0

    def add_message(
        self,
        role: MessageRole,
        content: str,
        tool_calls: list[ToolCall] | None = None,
        tool_results: list[ToolResult] | None = None,
        token_count: int | None = None,
    ) -> Message:
        """Add a message to the conversation history."""
        message = Message(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_results=tool_results,
            token_count=token_count,
        )
        self.messages.append(message)
        return message

    def add_artifact(
        self,
        artifact_type: str,
        path: str | None = None,
        data: Any | None = None,
        description: str | None = None,
    ) -> TaskArtifact:
        """Add an artifact to the task output."""
        artifact = TaskArtifact(
            type=artifact_type,
            path=path,
            data=data,
            description=description,
        )
        self.artifacts.append(artifact)
        return artifact


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Implement this interface to create a new specialist agent.
    All agents automatically support GEPA optimization through
    the OptimizableComponent interface.
    """

    # Tracks optimized prompts (can be loaded from storage)
    _optimized_prompts: dict[str, str]

    def __init__(self) -> None:
        self.logger = structlog.get_logger().bind(agent=self.agent_type.value)
        self._optimized_prompts = {}
        
        # GEPA trace collection
        self._gepa_trace: Any = None  # Will be ExecutionTrace when gepa is imported

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the agent type identifier."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this agent does."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> list[AgentCapability]:
        """List of capabilities this agent has."""
        ...

    @property
    @abstractmethod
    def tools(self) -> list[dict[str, Any]]:
        """
        Tool definitions for the LLM.
        
        Returns a list of tool schemas in the format expected by the LLM provider.
        Example:
        [
            {
                "name": "click",
                "description": "Click on an element",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector"}
                    },
                    "required": ["selector"]
                }
            }
        ]
        """
        ...

    @property
    def system_prompt(self) -> str:
        """
        System prompt for the agent.
        
        Returns optimized prompt if available, otherwise default.
        Override this to customize the agent's persona and instructions.
        """
        # Check for optimized version first
        if "system_prompt" in self._optimized_prompts:
            return self._optimized_prompts["system_prompt"]
        
        return self._default_system_prompt
    
    @property
    def _default_system_prompt(self) -> str:
        """Default system prompt before optimization."""
        return f"""You are a specialized AI agent: {self.description}

Your task is to help the user by using the tools available to you.
Think step by step about how to accomplish the user's request.
After each action, observe the result and decide what to do next.
When you have completed the task or cannot proceed, report your findings.

Be precise and efficient. Explain what you're doing at each step.
If you encounter an error, try to recover or explain what went wrong."""

    # =========================================================================
    # OptimizableComponent Interface (for GEPA)
    # =========================================================================

    @property
    def optimizable_text(self) -> dict[str, str]:
        """
        Return text components that can be optimized by GEPA.
        
        Override to add more optimizable components.
        """
        return {
            "system_prompt": self.system_prompt,
        }
    
    def set_optimized_text(self, component_name: str, text: str) -> None:
        """
        Update an optimizable component with new text.
        
        Called by GEPA after optimization.
        """
        self._optimized_prompts[component_name] = text
        self.logger.info(
            "prompt_optimized",
            component=component_name,
            text_length=len(text),
        )
    
    def load_optimized_prompts(self, prompts: dict[str, str]) -> None:
        """Load previously optimized prompts."""
        self._optimized_prompts.update(prompts)
        self.logger.info(
            "prompts_loaded",
            components=list(prompts.keys()),
        )
    
    async def execute_for_optimization(
        self,
        task: dict[str, Any],
    ) -> tuple[dict[str, Any], Any]:
        """
        Execute a task for GEPA optimization.
        
        Returns result and execution trace.
        """
        from src.core.gepa import ExecutionTrace
        
        # Create trace
        trace = ExecutionTrace(
            task_id=task.get("task_id", "opt-task"),
            agent_type=self.agent_type.value,
            instructions=task["instructions"],
            system_prompt=self.system_prompt,
        )
        self._gepa_trace = trace
        
        # Create context and run
        llm_client = LLMClient()
        context = AgentContext(
            task_id=trace.task_id,
            instructions=task["instructions"],
            config=TaskConfig(**task.get("config", {})),
            llm_client=llm_client,
            context_data=task.get("context", {}),
        )
        
        try:
            response = await self.run(context)
            
            trace.success = response.status == TaskStatus.COMPLETED
            trace.result = response.result
            trace.error = response.error
            trace.llm_calls = response.llm_calls
            trace.total_tokens = response.total_tokens
            trace.completed_at = datetime.utcnow()
            trace.duration_seconds = (
                trace.completed_at - trace.started_at
            ).total_seconds()
            
            return {
                "success": trace.success,
                "result": response.result,
                "error": response.error,
                "extracted_data": (
                    response.result.get("extracted_data")
                    if isinstance(response.result, dict) else None
                ),
            }, trace
            
        except Exception as e:
            trace.success = False
            trace.error = str(e)
            trace.completed_at = datetime.utcnow()
            
            return {
                "success": False,
                "error": str(e),
            }, trace
        finally:
            self._gepa_trace = None

    @abstractmethod
    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: AgentContext,
    ) -> ToolResult:
        """
        Execute a tool call.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            context: Current agent context
            
        Returns:
            ToolResult with success status and result/error
        """
        ...

    async def run(self, context: AgentContext) -> TaskResponse:
        """
        Main execution loop for the agent.
        
        This implements the standard agent loop:
        1. Send messages to LLM
        2. If LLM wants to use tools, execute them
        3. Send tool results back to LLM
        4. Repeat until done or limit reached
        
        Override this for custom execution logic.
        """
        started_at = datetime.utcnow()
        self.logger.info("starting_agent", task_id=context.task_id)

        try:
            # Initialize conversation with system prompt and user instructions
            context.add_message(MessageRole.SYSTEM, self.system_prompt)
            context.add_message(MessageRole.USER, context.instructions)

            # Agent loop
            while context.llm_calls < context.config.max_llm_calls:
                # Call LLM
                response = await self._call_llm(context)
                context.llm_calls += 1

                if response.get("stop_reason") == "end_turn":
                    # Agent is done
                    final_content = response.get("content", "Task completed.")
                    context.add_message(MessageRole.ASSISTANT, final_content)
                    
                    return self._build_response(
                        context=context,
                        status=TaskStatus.COMPLETED,
                        result=final_content,
                        started_at=started_at,
                    )

                elif response.get("stop_reason") == "tool_use":
                    # Process tool calls
                    tool_calls = response.get("tool_calls", [])
                    assistant_content = response.get("content", "")
                    
                    # Record assistant message with tool calls
                    context.add_message(
                        MessageRole.ASSISTANT,
                        assistant_content,
                        tool_calls=[
                            ToolCall(
                                id=tc["id"],
                                name=tc["name"],
                                arguments=tc["arguments"],
                            )
                            for tc in tool_calls
                        ],
                    )

                    # Execute each tool
                    tool_results = []
                    for tc in tool_calls:
                        self.logger.info(
                            "executing_tool",
                            tool=tc["name"],
                            task_id=context.task_id,
                        )
                        result = await self.execute_tool(
                            tc["name"],
                            tc["arguments"],
                            context,
                        )
                        result.tool_call_id = tc["id"]
                        tool_results.append(result)

                    # Record tool results
                    context.add_message(
                        MessageRole.TOOL,
                        "",
                        tool_results=tool_results,
                    )

                else:
                    # Unexpected stop reason
                    self.logger.warning(
                        "unexpected_stop_reason",
                        reason=response.get("stop_reason"),
                    )
                    break

            # Hit max calls limit
            return self._build_response(
                context=context,
                status=TaskStatus.FAILED,
                error=f"Reached maximum LLM calls ({context.config.max_llm_calls})",
                started_at=started_at,
            )

        except Exception as e:
            self.logger.exception("agent_error", error=str(e))
            return self._build_response(
                context=context,
                status=TaskStatus.FAILED,
                error=str(e),
                started_at=started_at,
            )

    async def _call_llm(self, context: AgentContext) -> dict[str, Any]:
        """Call the LLM with current conversation state."""
        # Convert messages to LLM format
        messages = []
        for msg in context.messages:
            if msg.role == MessageRole.SYSTEM:
                messages.append({"role": "system", "content": msg.content})
            elif msg.role == MessageRole.USER:
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == MessageRole.TOOL and msg.tool_results:
                # Format tool results for the LLM
                for result in msg.tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": str(result.result) if result.success else f"Error: {result.error}",
                    })

        response = await context.llm_client.chat(
            messages=messages,
            tools=self.tools,
            model=context.config.llm_model,
        )

        # Track token usage
        if "usage" in response:
            context.total_tokens += response["usage"].get("total_tokens", 0)

        return response

    def _build_response(
        self,
        context: AgentContext,
        status: TaskStatus,
        result: Any = None,
        error: str | None = None,
        started_at: datetime | None = None,
    ) -> TaskResponse:
        """Build the final task response."""
        return TaskResponse(
            task_id=context.task_id,
            agent_type=self.agent_type,
            status=status,
            instructions=context.instructions,
            result=result,
            error=error,
            messages=context.messages,
            artifacts=context.artifacts,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            llm_calls=context.llm_calls,
            total_tokens=context.total_tokens,
        )

    async def setup(self) -> None:
        """
        Called once when the agent is initialized.
        
        Override to set up resources (browser, connections, etc.)
        """
        pass

    async def teardown(self) -> None:
        """
        Called when the agent is being shut down.
        
        Override to clean up resources.
        """
        pass
