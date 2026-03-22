"""
GEPA Integration - Reflective Prompt Optimization.

This module integrates GEPA (Genetic-Pareto) optimization as an intrinsic
capability for all agents. GEPA evolves prompts based on execution feedback,
enabling agents to improve over time.

Key concepts:
- Every agent execution produces a trace with feedback
- GEPA uses these traces to evolve system prompts and instructions
- Pareto frontier maintains diverse strategies that excel on different tasks
- Rich textual feedback (not just scores) guides evolution

Usage:
    # Optimize an agent's prompts
    optimizer = AgentOptimizer(agent_type="browser")
    optimized_agent = await optimizer.optimize(
        trainset=historical_tasks,
        valset=validation_tasks,
    )
    
    # Inference-time search for complex tasks
    searcher = InferenceTimeSearch(agent)
    best_result = await searcher.search(task, num_candidates=10)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Protocol, Union
from pathlib import Path
import json

import structlog

logger = structlog.get_logger()


# =============================================================================
# Core Types
# =============================================================================


@dataclass
class ExecutionTrace:
    """
    Captures the full execution trace of an agent task.
    
    This is the raw material GEPA uses for reflection.
    """
    task_id: str
    agent_type: str
    instructions: str
    
    # The prompt/instruction that was used
    system_prompt: str
    
    # Step-by-step execution
    steps: list[dict[str, Any]] = field(default_factory=list)
    
    # Final outcome
    success: bool = False
    result: Any = None
    error: str | None = None
    
    # Metrics
    llm_calls: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    
    # Timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    
    def add_step(
        self,
        step_type: str,
        input_data: Any,
        output_data: Any,
        success: bool = True,
        error: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Record an execution step."""
        self.steps.append({
            "step_num": len(self.steps) + 1,
            "type": step_type,
            "input": input_data,
            "output": output_data,
            "success": success,
            "error": error,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def to_text(self) -> str:
        """Convert trace to human-readable text for GEPA reflection."""
        parts = [
            f"Task: {self.instructions}",
            f"Agent: {self.agent_type}",
            f"System Prompt: {self.system_prompt[:500]}...",
            "",
            "Execution Steps:",
        ]
        
        for step in self.steps:
            status = "✓" if step["success"] else "✗"
            parts.append(f"  {step['step_num']}. [{status}] {step['type']}")
            if step.get("error"):
                parts.append(f"      Error: {step['error']}")
        
        parts.extend([
            "",
            f"Outcome: {'SUCCESS' if self.success else 'FAILED'}",
            f"LLM Calls: {self.llm_calls}",
            f"Duration: {self.duration_seconds:.1f}s",
        ])
        
        if self.error:
            parts.append(f"Error: {self.error}")
        
        return "\n".join(parts)


@dataclass
class ScoreWithFeedback:
    """
    GEPA-compatible score with textual feedback.
    
    The feedback is what makes GEPA powerful - it provides
    actionable information for prompt evolution.
    """
    score: float
    feedback: str
    
    # Optional: per-objective breakdown
    subscores: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "feedback": self.feedback,
            "subscores": self.subscores,
        }


class FeedbackMetric(Protocol):
    """
    Protocol for GEPA feedback metrics.
    
    Implement this to define how agent executions are scored
    and what feedback is generated.
    """
    
    def __call__(
        self,
        gold: dict[str, Any],  # Expected outcome
        pred: dict[str, Any],  # Actual outcome
        trace: ExecutionTrace | None = None,
        pred_name: str | None = None,  # Which component is being optimized
        pred_trace: ExecutionTrace | None = None,  # Sub-trace for that component
    ) -> Union[float, ScoreWithFeedback]:
        """
        Score an execution and provide feedback.
        
        Args:
            gold: The expected/ideal outcome
            pred: The actual outcome from the agent
            trace: Full execution trace
            pred_name: Name of predictor being optimized (for GEPA)
            pred_trace: Sub-trace for that predictor
            
        Returns:
            Either a float score or ScoreWithFeedback with rich feedback
        """
        ...


# =============================================================================
# Built-in Feedback Metrics
# =============================================================================


def browser_task_metric(
    gold: dict[str, Any],
    pred: dict[str, Any],
    trace: ExecutionTrace | None = None,
    pred_name: str | None = None,
    pred_trace: ExecutionTrace | None = None,
) -> ScoreWithFeedback:
    """
    Default metric for browser automation tasks.
    
    Scores based on:
    - Task completion (50%)
    - Efficiency (LLM calls, duration) (25%)
    - Data extraction accuracy (25%)
    """
    feedback_parts = []
    subscores = {}
    
    # 1. Task completion
    if pred.get("success"):
        subscores["completion"] = 1.0
        feedback_parts.append("Task completed successfully.")
    else:
        subscores["completion"] = 0.0
        error = pred.get("error", "Unknown error")
        feedback_parts.append(f"Task failed: {error}")
        
        # Analyze failure from trace
        if trace:
            failed_steps = [s for s in trace.steps if not s.get("success")]
            if failed_steps:
                last_fail = failed_steps[-1]
                feedback_parts.append(
                    f"First failure at step {last_fail['step_num']}: "
                    f"{last_fail['type']} - {last_fail.get('error', 'no error message')}"
                )
    
    # 2. Efficiency
    if trace:
        # Penalize excessive LLM calls
        if trace.llm_calls <= 5:
            subscores["efficiency"] = 1.0
        elif trace.llm_calls <= 10:
            subscores["efficiency"] = 0.7
        elif trace.llm_calls <= 20:
            subscores["efficiency"] = 0.4
        else:
            subscores["efficiency"] = 0.2
            feedback_parts.append(
                f"Used {trace.llm_calls} LLM calls - consider more direct approaches."
            )
        
        # Analyze step patterns
        step_types = [s["type"] for s in trace.steps]
        if step_types.count("get_page_state") > 5:
            feedback_parts.append(
                "Excessive page state checks - try to plan actions more confidently."
            )
    else:
        subscores["efficiency"] = 0.5
    
    # 3. Data extraction (if applicable)
    expected_data = gold.get("expected_data")
    extracted_data = pred.get("extracted_data")
    
    if expected_data is not None:
        if extracted_data == expected_data:
            subscores["extraction"] = 1.0
            feedback_parts.append("Data extraction matched expected output.")
        elif extracted_data:
            # Partial match
            if isinstance(expected_data, dict) and isinstance(extracted_data, dict):
                matching = sum(1 for k, v in expected_data.items() 
                             if extracted_data.get(k) == v)
                subscores["extraction"] = matching / len(expected_data)
                feedback_parts.append(
                    f"Partial data match: {matching}/{len(expected_data)} fields correct."
                )
            else:
                subscores["extraction"] = 0.3
                feedback_parts.append("Data extracted but doesn't match expected format.")
        else:
            subscores["extraction"] = 0.0
            feedback_parts.append("No data extracted when extraction was expected.")
    else:
        subscores["extraction"] = 1.0  # No extraction required
    
    # Calculate weighted score
    weights = {"completion": 0.5, "efficiency": 0.25, "extraction": 0.25}
    total_score = sum(subscores[k] * weights[k] for k in weights)
    
    return ScoreWithFeedback(
        score=total_score,
        feedback=" ".join(feedback_parts),
        subscores=subscores,
    )


def general_task_metric(
    gold: dict[str, Any],
    pred: dict[str, Any],
    trace: ExecutionTrace | None = None,
    pred_name: str | None = None,
    pred_trace: ExecutionTrace | None = None,
) -> ScoreWithFeedback:
    """
    Generic metric for any agent task.
    
    Override with domain-specific metrics for better results.
    """
    feedback_parts = []
    
    if pred.get("success"):
        score = 0.8  # Base score for success
        feedback_parts.append("Task completed.")
        
        # Bonus for efficiency
        if trace and trace.llm_calls < 10:
            score += 0.1
            feedback_parts.append("Efficient execution.")
        
        # Bonus for fast completion
        if trace and trace.duration_seconds < 30:
            score += 0.1
            feedback_parts.append("Fast completion.")
    else:
        score = 0.0
        feedback_parts.append(f"Task failed: {pred.get('error', 'Unknown')}")
        
        if trace:
            # Provide actionable feedback
            failed_steps = [s for s in trace.steps if not s.get("success")]
            if failed_steps:
                feedback_parts.append(
                    f"Failed at step {failed_steps[0]['step_num']}: {failed_steps[0]['type']}"
                )
    
    return ScoreWithFeedback(
        score=min(1.0, max(0.0, score)),
        feedback=" ".join(feedback_parts),
    )


# =============================================================================
# GEPA Configuration
# =============================================================================


@dataclass
class GEPAConfig:
    """Configuration for GEPA optimization."""
    
    # Budget
    auto: str | None = "medium"  # "light", "medium", "heavy"
    max_full_evals: int | None = None
    max_metric_calls: int | None = None
    
    # Reflection settings
    reflection_minibatch_size: int = 3
    reflection_model: str = "claude-sonnet-4-20250514"  # Strong model for reflection
    reflection_temperature: float = 1.0
    reflection_max_tokens: int = 16000
    
    # Evolution settings
    candidate_selection_strategy: str = "pareto"  # or "current_best"
    use_merge: bool = True
    max_merge_invocations: int = 5
    
    # Performance
    num_threads: int = 4
    skip_perfect_score: bool = True
    
    # Persistence
    log_dir: str | None = None
    track_stats: bool = True
    track_best_outputs: bool = True
    
    # Scoring
    failure_score: float = 0.0
    perfect_score: float = 1.0
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> dict:
        return {
            "auto": self.auto,
            "max_full_evals": self.max_full_evals,
            "max_metric_calls": self.max_metric_calls,
            "reflection_minibatch_size": self.reflection_minibatch_size,
            "candidate_selection_strategy": self.candidate_selection_strategy,
            "use_merge": self.use_merge,
            "max_merge_invocations": self.max_merge_invocations,
            "num_threads": self.num_threads,
            "skip_perfect_score": self.skip_perfect_score,
            "failure_score": self.failure_score,
            "perfect_score": self.perfect_score,
            "seed": self.seed,
        }


# =============================================================================
# Trace Collection Mixin
# =============================================================================


class TraceCollectorMixin:
    """
    Mixin that adds GEPA trace collection to any agent.
    
    Agents that inherit from this will automatically capture
    execution traces suitable for GEPA optimization.
    """
    
    _current_trace: ExecutionTrace | None = None
    _traces: list[ExecutionTrace] = []
    
    def start_trace(
        self,
        task_id: str,
        agent_type: str,
        instructions: str,
        system_prompt: str,
    ) -> ExecutionTrace:
        """Start capturing a new execution trace."""
        self._current_trace = ExecutionTrace(
            task_id=task_id,
            agent_type=agent_type,
            instructions=instructions,
            system_prompt=system_prompt,
        )
        return self._current_trace
    
    def record_step(
        self,
        step_type: str,
        input_data: Any,
        output_data: Any,
        success: bool = True,
        error: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Record a step in the current trace."""
        if self._current_trace:
            self._current_trace.add_step(
                step_type=step_type,
                input_data=input_data,
                output_data=output_data,
                success=success,
                error=error,
                metadata=metadata,
            )
    
    def complete_trace(
        self,
        success: bool,
        result: Any = None,
        error: str | None = None,
        llm_calls: int = 0,
        total_tokens: int = 0,
    ) -> ExecutionTrace:
        """Complete and store the current trace."""
        if self._current_trace:
            self._current_trace.success = success
            self._current_trace.result = result
            self._current_trace.error = error
            self._current_trace.llm_calls = llm_calls
            self._current_trace.total_tokens = total_tokens
            self._current_trace.completed_at = datetime.utcnow()
            self._current_trace.duration_seconds = (
                self._current_trace.completed_at - self._current_trace.started_at
            ).total_seconds()
            
            self._traces.append(self._current_trace)
            trace = self._current_trace
            self._current_trace = None
            return trace
        
        raise ValueError("No active trace to complete")
    
    def get_traces(self) -> list[ExecutionTrace]:
        """Get all collected traces."""
        return self._traces.copy()
    
    def clear_traces(self) -> None:
        """Clear collected traces."""
        self._traces = []


# =============================================================================
# Optimizable Component Interface
# =============================================================================


class OptimizableComponent(ABC):
    """
    Interface for components that can be optimized by GEPA.
    
    Any agent or module that wants GEPA optimization should
    implement this interface.
    """
    
    @property
    @abstractmethod
    def optimizable_text(self) -> dict[str, str]:
        """
        Return the text components that can be optimized.
        
        Returns:
            Dict mapping component names to their current text.
            Example: {"system_prompt": "You are a...", "tool_selection_prompt": "Choose..."}
        """
        ...
    
    @abstractmethod
    def set_optimized_text(self, component_name: str, text: str) -> None:
        """
        Update an optimizable component with new text.
        
        Args:
            component_name: Name of component (from optimizable_text keys)
            text: New optimized text
        """
        ...
    
    @abstractmethod
    async def execute_for_optimization(
        self,
        task: dict[str, Any],
    ) -> tuple[dict[str, Any], ExecutionTrace]:
        """
        Execute a task and return result with trace.
        
        This method is called during GEPA optimization to
        evaluate candidate prompts.
        
        Args:
            task: Task definition with instructions and expected outcome
            
        Returns:
            Tuple of (result dict, execution trace)
        """
        ...


# =============================================================================
# Persistence
# =============================================================================


class OptimizationStore:
    """
    Persists optimization results and traces.
    
    Stores:
    - Execution traces for training
    - Optimized prompts (Pareto frontier)
    - Optimization run metadata
    """
    
    def __init__(self, storage_dir: Path | str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.traces_dir = self.storage_dir / "traces"
        self.traces_dir.mkdir(exist_ok=True)
        
        self.prompts_dir = self.storage_dir / "prompts"
        self.prompts_dir.mkdir(exist_ok=True)
        
        self.runs_dir = self.storage_dir / "runs"
        self.runs_dir.mkdir(exist_ok=True)
    
    def save_trace(self, trace: ExecutionTrace) -> Path:
        """Save an execution trace."""
        filepath = self.traces_dir / f"{trace.task_id}.json"
        
        with open(filepath, "w") as f:
            json.dump({
                "task_id": trace.task_id,
                "agent_type": trace.agent_type,
                "instructions": trace.instructions,
                "system_prompt": trace.system_prompt,
                "steps": trace.steps,
                "success": trace.success,
                "result": trace.result,
                "error": trace.error,
                "llm_calls": trace.llm_calls,
                "total_tokens": trace.total_tokens,
                "duration_seconds": trace.duration_seconds,
                "started_at": trace.started_at.isoformat(),
                "completed_at": trace.completed_at.isoformat() if trace.completed_at else None,
            }, f, indent=2, default=str)
        
        return filepath
    
    def load_traces(self, agent_type: str | None = None) -> list[ExecutionTrace]:
        """Load traces, optionally filtered by agent type."""
        traces = []
        
        for filepath in self.traces_dir.glob("*.json"):
            with open(filepath) as f:
                data = json.load(f)
            
            if agent_type and data.get("agent_type") != agent_type:
                continue
            
            trace = ExecutionTrace(
                task_id=data["task_id"],
                agent_type=data["agent_type"],
                instructions=data["instructions"],
                system_prompt=data["system_prompt"],
                steps=data["steps"],
                success=data["success"],
                result=data["result"],
                error=data["error"],
                llm_calls=data["llm_calls"],
                total_tokens=data["total_tokens"],
                duration_seconds=data["duration_seconds"],
                started_at=datetime.fromisoformat(data["started_at"]),
                completed_at=(
                    datetime.fromisoformat(data["completed_at"])
                    if data.get("completed_at") else None
                ),
            )
            traces.append(trace)
        
        return traces
    
    def save_optimized_prompt(
        self,
        agent_type: str,
        component_name: str,
        prompt_text: str,
        score: float,
        metadata: dict | None = None,
    ) -> Path:
        """Save an optimized prompt."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{agent_type}_{component_name}_{timestamp}.json"
        filepath = self.prompts_dir / filename
        
        with open(filepath, "w") as f:
            json.dump({
                "agent_type": agent_type,
                "component_name": component_name,
                "prompt_text": prompt_text,
                "score": score,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
            }, f, indent=2)
        
        return filepath
    
    def get_best_prompt(
        self,
        agent_type: str,
        component_name: str,
    ) -> tuple[str, float] | None:
        """Get the best optimized prompt for a component."""
        best_prompt = None
        best_score = -1.0
        
        pattern = f"{agent_type}_{component_name}_*.json"
        for filepath in self.prompts_dir.glob(pattern):
            with open(filepath) as f:
                data = json.load(f)
            
            if data["score"] > best_score:
                best_score = data["score"]
                best_prompt = data["prompt_text"]
        
        if best_prompt:
            return best_prompt, best_score
        return None


# =============================================================================
# Metric Registry
# =============================================================================


_METRICS: dict[str, FeedbackMetric] = {
    "browser": browser_task_metric,
    "default": general_task_metric,
}


def register_metric(agent_type: str, metric: FeedbackMetric) -> None:
    """Register a feedback metric for an agent type."""
    _METRICS[agent_type] = metric


def get_metric(agent_type: str) -> FeedbackMetric:
    """Get the feedback metric for an agent type."""
    return _METRICS.get(agent_type, _METRICS["default"])
