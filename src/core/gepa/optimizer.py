"""
GEPA Optimizer - Adapts DSPy's GEPA for agent optimization.

This module provides the main optimization loop that:
1. Collects execution traces from agent runs
2. Converts them to GEPA-compatible format
3. Runs GEPA optimization
4. Applies optimized prompts back to agents
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

import structlog

from src.core.gepa import (
    ExecutionTrace,
    ScoreWithFeedback,
    FeedbackMetric,
    GEPAConfig,
    OptimizableComponent,
    OptimizationStore,
    get_metric,
)

logger = structlog.get_logger()

T = TypeVar("T", bound=OptimizableComponent)


@dataclass
class OptimizationResult:
    """Result of a GEPA optimization run."""
    
    # The optimized component
    optimized_texts: dict[str, str]
    
    # Scores
    best_score: float
    initial_score: float
    improvement: float
    
    # Pareto frontier
    pareto_candidates: list[dict[str, str]]
    pareto_scores: list[float]
    
    # Metadata
    total_evaluations: int
    duration_seconds: float
    config: GEPAConfig
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Optimization complete:\n"
            f"  Initial score: {self.initial_score:.3f}\n"
            f"  Best score: {self.best_score:.3f}\n"
            f"  Improvement: {self.improvement:+.3f} ({self.improvement/max(0.001, self.initial_score)*100:+.1f}%)\n"
            f"  Evaluations: {self.total_evaluations}\n"
            f"  Pareto candidates: {len(self.pareto_candidates)}\n"
            f"  Duration: {self.duration_seconds:.1f}s"
        )


class AgentOptimizer:
    """
    GEPA-based optimizer for agents.
    
    This optimizer:
    1. Takes an agent that implements OptimizableComponent
    2. Runs it on a training set, collecting traces
    3. Uses GEPA to evolve better prompts
    4. Returns the optimized agent
    
    Example:
        optimizer = AgentOptimizer(
            agent=browser_agent,
            metric=browser_task_metric,
            config=GEPAConfig(auto="medium"),
        )
        
        result = await optimizer.optimize(
            trainset=[{"instructions": "...", "expected": {...}}, ...],
            valset=[...],  # Optional validation set
        )
        
        print(result.summary())
        # Agent is now using optimized prompts
    """
    
    def __init__(
        self,
        agent: OptimizableComponent,
        metric: FeedbackMetric | None = None,
        config: GEPAConfig | None = None,
        store: OptimizationStore | None = None,
    ):
        self.agent = agent
        self.metric = metric or get_metric(getattr(agent, "agent_type", "default"))
        self.config = config or GEPAConfig()
        self.store = store
        
        self.logger = logger.bind(
            optimizer="gepa",
            agent_type=getattr(agent, "agent_type", "unknown"),
        )
    
    async def optimize(
        self,
        trainset: list[dict[str, Any]],
        valset: list[dict[str, Any]] | None = None,
    ) -> OptimizationResult:
        """
        Run GEPA optimization on the agent.
        
        Args:
            trainset: Training tasks with expected outcomes
            valset: Optional validation tasks (uses trainset if not provided)
            
        Returns:
            OptimizationResult with optimized prompts and scores
        """
        start_time = datetime.utcnow()
        valset = valset or trainset
        
        self.logger.info(
            "starting_optimization",
            trainset_size=len(trainset),
            valset_size=len(valset),
            config=self.config.to_dict(),
        )
        
        # Get initial components
        initial_texts = self.agent.optimizable_text.copy()
        
        # Evaluate initial performance
        initial_score = await self._evaluate_batch(valset)
        self.logger.info("initial_evaluation", score=initial_score)
        
        # Run GEPA optimization
        try:
            result = await self._run_gepa(trainset, valset, initial_texts)
        except ImportError:
            # DSPy/GEPA not installed - use fallback
            self.logger.warning("gepa_not_installed", message="Using fallback optimization")
            result = await self._fallback_optimize(trainset, valset, initial_texts)
        
        # Apply best prompts
        for component_name, text in result.optimized_texts.items():
            self.agent.set_optimized_text(component_name, text)
            
            # Persist if store is configured
            if self.store:
                self.store.save_optimized_prompt(
                    agent_type=getattr(self.agent, "agent_type", "unknown"),
                    component_name=component_name,
                    prompt_text=text,
                    score=result.best_score,
                    metadata={"config": self.config.to_dict()},
                )
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        result.duration_seconds = duration
        result.initial_score = initial_score
        result.improvement = result.best_score - initial_score
        
        self.logger.info(
            "optimization_complete",
            initial_score=initial_score,
            best_score=result.best_score,
            improvement=result.improvement,
            duration_seconds=duration,
        )
        
        return result
    
    async def _evaluate_batch(
        self,
        tasks: list[dict[str, Any]],
    ) -> float:
        """Evaluate agent on a batch of tasks."""
        scores = []
        
        for task in tasks:
            try:
                result, trace = await self.agent.execute_for_optimization(task)
                
                score_result = self.metric(
                    gold=task.get("expected", {}),
                    pred=result,
                    trace=trace,
                )
                
                if isinstance(score_result, ScoreWithFeedback):
                    scores.append(score_result.score)
                else:
                    scores.append(float(score_result))
                    
            except Exception as e:
                self.logger.warning("evaluation_error", task=task, error=str(e))
                scores.append(self.config.failure_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    async def _run_gepa(
        self,
        trainset: list[dict[str, Any]],
        valset: list[dict[str, Any]],
        initial_texts: dict[str, str],
    ) -> OptimizationResult:
        """
        Run actual GEPA optimization using DSPy.
        
        This requires dspy and gepa packages to be installed.
        """
        try:
            import dspy
            from dspy.teleprompt import GEPA
        except ImportError:
            raise ImportError(
                "GEPA optimization requires dspy and gepa packages. "
                "Install with: pip install dspy-ai gepa"
            )
        
        # Create DSPy-compatible module wrapper
        # This is a simplified adapter - real implementation would be more sophisticated
        
        class AgentModule(dspy.Module):
            def __init__(self, agent: OptimizableComponent, component_name: str):
                super().__init__()
                self.agent = agent
                self.component_name = component_name
                
                # Create a predictor for the component
                self.predictor = dspy.Predict(
                    dspy.Signature(
                        "instructions -> action_plan",
                        instructions=dspy.InputField(),
                        action_plan=dspy.OutputField(),
                    )
                )
                self.predictor.signature.instructions = initial_texts.get(component_name, "")
            
            def forward(self, instructions: str):
                return self.predictor(instructions=instructions)
        
        # Wrap metric for DSPy
        def dspy_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            # Convert DSPy types to our types
            score_result = self.metric(
                gold=gold.toDict() if hasattr(gold, "toDict") else dict(gold),
                pred=pred.toDict() if hasattr(pred, "toDict") else {"output": pred},
                trace=None,  # DSPy trace is different format
                pred_name=pred_name,
                pred_trace=None,
            )
            
            if isinstance(score_result, ScoreWithFeedback):
                return {"score": score_result.score, "feedback": score_result.feedback}
            return float(score_result)
        
        # Create GEPA optimizer
        gepa = GEPA(
            metric=dspy_metric,
            auto=self.config.auto,
            max_full_evals=self.config.max_full_evals,
            max_metric_calls=self.config.max_metric_calls,
            reflection_minibatch_size=self.config.reflection_minibatch_size,
            candidate_selection_strategy=self.config.candidate_selection_strategy,
            use_merge=self.config.use_merge,
            max_merge_invocations=self.config.max_merge_invocations,
            num_threads=self.config.num_threads,
            skip_perfect_score=self.config.skip_perfect_score,
            failure_score=self.config.failure_score,
            perfect_score=self.config.perfect_score,
            track_stats=self.config.track_stats,
            track_best_outputs=self.config.track_best_outputs,
            log_dir=self.config.log_dir,
            seed=self.config.seed,
        )
        
        # Run optimization for each component
        optimized_texts = {}
        best_score = 0.0
        pareto_candidates = []
        pareto_scores = []
        total_evals = 0
        
        for component_name in initial_texts.keys():
            module = AgentModule(self.agent, component_name)
            
            # Convert tasks to DSPy Examples
            dspy_trainset = [
                dspy.Example(
                    instructions=t["instructions"],
                    expected=t.get("expected", {}),
                ).with_inputs("instructions")
                for t in trainset
            ]
            
            dspy_valset = [
                dspy.Example(
                    instructions=t["instructions"],
                    expected=t.get("expected", {}),
                ).with_inputs("instructions")
                for t in valset
            ]
            
            # Run GEPA
            optimized = gepa.compile(
                student=module,
                trainset=dspy_trainset,
                valset=dspy_valset,
            )
            
            # Extract results
            if hasattr(optimized, "detailed_results"):
                results = optimized.detailed_results
                if results.val_aggregate_scores:
                    component_best_score = max(results.val_aggregate_scores)
                    if component_best_score > best_score:
                        best_score = component_best_score
                    pareto_scores.extend(results.val_aggregate_scores)
                    total_evals += results.total_metric_calls or 0
            
            # Get optimized instruction
            if hasattr(optimized.predictor, "signature"):
                optimized_texts[component_name] = optimized.predictor.signature.instructions
            else:
                optimized_texts[component_name] = initial_texts[component_name]
        
        return OptimizationResult(
            optimized_texts=optimized_texts,
            best_score=best_score,
            initial_score=0.0,  # Set by caller
            improvement=0.0,  # Set by caller
            pareto_candidates=pareto_candidates,
            pareto_scores=pareto_scores,
            total_evaluations=total_evals,
            duration_seconds=0.0,  # Set by caller
            config=self.config,
        )
    
    async def _fallback_optimize(
        self,
        trainset: list[dict[str, Any]],
        valset: list[dict[str, Any]],
        initial_texts: dict[str, str],
    ) -> OptimizationResult:
        """
        Fallback optimization when GEPA is not available.
        
        Uses a simple hill-climbing approach with LLM-based reflection.
        """
        from src.core.llm_client import LLMClient
        
        llm = LLMClient(model=self.config.reflection_model)
        
        best_texts = initial_texts.copy()
        best_score = await self._evaluate_batch(valset)
        
        iterations = 5 if self.config.auto == "light" else (
            10 if self.config.auto == "medium" else 20
        )
        
        for i in range(iterations):
            self.logger.debug("fallback_iteration", iteration=i + 1, best_score=best_score)
            
            # Collect feedback from failed examples
            feedback_parts = []
            for task in trainset[:self.config.reflection_minibatch_size]:
                try:
                    result, trace = await self.agent.execute_for_optimization(task)
                    score_result = self.metric(
                        gold=task.get("expected", {}),
                        pred=result,
                        trace=trace,
                    )
                    
                    if isinstance(score_result, ScoreWithFeedback):
                        if score_result.score < self.config.perfect_score:
                            feedback_parts.append(f"Task: {task['instructions'][:100]}...")
                            feedback_parts.append(f"Score: {score_result.score:.2f}")
                            feedback_parts.append(f"Feedback: {score_result.feedback}")
                            feedback_parts.append("")
                except Exception as e:
                    feedback_parts.append(f"Task failed with error: {e}")
            
            if not feedback_parts:
                break  # All tasks succeeded
            
            # Generate improved prompts via reflection
            for component_name, current_text in best_texts.items():
                reflection_prompt = f"""You are improving a system prompt for an AI agent.

Current prompt:
{current_text}

Recent execution feedback:
{chr(10).join(feedback_parts)}

Based on this feedback, propose an improved version of the prompt that addresses the issues.
Return ONLY the improved prompt text, no explanations."""

                response = await llm.chat([
                    {"role": "user", "content": reflection_prompt}
                ])
                
                candidate_text = response.get("content", current_text)
                
                # Evaluate candidate
                self.agent.set_optimized_text(component_name, candidate_text)
                candidate_score = await self._evaluate_batch(valset)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_texts[component_name] = candidate_text
                    self.logger.info(
                        "improvement_found",
                        component=component_name,
                        new_score=candidate_score,
                    )
                else:
                    # Revert
                    self.agent.set_optimized_text(component_name, best_texts[component_name])
        
        return OptimizationResult(
            optimized_texts=best_texts,
            best_score=best_score,
            initial_score=0.0,
            improvement=0.0,
            pareto_candidates=[best_texts],
            pareto_scores=[best_score],
            total_evaluations=iterations * len(trainset),
            duration_seconds=0.0,
            config=self.config,
        )


class InferenceTimeSearch:
    """
    GEPA-based inference-time search.
    
    For complex tasks, runs multiple candidates in parallel
    and returns the best result from the Pareto frontier.
    
    Example:
        searcher = InferenceTimeSearch(agent, metric)
        best_result = await searcher.search(
            task={"instructions": "..."},
            num_candidates=5,
        )
    """
    
    def __init__(
        self,
        agent: OptimizableComponent,
        metric: FeedbackMetric | None = None,
        config: GEPAConfig | None = None,
    ):
        self.agent = agent
        self.metric = metric or get_metric(getattr(agent, "agent_type", "default"))
        self.config = config or GEPAConfig(
            auto="light",
            track_stats=True,
            track_best_outputs=True,
        )
        
        self.logger = logger.bind(optimizer="inference_search")
    
    async def search(
        self,
        task: dict[str, Any],
        num_candidates: int = 5,
        temperature_range: tuple[float, float] = (0.3, 1.0),
    ) -> tuple[dict[str, Any], float]:
        """
        Search for the best execution of a task.
        
        Runs multiple attempts with varying temperatures and
        returns the best result.
        
        Args:
            task: The task to execute
            num_candidates: Number of candidates to try
            temperature_range: Temperature range for variation
            
        Returns:
            Tuple of (best result, score)
        """
        candidates = []
        
        # Generate candidates with varying temperatures
        import random
        temperatures = [
            temperature_range[0] + (temperature_range[1] - temperature_range[0]) * i / (num_candidates - 1)
            for i in range(num_candidates)
        ]
        random.shuffle(temperatures)
        
        for temp in temperatures:
            try:
                # Execute with temperature variation
                # Note: This would need the agent to support temperature parameter
                result, trace = await self.agent.execute_for_optimization(task)
                
                score_result = self.metric(
                    gold=task.get("expected", {}),
                    pred=result,
                    trace=trace,
                )
                
                score = (
                    score_result.score if isinstance(score_result, ScoreWithFeedback)
                    else float(score_result)
                )
                
                candidates.append((result, score, trace))
                
                # Early exit on perfect score
                if score >= self.config.perfect_score:
                    self.logger.info("perfect_score_found", attempt=len(candidates))
                    break
                    
            except Exception as e:
                self.logger.warning("candidate_failed", error=str(e))
                candidates.append(({"error": str(e)}, self.config.failure_score, None))
        
        if not candidates:
            raise RuntimeError("All candidates failed")
        
        # Return best
        best = max(candidates, key=lambda x: x[1])
        
        self.logger.info(
            "search_complete",
            candidates=len(candidates),
            best_score=best[1],
        )
        
        return best[0], best[1]
