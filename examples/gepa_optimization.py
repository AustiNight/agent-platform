#!/usr/bin/env python3
"""
Example: GEPA Optimization for Browser Agent.

This script demonstrates how to use GEPA to optimize an agent's prompts
based on execution feedback.

GEPA (Genetic-Pareto) optimization:
1. Runs the agent on training tasks
2. Collects execution traces and feedback
3. Uses LLM reflection to propose improved prompts
4. Evaluates candidates on a validation set
5. Maintains a Pareto frontier of diverse strategies
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.browser.agent import BrowserAgent
from src.core.gepa import (
    GEPAConfig,
    ScoreWithFeedback,
    browser_task_metric,
    OptimizationStore,
)
from src.core.gepa.optimizer import AgentOptimizer, InferenceTimeSearch


# =============================================================================
# Training Data
# =============================================================================

# These are example tasks the agent should learn from
TRAINING_TASKS = [
    {
        "instructions": "Go to https://example.com and extract the main heading",
        "expected": {
            "success": True,
            "extracted_data": {"heading": "Example Domain"},
        },
    },
    {
        "instructions": "Navigate to https://httpbin.org/forms/post and fill the form with name='Test User' and email='test@example.com', then submit",
        "expected": {
            "success": True,
        },
    },
    {
        "instructions": "Go to https://news.ycombinator.com and extract the titles of the top 3 stories",
        "expected": {
            "success": True,
            "extracted_data": {"titles": ["...", "...", "..."]},  # Actual titles vary
        },
    },
]

# Validation tasks to measure optimization quality
VALIDATION_TASKS = [
    {
        "instructions": "Navigate to https://quotes.toscrape.com and extract the first 3 quotes",
        "expected": {
            "success": True,
            "extracted_data": {"quotes": ["...", "...", "..."]},
        },
    },
]


# =============================================================================
# Custom Metric (Optional)
# =============================================================================

def custom_browser_metric(
    gold: dict,
    pred: dict,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> ScoreWithFeedback:
    """
    Custom metric with domain-specific feedback.
    
    This is where you inject knowledge about what makes a good execution.
    """
    feedback_parts = []
    subscores = {}
    
    # 1. Did it succeed?
    if pred.get("success"):
        subscores["completion"] = 1.0
        feedback_parts.append("Task completed successfully.")
    else:
        subscores["completion"] = 0.0
        error = pred.get("error", "Unknown error")
        feedback_parts.append(f"Task failed: {error}")
        
        # Provide specific advice based on error type
        if "timeout" in error.lower():
            feedback_parts.append(
                "SUGGESTION: Try adding explicit wait_for calls before interacting with elements."
            )
        elif "not found" in error.lower():
            feedback_parts.append(
                "SUGGESTION: Use get_page_state to see available elements before clicking/typing."
            )
    
    # 2. Analyze the trace for patterns
    if trace:
        step_types = [s["type"] for s in trace.steps]
        
        # Good pattern: start with navigation, then page state, then actions
        if step_types and step_types[0] == "navigate":
            subscores["strategy"] = 0.3
        else:
            subscores["strategy"] = 0.0
            feedback_parts.append(
                "SUGGESTION: Always start by navigating to the target URL."
            )
        
        # Check if agent explored the page before acting
        if "get_page_state" in step_types[:3]:
            subscores["strategy"] += 0.4
        else:
            feedback_parts.append(
                "SUGGESTION: Use get_page_state early to understand the page structure."
            )
        
        # Check for excessive retries
        failed_steps = [s for s in trace.steps if not s.get("success")]
        retry_ratio = len(failed_steps) / max(1, len(trace.steps))
        
        if retry_ratio > 0.5:
            subscores["strategy"] += 0.0
            feedback_parts.append(
                f"ISSUE: {len(failed_steps)}/{len(trace.steps)} steps failed. "
                "Consider more careful selector choices."
            )
        else:
            subscores["strategy"] += 0.3
    else:
        subscores["strategy"] = 0.5
    
    # 3. Data extraction quality
    expected_data = gold.get("extracted_data")
    actual_data = pred.get("extracted_data")
    
    if expected_data is not None:
        if actual_data:
            # Check if key fields are present
            if isinstance(expected_data, dict) and isinstance(actual_data, dict):
                expected_keys = set(expected_data.keys())
                actual_keys = set(actual_data.keys())
                key_overlap = len(expected_keys & actual_keys) / max(1, len(expected_keys))
                subscores["extraction"] = key_overlap
                
                if key_overlap < 1.0:
                    missing = expected_keys - actual_keys
                    feedback_parts.append(
                        f"MISSING DATA: Expected keys {missing} not found in extraction."
                    )
            else:
                subscores["extraction"] = 0.5
        else:
            subscores["extraction"] = 0.0
            feedback_parts.append(
                "ISSUE: No data extracted. Use extract_text or complete_task with extracted_data."
            )
    else:
        subscores["extraction"] = 1.0  # Not applicable
    
    # Calculate weighted score
    weights = {"completion": 0.5, "strategy": 0.3, "extraction": 0.2}
    total_score = sum(subscores.get(k, 0) * v for k, v in weights.items())
    
    return ScoreWithFeedback(
        score=total_score,
        feedback=" | ".join(feedback_parts),
        subscores=subscores,
    )


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run GEPA optimization on the browser agent."""
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)
    
    print("=" * 60)
    print("GEPA Optimization for Browser Agent")
    print("=" * 60)
    
    # Create agent
    agent = BrowserAgent()
    
    print(f"\nInitial system prompt:\n{'-' * 40}")
    print(agent.system_prompt[:500] + "...")
    
    # Create optimizer
    config = GEPAConfig(
        auto="light",  # Use "medium" or "heavy" for more thorough optimization
        reflection_model="claude-sonnet-4-20250514",
        track_stats=True,
    )
    
    store = OptimizationStore("./data/gepa")
    
    optimizer = AgentOptimizer(
        agent=agent,
        metric=custom_browser_metric,  # Use our custom metric
        config=config,
        store=store,
    )
    
    print(f"\nStarting optimization with {len(TRAINING_TASKS)} training tasks...")
    print(f"Configuration: {config.auto} budget")
    
    # Run optimization
    result = await optimizer.optimize(
        trainset=TRAINING_TASKS,
        valset=VALIDATION_TASKS,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(result.summary())
    
    if result.improvement > 0:
        print(f"\nOptimized system prompt:\n{'-' * 40}")
        print(result.optimized_texts.get("system_prompt", "N/A")[:500] + "...")
    
    # Demonstrate inference-time search
    print("\n" + "=" * 60)
    print("INFERENCE-TIME SEARCH DEMO")
    print("=" * 60)
    
    searcher = InferenceTimeSearch(agent, metric=custom_browser_metric)
    
    test_task = {
        "instructions": "Go to https://example.com and get the page title",
        "expected": {"success": True},
    }
    
    print(f"\nSearching for best execution of: {test_task['instructions']}")
    
    best_result, best_score = await searcher.search(
        task=test_task,
        num_candidates=3,
    )
    
    print(f"\nBest result (score: {best_score:.3f}):")
    print(f"  Success: {best_result.get('success')}")
    if best_result.get("error"):
        print(f"  Error: {best_result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())
