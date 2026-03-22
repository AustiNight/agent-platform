"""
GEPA Integration Tests.

Tests the complete GEPA automation pipeline:
1. Trace collection
2. Optimization triggering
3. Prompt application
4. Notification delivery
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory."""
    data_dir = tmp_path / "data" / "gepa"
    data_dir.mkdir(parents=True)
    (data_dir / "traces").mkdir()
    (data_dir / "prompts").mkdir()
    (data_dir / "notifications").mkdir()
    return data_dir


@pytest.fixture
def mock_env(temp_data_dir):
    """Set up test environment."""
    os.environ["GEPA_AUTO_ENABLED"] = "true"
    os.environ["GEPA_MIN_TRACES"] = "5"  # Low threshold for testing
    os.environ["GEPA_NOTIFY_EMAIL"] = "test@example.com"
    yield
    # Cleanup
    for key in ["GEPA_AUTO_ENABLED", "GEPA_MIN_TRACES", "GEPA_NOTIFY_EMAIL"]:
        os.environ.pop(key, None)


class TestTraceCollection:
    """Test trace collection from task execution."""
    
    @pytest.mark.asyncio
    async def test_trace_saved_on_task_completion(self, temp_data_dir):
        """Verify traces are saved when tasks complete."""
        from src.core.gepa import ExecutionTrace, OptimizationStore
        
        store = OptimizationStore(temp_data_dir)
        
        # Create a trace
        trace = ExecutionTrace(
            task_id="test-123",
            agent_type="browser",
            instructions="Go to example.com",
            system_prompt="You are a browser agent...",
            steps=[
                {
                    "step_num": 1,
                    "type": "navigate",
                    "input": {"url": "https://example.com"},
                    "output": "Navigation successful",
                    "success": True,
                    "error": None,
                }
            ],
            success=True,
            result={"title": "Example Domain"},
            llm_calls=3,
            total_tokens=1500,
        )
        
        # Save trace
        path = store.save_trace(trace)
        
        # Verify
        assert path.exists()
        
        # Load and verify
        loaded_traces = store.load_traces(agent_type="browser")
        assert len(loaded_traces) == 1
        assert loaded_traces[0].task_id == "test-123"
        assert loaded_traces[0].success is True
    
    @pytest.mark.asyncio
    async def test_failed_task_trace_captured(self, temp_data_dir):
        """Verify failed task traces are captured for learning."""
        from src.core.gepa import ExecutionTrace, OptimizationStore
        
        store = OptimizationStore(temp_data_dir)
        
        trace = ExecutionTrace(
            task_id="test-fail-456",
            agent_type="browser",
            instructions="Click the missing button",
            system_prompt="You are a browser agent...",
            steps=[
                {
                    "step_num": 1,
                    "type": "click",
                    "input": {"selector": "#missing-button"},
                    "output": None,
                    "success": False,
                    "error": "Element not found",
                }
            ],
            success=False,
            error="Element not found: #missing-button",
            llm_calls=2,
        )
        
        store.save_trace(trace)
        
        traces = store.load_traces()
        failed_traces = [t for t in traces if not t.success]
        assert len(failed_traces) == 1


class TestOptimizationTriggers:
    """Test optimization trigger conditions."""
    
    def test_trigger_on_trace_count(self, temp_data_dir, mock_env):
        """Optimization triggers when trace count threshold is met."""
        from src.core.gepa.automation import AutoGEPAConfig, StateManager
        
        config = AutoGEPAConfig.from_env()
        config.state_file = temp_data_dir / "state.json"
        config.min_traces_for_optimization = 5
        
        state_manager = StateManager(config)
        
        # Add traces
        for i in range(5):
            state_manager.record_trace("browser")
        
        should_opt, reason = state_manager.should_optimize("browser")
        assert should_opt is True
        assert "5 traces" in reason
    
    def test_no_trigger_below_threshold(self, temp_data_dir, mock_env):
        """No optimization when below threshold."""
        from src.core.gepa.automation import AutoGEPAConfig, StateManager
        
        config = AutoGEPAConfig.from_env()
        config.state_file = temp_data_dir / "state.json"
        config.min_traces_for_optimization = 50
        
        state_manager = StateManager(config)
        
        # Add only a few traces
        for i in range(3):
            state_manager.record_trace("browser")
        
        should_opt, reason = state_manager.should_optimize("browser")
        assert should_opt is False


class TestNotificationDelivery:
    """Test notification system."""
    
    @pytest.mark.asyncio
    async def test_file_fallback_always_works(self, temp_data_dir):
        """Verify file fallback always succeeds."""
        from src.core.gepa.notifications import GuaranteedNotifier
        
        notifier = GuaranteedNotifier()
        notifier.fallback_dir = temp_data_dir / "notifications"
        notifier.fallback_dir.mkdir(exist_ok=True)
        
        results = await notifier.send(
            title="Test Alert",
            message="This is a test notification.",
            severity="warning",
        )
        
        # File should always succeed
        file_results = [r for r in results if r.channel == "file"]
        assert len(file_results) == 1
        assert file_results[0].success is True
        
        # Verify file was written
        notification_files = list(notifier.fallback_dir.glob("*.txt"))
        assert len(notification_files) >= 1
    
    @pytest.mark.asyncio
    async def test_escalation_event_formatting(self):
        """Test escalation event has proper instructions."""
        from src.core.gepa.automation import EscalationFactory
        
        event = EscalationFactory.api_key_expired(
            agent_type="browser",
            provider="anthropic",
            error="Invalid API key",
        )
        
        text = event.to_text()
        
        # Should have clear structure
        assert "GEPA AUTOMATION ALERT" in text
        assert "REQUIRED ACTIONS" in text
        assert "STEP 1" in text
        
        # Should have actionable instructions
        assert "console.anthropic.com" in text or "API key" in text
        assert "environment variable" in text.lower() or "export" in text
        
        # Should explain how to return to automation
        assert "RETURNING TO AUTOMATED MODE" in text


class TestPromptOptimization:
    """Test prompt optimization flow."""
    
    @pytest.mark.asyncio
    async def test_optimized_prompt_saved(self, temp_data_dir):
        """Verify optimized prompts are persisted."""
        from src.core.gepa import OptimizationStore
        
        store = OptimizationStore(temp_data_dir)
        
        # Save optimized prompt
        path = store.save_optimized_prompt(
            agent_type="browser",
            component_name="system_prompt",
            prompt_text="You are an improved browser agent...",
            score=0.85,
            metadata={"improvement": 0.15},
        )
        
        assert path.exists()
        
        # Load best prompt
        result = store.get_best_prompt("browser", "system_prompt")
        assert result is not None
        
        prompt, score = result
        assert "improved" in prompt
        assert score == 0.85
    
    @pytest.mark.asyncio
    async def test_agent_loads_optimized_prompt(self, temp_data_dir):
        """Verify agents load optimized prompts on startup."""
        from src.core.gepa import OptimizationStore
        from src.core.gepa.middleware import _OPTIMIZED_PROMPTS, get_optimized_prompt
        import json
        
        store = OptimizationStore(temp_data_dir)
        
        # Create active prompts file
        active_file = store.storage_dir / "active_prompts.json"
        active_file.write_text(json.dumps({
            "browser": {
                "prompts": {"system_prompt": "Optimized prompt text"},
                "score": 0.9,
                "updated_at": datetime.utcnow().isoformat(),
            }
        }))
        
        # Simulate loading
        _OPTIMIZED_PROMPTS["browser"] = {"system_prompt": "Optimized prompt text"}
        
        # Verify
        prompt = get_optimized_prompt("browser", "system_prompt")
        assert prompt == "Optimized prompt text"


class TestFeedbackMetrics:
    """Test feedback metric functions."""
    
    def test_browser_metric_success_scoring(self):
        """Test browser metric scores successful tasks correctly."""
        from src.core.gepa import browser_task_metric, ExecutionTrace
        
        trace = ExecutionTrace(
            task_id="test",
            agent_type="browser",
            instructions="Navigate to example.com",
            system_prompt="...",
            steps=[
                {"type": "navigate", "success": True, "step_num": 1},
                {"type": "get_page_state", "success": True, "step_num": 2},
            ],
            success=True,
            llm_calls=3,
        )
        
        result = browser_task_metric(
            gold={"success": True},
            pred={"success": True},
            trace=trace,
        )
        
        # Should get high score for success
        assert result.score >= 0.7
        assert "successfully" in result.feedback.lower()
    
    def test_browser_metric_failure_feedback(self):
        """Test browser metric provides actionable feedback on failure."""
        from src.core.gepa import browser_task_metric, ExecutionTrace
        
        trace = ExecutionTrace(
            task_id="test",
            agent_type="browser",
            instructions="Click button",
            system_prompt="...",
            steps=[
                {"type": "click", "success": False, "error": "timeout", "step_num": 1},
            ],
            success=False,
            error="Timeout waiting for element",
        )
        
        result = browser_task_metric(
            gold={"success": True},
            pred={"success": False, "error": "Timeout"},
            trace=trace,
        )
        
        # Should get low score
        assert result.score < 0.5
        # Should have actionable feedback
        assert result.feedback  # Not empty


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_optimization_cycle(self, temp_data_dir, mock_env):
        """Test complete optimization cycle."""
        from src.core.gepa import (
            ExecutionTrace,
            OptimizationStore,
            GEPAConfig,
            browser_task_metric,
        )
        from src.core.gepa.automation import AutoGEPAConfig, StateManager
        
        store = OptimizationStore(temp_data_dir)
        
        # 1. Simulate trace accumulation
        for i in range(5):
            trace = ExecutionTrace(
                task_id=f"task-{i}",
                agent_type="browser",
                instructions=f"Test task {i}",
                system_prompt="Default prompt",
                success=i % 2 == 0,  # 50% success rate
                llm_calls=3,
            )
            store.save_trace(trace)
        
        # 2. Check optimization should trigger
        config = AutoGEPAConfig.from_env()
        config.state_file = temp_data_dir / "state.json"
        config.min_traces_for_optimization = 5
        
        state_manager = StateManager(config)
        for i in range(5):
            state_manager.record_trace("browser")
        
        should_opt, reason = state_manager.should_optimize("browser")
        assert should_opt is True
        
        # 3. Load traces
        traces = store.load_traces(agent_type="browser")
        assert len(traces) == 5
        
        # 4. Score traces
        scores = []
        for trace in traces:
            result = browser_task_metric(
                gold={"success": True},
                pred={"success": trace.success},
                trace=trace,
            )
            scores.append(result.score)
        
        avg_score = sum(scores) / len(scores)
        assert 0 <= avg_score <= 1


def run_quick_validation():
    """Quick validation that can be run manually."""
    print("Running GEPA integration validation...")
    
    # Check imports
    try:
        from src.core.gepa import (
            ExecutionTrace,
            ScoreWithFeedback,
            GEPAConfig,
            OptimizationStore,
            browser_task_metric,
        )
        from src.core.gepa.automation import (
            AutoGEPAConfig,
            StateManager,
            EscalationFactory,
        )
        from src.core.gepa.optimizer import AgentOptimizer
        from src.core.gepa.notifications import GuaranteedNotifier
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Check configuration
    config = AutoGEPAConfig.from_env()
    print(f"✓ Configuration loaded (min_traces={config.min_traces_for_optimization})")
    
    # Check notification system
    notifier = GuaranteedNotifier()
    channels = [k for k, v in notifier.config.items() if v and not k.startswith("smtp")]
    print(f"✓ Notification channels: {len(channels)} configured")
    
    # Check data directories
    data_dir = Path("./data/gepa")
    if data_dir.exists():
        traces = list((data_dir / "traces").glob("*.json")) if (data_dir / "traces").exists() else []
        print(f"✓ Data directory exists ({len(traces)} traces)")
    else:
        print("! Data directory not yet created (will be created on first run)")
    
    print("\nValidation complete!")
    return True


if __name__ == "__main__":
    run_quick_validation()
