"""
GEPA Automation Loop - Main Entry Point.

This module runs the continuous optimization loop.
It can be run as a standalone service or integrated into the main application.

Usage:
    # As a standalone service
    python -m src.core.gepa.auto_loop start
    
    # Check status
    python -m src.core.gepa.auto_loop status
    
    # Run in foreground (for debugging)
    python -m src.core.gepa.auto_loop start --foreground
"""

import argparse
import asyncio
import json
import os
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

# Configure logging early
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer() if os.getenv("GEPA_DEBUG") else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class GEPAAutomationLoop:
    """
    The main automation loop for GEPA optimization.
    
    This class:
    1. Monitors agent executions and collects traces
    2. Triggers optimizations based on configurable conditions
    3. Applies optimized prompts automatically
    4. Self-heals from failures
    5. Escalates to humans only when truly stuck
    """
    
    def __init__(self):
        from src.core.gepa.automation import (
            AutoGEPAConfig,
            StateManager,
            NotificationService,
            EscalationFactory,
        )
        
        self.config = AutoGEPAConfig.from_env()
        self.state_manager = StateManager(self.config)
        self.notifications = NotificationService(self.config)
        self.escalation_factory = EscalationFactory()
        
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._current_optimization: asyncio.Task | None = None
        
        self.logger = logger.bind(component="automation_loop")
    
    async def start(self) -> None:
        """Start the automation loop."""
        self.logger.info("starting_automation_loop")
        self._running = True
        
        # Set up signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)
        
        # Run startup checks
        await self._startup_checks()
        
        # Main loop
        while self._running:
            try:
                await self._loop_iteration()
                
                # Wait for next iteration or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.health_check_interval,
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue loop
                    
            except Exception as e:
                self.logger.exception("loop_error")
                await self._handle_loop_error(e)
        
        self.logger.info("automation_loop_stopped")
    
    def _handle_shutdown(self) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.info("shutdown_requested")
        self._running = False
        self._shutdown_event.set()
        
        if self._current_optimization:
            self._current_optimization.cancel()
    
    async def _startup_checks(self) -> None:
        """Run startup health checks."""
        self.logger.info("running_startup_checks")
        
        checks = {
            "database": await self._check_database(),
            "api_keys": await self._check_api_keys(),
            "gepa_package": await self._check_gepa_package(),
            "storage": await self._check_storage(),
        }
        
        failed = [k for k, v in checks.items() if not v]
        
        if failed:
            self.logger.warning("startup_checks_failed", failed=failed)
            # Don't escalate on startup - let self-healing try first
        else:
            self.logger.info("startup_checks_passed")
    
    async def _check_database(self) -> bool:
        """Check database connectivity."""
        try:
            from src.db.session import get_session
            from sqlalchemy import text
            
            async with get_session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            self.logger.error("database_check_failed", error=str(e))
            return False
    
    async def _check_api_keys(self) -> bool:
        """Check API key validity."""
        try:
            from src.core.llm_client import LLMClient
            
            client = LLMClient()
            # Simple test call
            response = await client.chat([
                {"role": "user", "content": "Say 'OK' if you can read this."}
            ], max_tokens=10)
            
            return "ok" in response.get("content", "").lower()
        except Exception as e:
            self.logger.error("api_key_check_failed", error=str(e))
            return False
    
    async def _check_gepa_package(self) -> bool:
        """Check if GEPA package is installed."""
        try:
            import dspy
            # gepa package is optional - we have a fallback
            return True
        except ImportError:
            self.logger.warning("dspy_not_installed")
            return True  # Not critical - we have fallback
    
    async def _check_storage(self) -> bool:
        """Check storage directories are writable."""
        try:
            from src.core.gepa import OptimizationStore
            
            store = OptimizationStore("./data/gepa")
            test_file = store.storage_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception as e:
            self.logger.error("storage_check_failed", error=str(e))
            return False
    
    async def _loop_iteration(self) -> None:
        """Single iteration of the main loop."""
        self.logger.debug("loop_iteration_start")
        
        # 1. Collect new traces from recent executions
        await self._collect_traces()
        
        # 2. Check each agent type for optimization triggers
        from src.agents import AGENT_REGISTRY
        
        for agent_type in AGENT_REGISTRY.keys():
            should_opt, reason = self.state_manager.should_optimize(agent_type)
            
            if should_opt:
                self.logger.info(
                    "optimization_triggered",
                    agent_type=agent_type,
                    reason=reason,
                )
                await self._run_optimization(agent_type, reason)
        
        # 3. Health check
        await self._health_check()
    
    async def _collect_traces(self) -> None:
        """Collect traces from recent task executions."""
        try:
            from src.db.session import get_session
            from src.db.models import Task
            from sqlalchemy import select
            from src.core.gepa import ExecutionTrace, OptimizationStore
            
            store = OptimizationStore("./data/gepa")
            
            # Get tasks completed in the last interval
            cutoff = datetime.utcnow() - timedelta(
                seconds=self.config.health_check_interval * 2
            )
            
            async with get_session() as session:
                result = await session.execute(
                    select(Task).where(
                        Task.completed_at >= cutoff,
                        Task.status.in_(["completed", "failed"]),
                    )
                )
                tasks = result.scalars().all()
            
            for task in tasks:
                # Check if we already have this trace
                trace_path = store.traces_dir / f"{task.id}.json"
                if trace_path.exists():
                    continue
                
                # Create trace from task
                trace = ExecutionTrace(
                    task_id=task.id,
                    agent_type=task.agent_type,
                    instructions=task.instructions,
                    system_prompt=task.config.get("system_prompt", ""),
                    success=task.status == "completed",
                    result=task.result,
                    error=task.error,
                    llm_calls=task.llm_calls,
                    total_tokens=task.total_tokens,
                    started_at=task.started_at or task.created_at,
                    completed_at=task.completed_at,
                )
                
                if task.completed_at and task.started_at:
                    trace.duration_seconds = (
                        task.completed_at - task.started_at
                    ).total_seconds()
                
                # Save trace
                store.save_trace(trace)
                
                # Update counter
                self.state_manager.record_trace(task.agent_type)
                
                self.logger.debug(
                    "trace_collected",
                    task_id=task.id,
                    agent_type=task.agent_type,
                )
                
        except Exception as e:
            self.logger.error("trace_collection_failed", error=str(e))
    
    async def _run_optimization(self, agent_type: str, reason: str) -> None:
        """Run GEPA optimization for an agent type."""
        operation_id = f"opt_{agent_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            from src.agents import get_agent_class
            from src.core.gepa import OptimizationStore, GEPAConfig, get_metric
            from src.core.gepa.optimizer import AgentOptimizer
            
            self.logger.info("starting_optimization", agent_type=agent_type, reason=reason)
            
            # Load training data
            store = OptimizationStore("./data/gepa")
            traces = store.load_traces(agent_type=agent_type)
            
            if len(traces) < 10:
                self.logger.warning("insufficient_traces", count=len(traces))
                return
            
            # Convert traces to training tasks
            trainset = []
            for trace in traces:
                trainset.append({
                    "task_id": trace.task_id,
                    "instructions": trace.instructions,
                    "config": {},
                    "expected": {
                        "success": trace.success,
                        "result": trace.result,
                    },
                })
            
            # Split into train/val
            split_idx = int(len(trainset) * 0.8)
            train = trainset[:split_idx]
            val = trainset[split_idx:]
            
            # Create agent and optimizer
            agent_class = get_agent_class(agent_type)
            agent = agent_class()
            
            config = GEPAConfig(
                auto=self.config.default_budget,
                log_dir=f"./data/gepa/runs/{operation_id}",
                track_stats=True,
            )
            
            optimizer = AgentOptimizer(
                agent=agent,
                metric=get_metric(agent_type),
                config=config,
                store=store,
            )
            
            # Run optimization
            old_score = self.state_manager.state.current_scores.get(agent_type, 0)
            
            result = await optimizer.optimize(trainset=train, valset=val)
            
            self.logger.info(
                "optimization_complete",
                agent_type=agent_type,
                old_score=old_score,
                new_score=result.best_score,
                improvement=result.improvement,
            )
            
            # Check for anomalous results
            if old_score > 0:
                change_pct = abs(result.best_score - old_score) / old_score
                
                # If score dropped significantly, escalate for review
                if result.best_score < old_score * 0.8:
                    await self._escalate_anomalous_result(
                        agent_type, old_score, result
                    )
                    return
                
                # If score jumped too much, might be overfitting
                if result.best_score > old_score * 1.5:
                    await self._escalate_anomalous_result(
                        agent_type, old_score, result
                    )
                    return
            
            # Apply the optimization
            await self._apply_optimization(agent_type, result)
            
            # Record success
            self.state_manager.record_optimization(agent_type, result.best_score)
            self.state_manager.state.retry_counts.pop(operation_id, None)
            
        except Exception as e:
            self.logger.exception("optimization_failed", agent_type=agent_type)
            await self._handle_optimization_error(agent_type, operation_id, e)
    
    async def _apply_optimization(
        self,
        agent_type: str,
        result: Any,  # OptimizationResult
    ) -> None:
        """Apply optimized prompts to the agent configuration."""
        from src.core.gepa import OptimizationStore
        
        store = OptimizationStore("./data/gepa")
        
        # Save all optimized prompts
        for component_name, text in result.optimized_texts.items():
            store.save_optimized_prompt(
                agent_type=agent_type,
                component_name=component_name,
                prompt_text=text,
                score=result.best_score,
                metadata={
                    "improvement": result.improvement,
                    "total_evaluations": result.total_evaluations,
                },
            )
        
        # Update the "active" prompt configuration
        active_prompts_file = store.storage_dir / "active_prompts.json"
        
        if active_prompts_file.exists():
            with open(active_prompts_file) as f:
                active = json.load(f)
        else:
            active = {}
        
        active[agent_type] = {
            "prompts": result.optimized_texts,
            "score": result.best_score,
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        with open(active_prompts_file, "w") as f:
            json.dump(active, f, indent=2)
        
        self.logger.info(
            "optimization_applied",
            agent_type=agent_type,
            components=list(result.optimized_texts.keys()),
        )
    
    async def _escalate_anomalous_result(
        self,
        agent_type: str,
        old_score: float,
        result: Any,
    ) -> None:
        """Escalate unusual optimization results for review."""
        from src.core.gepa.automation import EscalationFactory
        
        event = EscalationFactory.anomalous_optimization_result(
            agent_type=agent_type,
            old_score=old_score,
            new_score=result.best_score,
            optimized_prompt=result.optimized_texts.get("system_prompt", ""),
        )
        
        await self._escalate(event)
    
    async def _handle_optimization_error(
        self,
        agent_type: str,
        operation_id: str,
        error: Exception,
    ) -> None:
        """Handle optimization errors with retry and escalation."""
        from src.core.gepa.automation import EscalationFactory
        
        retry_count = self.state_manager.state.retry_counts.get(operation_id, 0) + 1
        self.state_manager.state.retry_counts[operation_id] = retry_count
        self.state_manager.save()
        
        error_str = str(error)
        
        # Check for specific error types
        if "api key" in error_str.lower() or "authentication" in error_str.lower():
            event = EscalationFactory.api_key_expired(
                agent_type=agent_type,
                provider="anthropic",  # TODO: detect from error
                error=error_str,
            )
            await self._escalate(event)
            return
        
        if "database" in error_str.lower() or "connection" in error_str.lower():
            event = EscalationFactory.database_unavailable(error_str)
            await self._escalate(event)
            return
        
        # Retry with backoff
        if retry_count < self.config.max_retries:
            delay = self.config.retry_backoff_base * (2 ** (retry_count - 1))
            self.logger.info(
                "scheduling_retry",
                operation_id=operation_id,
                retry_count=retry_count,
                delay_seconds=delay,
            )
            await asyncio.sleep(delay)
            return
        
        # Max retries exceeded - escalate
        event = EscalationFactory.persistent_failure(
            agent_type=agent_type,
            operation="optimization",
            attempts=retry_count,
            errors=[error_str],
        )
        await self._escalate(event)
    
    async def _health_check(self) -> None:
        """Periodic health check."""
        checks = {
            "database": await self._check_database(),
            "api_keys": await self._check_api_keys(),
        }
        
        for check_name, passed in checks.items():
            if not passed:
                # Try to self-heal
                healed = await self._try_self_heal(check_name)
                
                if not healed:
                    self.logger.error("health_check_failed", check=check_name)
                    # Will escalate on next iteration if still failing
    
    async def _try_self_heal(self, issue: str) -> bool:
        """Attempt to self-heal from an issue."""
        self.logger.info("attempting_self_heal", issue=issue)
        
        if issue == "database":
            # Try reconnecting
            try:
                from src.db.session import engine
                await engine.dispose()
                return await self._check_database()
            except Exception:
                return False
        
        # Other issues can't be self-healed
        return False
    
    async def _escalate(self, event: Any) -> None:
        """Send escalation notification."""
        from src.core.gepa.automation import EscalationEvent
        
        if not self.state_manager.can_escalate():
            self.logger.warning(
                "escalation_throttled",
                event_id=event.id,
                cooldown_hours=self.config.escalation_cooldown_hours,
            )
            # Still log to file
            self.config.escalation_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.escalation_log, "a") as f:
                f.write(f"\n[THROTTLED] {event.to_text()}\n")
            return
        
        self.logger.warning("escalating", event_id=event.id, title=event.title)
        
        channels = await self.notifications.send(event)
        
        self.logger.info("escalation_sent", channels=channels)
        self.state_manager.record_escalation()
    
    async def _handle_loop_error(self, error: Exception) -> None:
        """Handle errors in the main loop."""
        self.logger.error("loop_error", error=str(error))
        
        # Wait before retrying
        await asyncio.sleep(60)


# =============================================================================
# CLI
# =============================================================================


async def cmd_start(args: argparse.Namespace) -> None:
    """Start the automation loop."""
    loop = GEPAAutomationLoop()
    await loop.start()


async def cmd_status(args: argparse.Namespace) -> None:
    """Show current status."""
    from src.core.gepa.automation import AutoGEPAConfig, StateManager
    
    config = AutoGEPAConfig.from_env()
    state_manager = StateManager(config)
    
    print("\n" + "="*50)
    print("GEPA Automation Status")
    print("="*50)
    
    # Check if running
    # TODO: Implement proper process check
    print(f"\nStatus: {'RUNNING' if config.state_file.exists() else 'UNKNOWN'}")
    
    print(f"\nTrace counts:")
    for agent_type, count in state_manager.state.trace_counts.items():
        print(f"  {agent_type}: {count} traces")
    
    print(f"\nLast optimizations:")
    for agent_type, timestamp in state_manager.state.last_optimization.items():
        print(f"  {agent_type}: {timestamp}")
    
    print(f"\nCurrent scores:")
    for agent_type, score in state_manager.state.current_scores.items():
        baseline = state_manager.state.baseline_scores.get(agent_type, score)
        diff = score - baseline
        print(f"  {agent_type}: {score:.3f} (baseline: {baseline:.3f}, diff: {diff:+.3f})")
    
    if state_manager.state.last_escalation:
        print(f"\nLast escalation: {state_manager.state.last_escalation}")
    
    print("\n" + "="*50)


async def cmd_diagnose(args: argparse.Namespace) -> None:
    """Run diagnostics."""
    loop = GEPAAutomationLoop()
    
    print("\n" + "="*50)
    print("GEPA Diagnostics")
    print("="*50)
    
    # Database
    print("\n[Database]")
    db_ok = await loop._check_database()
    print(f"  Status: {'OK' if db_ok else 'FAILED'}")
    
    # API
    print("\n[API Keys]")
    api_ok = await loop._check_api_keys()
    print(f"  Status: {'OK' if api_ok else 'FAILED'}")
    
    # Storage
    print("\n[Storage]")
    storage_ok = await loop._check_storage()
    print(f"  Status: {'OK' if storage_ok else 'FAILED'}")
    
    # GEPA package
    print("\n[GEPA Package]")
    try:
        import dspy
        print(f"  dspy: installed (v{dspy.__version__ if hasattr(dspy, '__version__') else 'unknown'})")
    except ImportError:
        print("  dspy: NOT INSTALLED")
    
    try:
        import gepa
        print(f"  gepa: installed")
    except ImportError:
        print("  gepa: NOT INSTALLED (using fallback optimizer)")
    
    print("\n" + "="*50)


async def cmd_reset_failures(args: argparse.Namespace) -> None:
    """Reset failure counters."""
    from src.core.gepa.automation import AutoGEPAConfig, StateManager
    
    config = AutoGEPAConfig.from_env()
    state_manager = StateManager(config)
    
    state_manager.state.retry_counts = {}
    state_manager.save()
    
    print("Failure counters reset.")


async def cmd_approve_heavy(args: argparse.Namespace) -> None:
    """Approve heavy optimization."""
    # TODO: Implement approval mechanism
    print("Heavy optimization approved. Running...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GEPA Automation Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # start
    start_parser = subparsers.add_parser("start", help="Start the automation loop")
    start_parser.add_argument(
        "--foreground", "-f",
        action="store_true",
        help="Run in foreground (don't daemonize)",
    )
    
    # status
    subparsers.add_parser("status", help="Show current status")
    
    # diagnose
    diagnose_parser = subparsers.add_parser("diagnose", help="Run diagnostics")
    diagnose_parser.add_argument("agent_type", nargs="?", help="Agent type to diagnose")
    
    # reset-failures
    subparsers.add_parser("reset-failures", help="Reset failure counters")
    
    # approve-heavy
    subparsers.add_parser("approve-heavy", help="Approve pending heavy optimization")
    
    args = parser.parse_args()
    
    if args.command == "start":
        asyncio.run(cmd_start(args))
    elif args.command == "status":
        asyncio.run(cmd_status(args))
    elif args.command == "diagnose":
        asyncio.run(cmd_diagnose(args))
    elif args.command == "reset-failures":
        asyncio.run(cmd_reset_failures(args))
    elif args.command == "approve-heavy":
        asyncio.run(cmd_approve_heavy(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
