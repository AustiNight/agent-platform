"""
Automated GEPA Feedback Loop.

This module implements a fully autonomous optimization system that:
1. Continuously collects execution traces from all agent runs
2. Automatically triggers optimization when performance drops or data accumulates
3. Applies optimized prompts without human intervention
4. Self-heals from failures with exponential backoff
5. Escalates to human ONLY when truly stuck (with detailed instructions)

Human intervention required only for:
- API key rotation/expiration
- Infrastructure failures (database down, etc.)
- Budget approval for heavy optimization runs
- Reviewing optimization results that seem anomalous
"""

import asyncio
import json
import os
import smtplib
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import httpx
import structlog

from src.core.config import settings
from src.core.gepa import (
    ExecutionTrace,
    ScoreWithFeedback,
    FeedbackMetric,
    GEPAConfig,
    OptimizationStore,
    get_metric,
)
from src.core.gepa.optimizer import AgentOptimizer, OptimizationResult

logger = structlog.get_logger()


# =============================================================================
# Configuration
# =============================================================================


class EscalationChannel(str, Enum):
    """Channels for human escalation."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"  # Via Twilio
    PUSHOVER = "pushover"
    FILE = "file"  # Fallback: write to file


@dataclass
class AutoGEPAConfig:
    """Configuration for the automated GEPA loop."""
    
    # Trigger conditions
    min_traces_for_optimization: int = 50
    optimization_interval_hours: int = 24
    performance_drop_threshold: float = 0.15  # 15% drop triggers optimization
    
    # Optimization settings
    default_budget: str = "medium"  # "light", "medium", "heavy"
    max_concurrent_optimizations: int = 1
    
    # Self-healing
    max_retries: int = 3
    retry_backoff_base: int = 60  # seconds
    health_check_interval: int = 300  # 5 minutes
    
    # Escalation
    escalation_channels: list[EscalationChannel] = field(
        default_factory=lambda: [EscalationChannel.EMAIL, EscalationChannel.FILE]
    )
    escalation_cooldown_hours: int = 4  # Don't spam
    
    # Notification settings
    email_to: str = ""
    email_from: str = "gepa-automation@agent-platform.local"
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    
    slack_webhook_url: str = ""
    pushover_user_key: str = ""
    pushover_api_token: str = ""
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    twilio_to_number: str = ""
    
    generic_webhook_url: str = ""
    
    # Storage
    state_file: Path = Path("./data/gepa/auto_state.json")
    escalation_log: Path = Path("./data/gepa/escalations.log")
    
    @classmethod
    def from_env(cls) -> "AutoGEPAConfig":
        """Load configuration from environment variables."""
        return cls(
            min_traces_for_optimization=int(os.getenv("GEPA_MIN_TRACES", "50")),
            optimization_interval_hours=int(os.getenv("GEPA_INTERVAL_HOURS", "24")),
            performance_drop_threshold=float(os.getenv("GEPA_PERF_DROP_THRESHOLD", "0.15")),
            default_budget=os.getenv("GEPA_BUDGET", "medium"),
            
            email_to=os.getenv("GEPA_NOTIFY_EMAIL", ""),
            email_from=os.getenv("GEPA_FROM_EMAIL", "gepa-automation@agent-platform.local"),
            smtp_host=os.getenv("SMTP_HOST", "localhost"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            
            slack_webhook_url=os.getenv("GEPA_SLACK_WEBHOOK", ""),
            pushover_user_key=os.getenv("PUSHOVER_USER_KEY", ""),
            pushover_api_token=os.getenv("PUSHOVER_API_TOKEN", ""),
            twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
            twilio_from_number=os.getenv("TWILIO_FROM_NUMBER", ""),
            twilio_to_number=os.getenv("TWILIO_TO_NUMBER", ""),
            
            generic_webhook_url=os.getenv("GEPA_WEBHOOK_URL", ""),
        )


# =============================================================================
# State Management
# =============================================================================


@dataclass
class OptimizationState:
    """Persistent state for the automation loop."""
    
    last_optimization: dict[str, str] = field(default_factory=dict)  # agent_type -> ISO timestamp
    last_escalation: str | None = None
    pending_escalations: list[dict] = field(default_factory=list)
    
    # Performance tracking
    baseline_scores: dict[str, float] = field(default_factory=dict)  # agent_type -> score
    current_scores: dict[str, float] = field(default_factory=dict)
    
    # Retry tracking
    retry_counts: dict[str, int] = field(default_factory=dict)  # operation_id -> count
    
    # Trace counters
    trace_counts: dict[str, int] = field(default_factory=dict)  # agent_type -> count since last opt
    
    def to_dict(self) -> dict:
        return {
            "last_optimization": self.last_optimization,
            "last_escalation": self.last_escalation,
            "pending_escalations": self.pending_escalations,
            "baseline_scores": self.baseline_scores,
            "current_scores": self.current_scores,
            "retry_counts": self.retry_counts,
            "trace_counts": self.trace_counts,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "OptimizationState":
        return cls(
            last_optimization=data.get("last_optimization", {}),
            last_escalation=data.get("last_escalation"),
            pending_escalations=data.get("pending_escalations", []),
            baseline_scores=data.get("baseline_scores", {}),
            current_scores=data.get("current_scores", {}),
            retry_counts=data.get("retry_counts", {}),
            trace_counts=data.get("trace_counts", {}),
        )


class StateManager:
    """Manages persistent state for the automation loop."""
    
    def __init__(self, config: AutoGEPAConfig):
        self.config = config
        self.state = OptimizationState()
        self._load()
    
    def _load(self) -> None:
        """Load state from disk."""
        if self.config.state_file.exists():
            try:
                with open(self.config.state_file) as f:
                    self.state = OptimizationState.from_dict(json.load(f))
                logger.info("state_loaded", path=str(self.config.state_file))
            except Exception as e:
                logger.warning("state_load_failed", error=str(e))
    
    def save(self) -> None:
        """Persist state to disk."""
        self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
    
    def record_trace(self, agent_type: str) -> int:
        """Record a new trace and return count since last optimization."""
        self.state.trace_counts[agent_type] = self.state.trace_counts.get(agent_type, 0) + 1
        self.save()
        return self.state.trace_counts[agent_type]
    
    def record_optimization(self, agent_type: str, score: float) -> None:
        """Record a completed optimization."""
        self.state.last_optimization[agent_type] = datetime.utcnow().isoformat()
        self.state.trace_counts[agent_type] = 0
        self.state.current_scores[agent_type] = score
        
        # Update baseline if this is better
        if score > self.state.baseline_scores.get(agent_type, 0):
            self.state.baseline_scores[agent_type] = score
        
        self.save()
    
    def should_optimize(self, agent_type: str) -> tuple[bool, str]:
        """Check if optimization should run for this agent type."""
        trace_count = self.state.trace_counts.get(agent_type, 0)
        last_opt = self.state.last_optimization.get(agent_type)
        
        # Check trace count
        if trace_count >= self.config.min_traces_for_optimization:
            return True, f"Accumulated {trace_count} traces"
        
        # Check time since last optimization
        if last_opt:
            last_opt_dt = datetime.fromisoformat(last_opt)
            hours_since = (datetime.utcnow() - last_opt_dt).total_seconds() / 3600
            if hours_since >= self.config.optimization_interval_hours and trace_count >= 10:
                return True, f"{hours_since:.1f} hours since last optimization"
        
        # Check performance drop
        baseline = self.state.baseline_scores.get(agent_type, 0)
        current = self.state.current_scores.get(agent_type, baseline)
        if baseline > 0 and current < baseline * (1 - self.config.performance_drop_threshold):
            return True, f"Performance dropped from {baseline:.3f} to {current:.3f}"
        
        return False, "No optimization trigger"
    
    def can_escalate(self) -> bool:
        """Check if we're allowed to escalate (respects cooldown)."""
        if not self.state.last_escalation:
            return True
        
        last_esc = datetime.fromisoformat(self.state.last_escalation)
        hours_since = (datetime.utcnow() - last_esc).total_seconds() / 3600
        return hours_since >= self.config.escalation_cooldown_hours
    
    def record_escalation(self) -> None:
        """Record that an escalation was sent."""
        self.state.last_escalation = datetime.utcnow().isoformat()
        self.save()


# =============================================================================
# Notification System
# =============================================================================


@dataclass
class EscalationEvent:
    """An event requiring human attention."""
    
    id: str
    severity: str  # "critical", "warning", "info"
    title: str
    description: str
    agent_type: str | None
    error: str | None
    
    # What the human needs to do
    required_actions: list[str]
    
    # Context
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_text(self) -> str:
        """Format as plain text."""
        lines = [
            f"{'='*60}",
            f"GEPA AUTOMATION ALERT - {self.severity.upper()}",
            f"{'='*60}",
            f"",
            f"Title: {self.title}",
            f"Time: {self.timestamp.isoformat()}",
            f"Event ID: {self.id}",
            f"",
            f"DESCRIPTION:",
            f"{self.description}",
            f"",
        ]
        
        if self.error:
            lines.extend([
                f"ERROR DETAILS:",
                f"{self.error}",
                f"",
            ])
        
        lines.extend([
            f"{'='*60}",
            f"REQUIRED ACTIONS (in order):",
            f"{'='*60}",
        ])
        
        for i, action in enumerate(self.required_actions, 1):
            lines.append(f"")
            lines.append(f"STEP {i}:")
            lines.append(f"{action}")
        
        lines.extend([
            f"",
            f"{'='*60}",
            f"RETURNING TO AUTOMATED MODE:",
            f"{'='*60}",
            f"",
            f"After completing the above steps, the system will automatically:",
            f"1. Detect that the issue is resolved (within 5 minutes)",
            f"2. Resume normal optimization operations",
            f"3. Send a confirmation notification",
            f"",
            f"If you need to manually restart the optimization loop:",
            f"  python -m src.core.gepa.auto_loop start",
            f"",
            f"To check current status:",
            f"  python -m src.core.gepa.auto_loop status",
            f"",
        ])
        
        if self.context:
            lines.extend([
                f"ADDITIONAL CONTEXT:",
                json.dumps(self.context, indent=2, default=str),
            ])
        
        return "\n".join(lines)
    
    def to_html(self) -> str:
        """Format as HTML for email."""
        actions_html = "".join(
            f"<li><strong>Step {i}:</strong><br/><pre>{action}</pre></li>"
            for i, action in enumerate(self.required_actions, 1)
        )
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto;">
            <div style="background: {'#dc3545' if self.severity == 'critical' else '#ffc107'}; 
                        color: {'white' if self.severity == 'critical' else 'black'}; 
                        padding: 20px; border-radius: 5px 5px 0 0;">
                <h1 style="margin: 0;">🤖 GEPA Automation Alert</h1>
                <p style="margin: 5px 0 0 0;">{self.severity.upper()} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            
            <div style="border: 1px solid #ddd; padding: 20px; border-radius: 0 0 5px 5px;">
                <h2>{self.title}</h2>
                <p>{self.description}</p>
                
                {f'<div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0;"><strong>Error:</strong><pre style="white-space: pre-wrap;">{self.error}</pre></div>' if self.error else ''}
                
                <h3 style="color: #dc3545;">⚠️ Required Actions</h3>
                <ol style="line-height: 1.8;">
                    {actions_html}
                </ol>
                
                <h3 style="color: #28a745;">✅ Returning to Automated Mode</h3>
                <p>After completing the above steps:</p>
                <ol>
                    <li>The system will automatically detect resolution (within 5 minutes)</li>
                    <li>Normal optimization operations will resume</li>
                    <li>You'll receive a confirmation notification</li>
                </ol>
                
                <div style="background: #e9ecef; padding: 15px; border-radius: 5px; margin-top: 20px;">
                    <strong>Manual Commands:</strong><br/>
                    <code>python -m src.core.gepa.auto_loop start</code> - Restart loop<br/>
                    <code>python -m src.core.gepa.auto_loop status</code> - Check status
                </div>
                
                <p style="color: #6c757d; font-size: 12px; margin-top: 20px;">
                    Event ID: {self.id}
                </p>
            </div>
        </body>
        </html>
        """


class NotificationService:
    """Multi-channel notification service with guaranteed delivery."""
    
    def __init__(self, config: AutoGEPAConfig):
        self.config = config
        self.logger = logger.bind(service="notifications")
    
    async def send(self, event: EscalationEvent) -> list[str]:
        """
        Send notification through all configured channels.
        
        Uses GuaranteedNotifier to ensure delivery.
        Returns list of successful channels.
        """
        from src.core.gepa.notifications import get_notifier
        
        notifier = get_notifier()
        
        results = await notifier.send(
            title=event.title,
            message=event.to_text(),
            severity=event.severity,
            html_message=event.to_html(),
            context={
                "event_id": event.id,
                "agent_type": event.agent_type,
                "required_actions_count": len(event.required_actions),
                **event.context,
            },
        )
        
        successful = [r.channel for r in results if r.success]
        
        self.logger.info(
            "escalation_sent",
            event_id=event.id,
            channels=successful,
        )
        
        return successful
    
    async def _send_email(self, event: EscalationEvent) -> None:
        """Send email notification."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[GEPA {event.severity.upper()}] {event.title}"
        msg["From"] = self.config.email_from
        msg["To"] = self.config.email_to
        
        msg.attach(MIMEText(event.to_text(), "plain"))
        msg.attach(MIMEText(event.to_html(), "html"))
        
        # Send email
        if self.config.smtp_user:
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.send_message(msg)
        
        self.logger.info("email_sent", to=self.config.email_to)
    
    async def _send_slack(self, event: EscalationEvent) -> None:
        """Send Slack notification."""
        color = "#dc3545" if event.severity == "critical" else "#ffc107"
        
        payload = {
            "attachments": [{
                "color": color,
                "title": f"🤖 GEPA Alert: {event.title}",
                "text": event.description,
                "fields": [
                    {"title": "Severity", "value": event.severity.upper(), "short": True},
                    {"title": "Agent", "value": event.agent_type or "N/A", "short": True},
                ],
                "footer": f"Event ID: {event.id}",
                "ts": int(event.timestamp.timestamp()),
            }],
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Required Actions:*\n" + "\n".join(
                        f"{i}. {action[:200]}..." if len(action) > 200 else f"{i}. {action}"
                        for i, action in enumerate(event.required_actions, 1)
                    )}
                }
            ]
        }
        
        async with httpx.AsyncClient() as client:
            await client.post(self.config.slack_webhook_url, json=payload)
        
        self.logger.info("slack_sent")
    
    async def _send_pushover(self, event: EscalationEvent) -> None:
        """Send Pushover notification (mobile push)."""
        priority = 2 if event.severity == "critical" else 1  # 2 = requires confirmation
        
        data = {
            "token": self.config.pushover_api_token,
            "user": self.config.pushover_user_key,
            "title": f"GEPA: {event.title}",
            "message": f"{event.description}\n\nSteps: {len(event.required_actions)} actions required",
            "priority": priority,
            "sound": "siren" if event.severity == "critical" else "pushover",
            "url": f"file://{self.config.escalation_log}",
            "url_title": "View Full Details",
        }
        
        if priority == 2:
            data["retry"] = 60
            data["expire"] = 3600
        
        async with httpx.AsyncClient() as client:
            await client.post("https://api.pushover.net/1/messages.json", data=data)
        
        self.logger.info("pushover_sent")
    
    async def _send_sms(self, event: EscalationEvent) -> None:
        """Send SMS via Twilio."""
        from twilio.rest import Client
        
        client = Client(self.config.twilio_account_sid, self.config.twilio_auth_token)
        
        message = client.messages.create(
            body=f"GEPA {event.severity.upper()}: {event.title}. {len(event.required_actions)} actions required. Check email for details.",
            from_=self.config.twilio_from_number,
            to=self.config.twilio_to_number,
        )
        
        self.logger.info("sms_sent", sid=message.sid)
    
    async def _send_webhook(self, event: EscalationEvent) -> None:
        """Send to generic webhook."""
        payload = {
            "event_id": event.id,
            "severity": event.severity,
            "title": event.title,
            "description": event.description,
            "agent_type": event.agent_type,
            "error": event.error,
            "required_actions": event.required_actions,
            "context": event.context,
            "timestamp": event.timestamp.isoformat(),
        }
        
        async with httpx.AsyncClient() as client:
            await client.post(
                self.config.generic_webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        
        self.logger.info("webhook_sent")
    
    async def _write_file(self, event: EscalationEvent) -> None:
        """Write to escalation log file (guaranteed delivery)."""
        self.config.escalation_log.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config.escalation_log, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(event.to_text())
            f.write(f"\n{'='*80}\n\n")
        
        self.logger.info("file_written", path=str(self.config.escalation_log))


# =============================================================================
# Escalation Event Factory
# =============================================================================


class EscalationFactory:
    """Creates detailed escalation events with step-by-step instructions."""
    
    @staticmethod
    def api_key_expired(agent_type: str, provider: str, error: str) -> EscalationEvent:
        return EscalationEvent(
            id=str(uuid4())[:8],
            severity="critical",
            title=f"API Key Expired: {provider}",
            description=f"The {provider} API key has expired or is invalid. GEPA optimization for {agent_type} agents cannot proceed.",
            agent_type=agent_type,
            error=error,
            required_actions=[
                f"""1. Generate a new API key from {provider}:
   - Go to https://console.anthropic.com/settings/keys (for Anthropic)
   - Or https://platform.openai.com/api-keys (for OpenAI)
   - Click "Create new secret key"
   - Copy the key immediately (it won't be shown again)""",
                
                f"""2. Update the environment variable:
   
   Option A - Direct environment (temporary):
   $ export {provider.upper()}_API_KEY="sk-your-new-key-here"
   
   Option B - Update .env file (permanent):
   $ nano /path/to/agent-platform/.env
   # Find the line: {provider.upper()}_API_KEY=...
   # Replace with your new key
   # Save and exit (Ctrl+X, Y, Enter)""",
                
                """3. Restart the GEPA automation service:
   $ systemctl restart gepa-automation
   # Or if running manually:
   $ pkill -f "gepa.auto_loop"
   $ python -m src.core.gepa.auto_loop start &""",
                
                """4. Verify the fix:
   $ python -m src.core.gepa.auto_loop status
   # Should show "Status: RUNNING" and "API: OK" """,
            ],
        )
    
    @staticmethod
    def database_unavailable(error: str) -> EscalationEvent:
        return EscalationEvent(
            id=str(uuid4())[:8],
            severity="critical",
            title="Database Connection Failed",
            description="Cannot connect to the database. Trace collection and optimization history are unavailable.",
            agent_type=None,
            error=error,
            required_actions=[
                """1. Check if the database service is running:
   
   For SQLite (default):
   $ ls -la ./agent_platform.db
   # If missing, the database needs to be recreated
   
   For PostgreSQL:
   $ docker-compose ps postgres
   # Should show "Up"
   $ docker-compose logs postgres --tail=50
   # Check for errors""",
                
                """2. If database container is down, restart it:
   $ docker-compose up -d postgres
   $ sleep 10  # Wait for startup
   $ docker-compose ps  # Verify it's running""",
                
                """3. If database is corrupted, restore from backup:
   
   For SQLite:
   $ cp ./data/backups/agent_platform.db.latest ./agent_platform.db
   
   For PostgreSQL:
   $ docker-compose exec postgres pg_restore -U agent_platform -d agent_platform /backups/latest.dump""",
                
                """4. Run migrations to ensure schema is current:
   $ alembic upgrade head""",
                
                """5. Restart the GEPA automation:
   $ python -m src.core.gepa.auto_loop start""",
            ],
        )
    
    @staticmethod
    def optimization_budget_exhausted(agent_type: str, estimated_cost: float) -> EscalationEvent:
        return EscalationEvent(
            id=str(uuid4())[:8],
            severity="warning",
            title="Optimization Budget Approval Required",
            description=f"GEPA wants to run a heavy optimization for {agent_type} agents. Estimated cost: ${estimated_cost:.2f}",
            agent_type=agent_type,
            error=None,
            required_actions=[
                f"""1. Review the optimization request:
   
   Agent Type: {agent_type}
   Estimated Cost: ${estimated_cost:.2f}
   Accumulated Traces: Check ./data/gepa/traces/{agent_type}/
   
   This optimization is recommended because performance metrics
   have dropped significantly or substantial new training data
   has accumulated.""",
                
                """2. To APPROVE and run the optimization:
   $ python -m src.core.gepa.auto_loop approve-heavy
   
   This will:
   - Run the heavy optimization (may take 30-60 minutes)
   - Apply the best prompts automatically
   - Resume normal operations""",
                
                """3. To SKIP this optimization:
   $ python -m src.core.gepa.auto_loop skip-heavy
   
   This will:
   - Reset the trace counter
   - Resume normal operations with current prompts
   - Trigger again when more traces accumulate""",
                
                """4. To set auto-approval for future heavy optimizations:
   $ export GEPA_AUTO_APPROVE_HEAVY=true
   # Add to .env for permanent setting""",
            ],
            context={
                "estimated_cost_usd": estimated_cost,
                "agent_type": agent_type,
            },
        )
    
    @staticmethod
    def anomalous_optimization_result(
        agent_type: str,
        old_score: float,
        new_score: float,
        optimized_prompt: str,
    ) -> EscalationEvent:
        return EscalationEvent(
            id=str(uuid4())[:8],
            severity="warning",
            title="Optimization Result Review Required",
            description=f"GEPA optimization for {agent_type} produced unusual results. Score changed from {old_score:.3f} to {new_score:.3f}.",
            agent_type=agent_type,
            error=None,
            required_actions=[
                f"""1. Review the optimization result:
   
   Previous Score: {old_score:.3f}
   New Score: {new_score:.3f}
   Change: {new_score - old_score:+.3f} ({(new_score - old_score) / max(0.001, old_score) * 100:+.1f}%)
   
   The new optimized prompt is shown below. Check if it makes sense.""",
                
                f"""2. Optimized prompt preview:
   
   ---BEGIN PROMPT---
   {optimized_prompt[:1000]}{'...' if len(optimized_prompt) > 1000 else ''}
   ---END PROMPT---""",
                
                """3. To ACCEPT the new prompt:
   $ python -m src.core.gepa.auto_loop accept-prompt
   
   This will:
   - Save the prompt as the new baseline
   - Resume normal operations""",
                
                """4. To REJECT and revert:
   $ python -m src.core.gepa.auto_loop reject-prompt
   
   This will:
   - Keep the previous prompt
   - Mark this optimization as failed
   - Resume normal operations""",
                
                """5. To run a test before deciding:
   $ python -m src.core.gepa.auto_loop test-prompt --tasks 5
   
   This will run 5 test tasks with the new prompt and show results.""",
            ],
            context={
                "old_score": old_score,
                "new_score": new_score,
                "score_change": new_score - old_score,
                "prompt_length": len(optimized_prompt),
            },
        )
    
    @staticmethod
    def persistent_failure(
        agent_type: str,
        operation: str,
        attempts: int,
        errors: list[str],
    ) -> EscalationEvent:
        return EscalationEvent(
            id=str(uuid4())[:8],
            severity="critical",
            title=f"Persistent Failure: {operation}",
            description=f"Operation '{operation}' for {agent_type} has failed {attempts} times consecutively. Manual intervention required.",
            agent_type=agent_type,
            error="\n---\n".join(errors[-3:]),  # Last 3 errors
            required_actions=[
                f"""1. Diagnose the issue:
   
   $ python -m src.core.gepa.auto_loop diagnose {agent_type}
   
   This will run diagnostics and show:
   - API connectivity status
   - Database status
   - Recent error patterns
   - Recommended fixes""",
                
                """2. Check the detailed logs:
   $ tail -100 ./data/gepa/logs/gepa-automation.log
   $ grep -i error ./data/gepa/logs/gepa-automation.log | tail -20""",
                
                """3. Common fixes:
   
   If "rate limit" errors:
   $ export GEPA_RATE_LIMIT_DELAY=60  # Add 60s delay between API calls
   
   If "context length" errors:
   $ export GEPA_MAX_TRACES_PER_BATCH=10  # Reduce batch size
   
   If "timeout" errors:
   $ export GEPA_TIMEOUT_SECONDS=300  # Increase timeout""",
                
                """4. Reset the failure counter and retry:
   $ python -m src.core.gepa.auto_loop reset-failures
   $ python -m src.core.gepa.auto_loop start""",
                
                """5. If the issue persists, run in debug mode:
   $ GEPA_DEBUG=true python -m src.core.gepa.auto_loop start --foreground
   
   This will show detailed output for each operation.""",
            ],
            context={
                "attempts": attempts,
                "operation": operation,
            },
        )
    
    @staticmethod
    def gepa_package_missing() -> EscalationEvent:
        return EscalationEvent(
            id=str(uuid4())[:8],
            severity="critical",
            title="GEPA Package Not Installed",
            description="The GEPA optimization package is not installed. The fallback optimizer will be used, but full GEPA features are unavailable.",
            agent_type=None,
            error="ImportError: No module named 'gepa'",
            required_actions=[
                """1. Install the GEPA package:
   
   $ pip install gepa dspy-ai
   
   Or with the project extras:
   $ pip install -e ".[gepa]" """,
                
                """2. Verify installation:
   $ python -c "import gepa; import dspy; print('OK')"
   
   Should print "OK" without errors.""",
                
                """3. Restart the automation loop:
   $ python -m src.core.gepa.auto_loop start""",
            ],
        )
    
    @staticmethod
    def system_recovered(issue: str, resolution: str) -> EscalationEvent:
        return EscalationEvent(
            id=str(uuid4())[:8],
            severity="info",
            title="System Recovered - Automation Resumed",
            description=f"The previous issue has been resolved. GEPA automation is back to normal operation.",
            agent_type=None,
            error=None,
            required_actions=[
                f"""Previous issue: {issue}
   
Resolution: {resolution}

No action required. This is a confirmation that the system has recovered.
Normal optimization operations have resumed automatically.""",
            ],
            context={
                "issue": issue,
                "resolution": resolution,
            },
        )
