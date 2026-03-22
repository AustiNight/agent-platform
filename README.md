# Agent Platform

A modular, portable agent orchestration system with a standardized interface for building specialized AI agents.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Apps (UIs)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP/WebSocket
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Gateway API (FastAPI)                    │
│  • Auth, rate limiting, cost tracking                       │
│  • Task queue / async job dispatch                          │
│  • Unified request envelope                                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Browser  │    │ Research │    │  Fixer   │
    │  Agent   │    │  Agent   │    │  Agent   │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         └───────────────┴───────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ Anthropic│   │  OpenAI  │   │  Ollama  │
   │   API    │   │   API    │   │  (Local) │
   └──────────┘   └──────────┘   └──────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- An Anthropic API key

### Local Development Setup

```bash
# 1. Clone and enter directory
cd agent-platform

# 2. Copy environment template
cp .env.example .env

# 3. Edit .env and add your API keys
#    ANTHROPIC_API_KEY=sk-ant-...

# 4. Start services (Redis + Postgres)
docker-compose up -d

# 5. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 6. Install dependencies
pip install -e ".[dev]"

# 7. Install Playwright browsers
playwright install chromium

# 8. Run database migrations
alembic upgrade head

# 9. Start the API server (development mode)
uvicorn src.gateway.main:app --reload --host 0.0.0.0 --port 8000

# 10. In a separate terminal, start the task worker
python -m src.worker.main
```

### Running the Browser Agent

```bash
# Example: Submit a browser automation task
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "browser",
    "instructions": "Go to https://example.com and extract the main heading text",
    "config": {
      "headless": false,
      "timeout_seconds": 60
    }
  }'
```

## Project Structure

```
agent-platform/
├── src/
│   ├── core/                 # Common interface layer
│   │   ├── base_agent.py     # Abstract agent contract
│   │   ├── llm_client.py     # Multi-provider LLM abstraction
│   │   ├── schemas.py        # Shared Pydantic models
│   │   └── config.py         # Configuration management
│   │
│   ├── agents/               # Specialist implementations
│   │   ├── browser/          # Browser automation agent
│   │   │   ├── agent.py
│   │   │   ├── tools.py
│   │   │   └── page_parser.py
│   │   └── ...               # Future agents
│   │
│   ├── gateway/              # FastAPI application
│   │   ├── main.py
│   │   ├── routes/
│   │   └── middleware/
│   │
│   ├── worker/               # Background task processor
│   │   └── main.py
│   │
│   └── db/                   # Database models and migrations
│       ├── models.py
│       └── session.py
│
├── tests/
├── alembic/                  # Database migrations
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## Adding a New Agent

1. Create a new directory under `src/agents/`
2. Implement the `BaseAgent` interface
3. Register the agent in `src/agents/__init__.py`
4. The gateway will automatically route tasks to it

See `src/agents/browser/` for a complete example.

## GEPA Optimization

All agents support automatic prompt optimization via [GEPA](https://arxiv.org/abs/2507.19457) (Genetic-Pareto Reflective Optimization).

### How It Works

1. **Collect traces**: Every agent execution captures a detailed trace
2. **Provide feedback**: Custom metrics return scores + textual feedback
3. **Reflect & evolve**: GEPA uses LLM reflection to propose improved prompts
4. **Pareto selection**: Maintains diverse strategies that excel on different tasks

### Quick Start

```python
from src.agents.browser.agent import BrowserAgent
from src.core.gepa import GEPAConfig, browser_task_metric
from src.core.gepa.optimizer import AgentOptimizer

# Create agent and optimizer
agent = BrowserAgent()
optimizer = AgentOptimizer(
    agent=agent,
    metric=browser_task_metric,
    config=GEPAConfig(auto="medium"),
)

# Optimize with training data
result = await optimizer.optimize(
    trainset=[
        {"instructions": "Go to example.com and extract the heading", 
         "expected": {"success": True}},
        # ... more training tasks
    ]
)

print(result.summary())
# Agent now uses optimized prompts!
```

### Custom Feedback Metrics

The key to GEPA's power is rich textual feedback. Define metrics that explain *why* something failed:

```python
from src.core.gepa import ScoreWithFeedback

def my_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    if pred.get("success"):
        return ScoreWithFeedback(
            score=1.0,
            feedback="Task completed successfully."
        )
    else:
        error = pred.get("error", "")
        
        # Provide actionable feedback
        if "timeout" in error.lower():
            feedback = "Element not found in time. Try adding wait_for calls."
        elif "not found" in error.lower():
            feedback = "Selector invalid. Use get_page_state to inspect available elements."
        else:
            feedback = f"Unknown error: {error}"
        
        return ScoreWithFeedback(score=0.0, feedback=feedback)
```

### Inference-Time Search

For complex tasks, use GEPA's Pareto frontier at inference time:

```python
from src.core.gepa.optimizer import InferenceTimeSearch

searcher = InferenceTimeSearch(agent)
best_result, score = await searcher.search(
    task={"instructions": "Complex multi-step task..."},
    num_candidates=5,
)
```

### Installation

GEPA is optional. Install with:

```bash
pip install -e ".[gepa]"
```

## Fully Automated GEPA Loop

The platform includes a **fully autonomous** optimization system that:

1. ✅ **Automatically collects traces** from every agent execution
2. ✅ **Triggers optimization** when performance drops or data accumulates  
3. ✅ **Applies improved prompts** without human intervention
4. ✅ **Self-heals** from transient failures with exponential backoff
5. ✅ **Escalates to you** ONLY when truly stuck (with step-by-step instructions)

### Quick Setup

```bash
# 1. Configure notifications (edit .env)
GEPA_NOTIFY_EMAIL=your-email@example.com  # Required
GEPA_SLACK_WEBHOOK=https://hooks.slack.com/...  # Optional but recommended

# 2. Start the automation
python -m src.core.gepa.auto_loop start

# Or run as a systemd service (production)
sudo cp deploy/gepa-automation.service /etc/systemd/system/
sudo systemctl enable --now gepa-automation
```

### What Gets Automated

| Task | Automated? | Notes |
|------|------------|-------|
| Trace collection | ✅ Yes | Every task completion is captured |
| Performance monitoring | ✅ Yes | Baseline tracking, drift detection |
| Optimization triggers | ✅ Yes | Based on traces, time, or performance drop |
| Prompt evolution | ✅ Yes | GEPA runs automatically |
| Applying new prompts | ✅ Yes | Hot-reloaded without restart |
| Failure recovery | ✅ Yes | Exponential backoff, self-healing |
| API key rotation | ❌ No | Requires human (you'll get detailed instructions) |
| Budget approval | ⚠️ Configurable | Set `GEPA_AUTO_APPROVE_HEAVY=true` to automate |
| Anomaly review | ⚠️ Optional | Large score changes flagged for review |

### When You'll Be Notified

You'll receive a notification **only** when:

1. **API keys expire** - With exact steps to rotate them
2. **Database is down** - With commands to diagnose/fix
3. **Heavy optimization needs approval** - Cost estimate included
4. **Results look anomalous** - Unusual score changes
5. **Persistent failures** - After 3 automatic retries fail

Each notification includes:
- ✅ Exact copy-paste commands to fix the issue
- ✅ What the system already tried automatically
- ✅ How to return to automated mode after fixing

### Monitoring

```bash
# Check status
python -m src.core.gepa.auto_loop status

# Run diagnostics
python -m src.core.gepa.auto_loop diagnose

# View dashboard (optional)
python -m src.core.gepa.dashboard
# Open http://localhost:8082
```

### Notification Channels

Configure multiple channels for redundancy:

| Channel | Config Variable | Best For |
|---------|----------------|----------|
| Email | `GEPA_NOTIFY_EMAIL` | Detailed instructions |
| Slack | `GEPA_SLACK_WEBHOOK` | Team visibility |
| Pushover | `PUSHOVER_USER_KEY` | Mobile push (critical) |
| SMS | `TWILIO_*` | Critical alerts |
| Webhook | `GEPA_WEBHOOK_URL` | Custom integrations |
| File | Always enabled | Guaranteed delivery |

### Example Escalation

When something requires your attention, you'll receive:

```
============================================================
GEPA AUTOMATION ALERT - CRITICAL
============================================================

Title: API Key Expired: anthropic
Time: 2024-01-15T10:30:00Z
Event ID: a1b2c3d4

DESCRIPTION:
The anthropic API key has expired or is invalid. GEPA optimization 
for browser agents cannot proceed.

============================================================
REQUIRED ACTIONS (in order):
============================================================

STEP 1:
Generate a new API key from anthropic:
   - Go to https://console.anthropic.com/settings/keys
   - Click "Create new secret key"
   - Copy the key immediately (it won't be shown again)

STEP 2:
Update the environment variable:
   $ export ANTHROPIC_API_KEY="sk-your-new-key-here"
   # Or update .env file for permanent change

STEP 3:
Restart the GEPA automation service:
   $ systemctl restart gepa-automation

STEP 4:
Verify the fix:
   $ python -m src.core.gepa.auto_loop status
   # Should show "Status: RUNNING" and "API: OK"

============================================================
RETURNING TO AUTOMATED MODE:
============================================================

After completing the above steps, the system will automatically:
1. Detect that the issue is resolved (within 5 minutes)
2. Resume normal optimization operations
3. Send a confirmation notification
```

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | Required |
| `OPENAI_API_KEY` | OpenAI API key (optional) | - |
| `OLLAMA_BASE_URL` | Local Ollama server URL | `http://localhost:11434` |
| `DATABASE_URL` | PostgreSQL connection string | SQLite for dev |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `DEFAULT_LLM_PROVIDER` | Default LLM provider | `anthropic` |
| `DEFAULT_LLM_MODEL` | Default model name | `claude-sonnet-4-20250514` |

## License

MIT
