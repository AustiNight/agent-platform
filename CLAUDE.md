# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

```bash
# Install dependencies
pip install -e ".[dev]"           # Development environment
pip install -e ".[gepa]"          # Include GEPA optimization
pip install -e ".[all]"           # All features

# Browser automation (required for browser agent)
playwright install chromium

# Start infrastructure (Redis + optional Postgres)
docker-compose up -d
docker-compose --profile postgres up -d   # Include Postgres
docker-compose --profile ollama up -d     # Include local LLMs
docker-compose --profile debug up -d      # Include Redis Commander

# Run database migrations
alembic upgrade head

# Start services
uvicorn src.gateway.main:app --reload --host 0.0.0.0 --port 8000
python -m src.worker.main
```

## Testing and Linting

```bash
pytest                            # All tests
pytest tests/path/to/test.py     # Single test file
pytest -k "test_name"            # Single test by name
pytest --cov=src                 # With coverage

ruff check .                     # Lint
ruff format .                    # Format
mypy src/                        # Type checking
```

## Architecture

Agent Platform is an async agent orchestration system. HTTP requests hit the **Gateway** (FastAPI), tasks are queued in **Redis** (RQ), and **Workers** execute agents against LLMs with optional browser automation.

```
Client → FastAPI Gateway → Redis Queue → Worker → Agent → LLM Provider
                                ↓                    ↓
                          PostgreSQL/SQLite      Playwright
```

**Key directories:**
- `src/core/` — Shared components: `BaseAgent`, `LLMClient`, Pydantic schemas, config, credentials, GEPA optimization
- `src/agents/browser/` — Browser agent (Playwright), browser tools, session manager
- `src/gateway/` — FastAPI app, routes (`/api/v1/tasks`, `/api/v1/agents`, `/health`), task executor service
- `src/worker/` — RQ worker process consuming the Redis queue
- `src/db/` — SQLAlchemy models (`Task`, `TaskMessage`), async session management
- `alembic/` — Database migration scripts

## Agent Execution Flow

1. `POST /api/v1/tasks` creates a `Task` record (status: pending) and enqueues it
2. Worker dequeues, loads the agent class, initializes Playwright session if needed
3. Agent loop: call LLM → if tool use requested → execute tool → append result → repeat
4. Each step is persisted as `TaskMessage` records; token counts and costs tracked
5. On completion, task status set to `completed`/`failed` with result/error

## Adding a New Agent

Inherit from `BaseAgent` (`src/core/base_agent.py`) and implement:
- `agent_type` → `AgentType` enum value
- `description` / `capabilities` → metadata
- `tools` → list of LLM-compatible tool definitions
- `execute_tool(tool_name, tool_input)` → dispatch tool calls

Register the agent in `src/core/schemas.py` (`AgentType` enum) and the gateway's agent loader.

## GEPA Prompt Optimization

GEPA (Genetic-Pareto optimization) automatically evolves agent system prompts based on execution traces. It's opt-in and lives in `src/core/gepa/`. To use:

- **Training:** run `examples/gepa_optimization.py` with labeled traces
- **Inference-time search:** `InferenceTimeSearch` tries candidate prompts at runtime
- **Autonomous loop:** `src/core/gepa/auto_loop.py` (deployable as `deploy/gepa-automation.service`)

Agents load optimized prompts automatically via `GEPAPromptLoader` middleware if GEPA is enabled.

## Key Config (`.env`)

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | LLM providers |
| `DATABASE_URL` | Postgres (`postgresql+asyncpg://...`) or SQLite (`sqlite+aiosqlite:///...`) |
| `REDIS_URL` | Redis connection (default: `redis://localhost:6379`) |
| `DEFAULT_LLM_MODEL` | Model string passed to LiteLLM (e.g. `claude-opus-4-6`) |
| `BROWSER_HEADLESS` | `true` for headless Playwright |
| `SITE_<DOMAIN>_USERNAME/PASSWORD` | Per-domain credentials injected by `src/core/secrets.py` |

See `.env.example` for the full list (~56 variables).
