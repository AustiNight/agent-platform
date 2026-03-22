"""
Agent implementations.

Each subdirectory contains a specialist agent implementation.
"""

from src.agents.browser.agent import BrowserAgent

# Registry of available agents
AGENT_REGISTRY = {
    "browser": BrowserAgent,
    # Add more agents here as they're implemented
    # "research": ResearchAgent,
    # "writer": WriterAgent,
    # "fixer": FixerAgent,
}


def get_agent_class(agent_type: str) -> type:
    """Get the agent class for a given agent type."""
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return AGENT_REGISTRY[agent_type]


__all__ = ["AGENT_REGISTRY", "get_agent_class", "BrowserAgent"]
