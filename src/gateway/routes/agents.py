"""
Agent discovery and information endpoints.
"""

from fastapi import APIRouter

from src.agents import AGENT_REGISTRY
from src.core.schemas import AgentType

router = APIRouter()


@router.get("")
async def list_agents() -> dict:
    """List all available agents and their capabilities."""
    agents = []
    
    for agent_type, agent_class in AGENT_REGISTRY.items():
        # Instantiate to get properties (they're not class-level)
        agent = agent_class()
        
        agents.append({
            "type": agent_type,
            "description": agent.description,
            "capabilities": [cap.value for cap in agent.capabilities],
            "tools": [
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                }
                for tool in agent.tools
            ],
        })
    
    return {
        "agents": agents,
        "count": len(agents),
    }


@router.get("/{agent_type}")
async def get_agent_info(agent_type: AgentType) -> dict:
    """Get detailed information about a specific agent."""
    if agent_type.value not in AGENT_REGISTRY:
        return {"error": f"Unknown agent type: {agent_type}"}
    
    agent_class = AGENT_REGISTRY[agent_type.value]
    agent = agent_class()
    
    return {
        "type": agent_type.value,
        "description": agent.description,
        "capabilities": [cap.value for cap in agent.capabilities],
        "system_prompt": agent.system_prompt,
        "tools": agent.tools,
    }
