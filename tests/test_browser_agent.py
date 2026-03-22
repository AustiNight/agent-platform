"""
Example test for the browser agent.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.browser.agent import BrowserAgent
from src.core.base_agent import AgentContext
from src.core.schemas import TaskConfig


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = AsyncMock()
    client.chat = AsyncMock(return_value={
        "content": "Task completed successfully.",
        "stop_reason": "end_turn",
        "usage": {"total_tokens": 100},
    })
    return client


@pytest.fixture
def agent_context(mock_llm_client):
    """Create a test agent context."""
    return AgentContext(
        task_id="test-123",
        instructions="Go to example.com and extract the heading",
        config=TaskConfig(timeout_seconds=30),
        llm_client=mock_llm_client,
    )


class TestBrowserAgent:
    """Tests for the browser agent."""

    def test_agent_properties(self):
        """Test agent property definitions."""
        agent = BrowserAgent()
        
        assert agent.agent_type.value == "browser"
        assert len(agent.capabilities) > 0
        assert len(agent.tools) > 0
        assert "navigate" in [t["name"] for t in agent.tools]

    def test_tool_definitions(self):
        """Test that all tools have required fields."""
        agent = BrowserAgent()
        
        for tool in agent.tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, agent_context):
        """Test handling of unknown tool names."""
        agent = BrowserAgent()
        
        # Don't need to call setup for this test
        result = await agent.execute_tool(
            "nonexistent_tool",
            {},
            agent_context,
        )
        
        assert not result.success
        assert "Unknown tool" in result.error
