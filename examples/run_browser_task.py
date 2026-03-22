#!/usr/bin/env python3
"""
Example: Run a browser automation task.

This script demonstrates how to use the browser agent directly,
without going through the API.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.browser.agent import BrowserAgent
from src.core.base_agent import AgentContext
from src.core.llm_client import LLMClient
from src.core.schemas import TaskConfig


async def main():
    """Run a simple browser task."""
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    
    # Create LLM client
    llm_client = LLMClient(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
    )
    
    # Define the task
    instructions = """
    Go to https://example.com and:
    1. Get the page state to see what's on the page
    2. Extract the main heading text
    3. Take a screenshot
    4. Complete the task with the extracted heading
    """
    
    # Create execution context
    context = AgentContext(
        task_id="example-001",
        instructions=instructions,
        config=TaskConfig(
            timeout_seconds=60,
            max_llm_calls=10,
        ),
        llm_client=llm_client,
    )
    
    # Create and run agent
    agent = BrowserAgent()
    
    print("Starting browser agent...")
    print(f"Instructions: {instructions.strip()}")
    print("-" * 50)
    
    response = await agent.run(context)
    
    # Print results
    print("-" * 50)
    print(f"Status: {response.status.value}")
    print(f"LLM Calls: {response.llm_calls}")
    print(f"Total Tokens: {response.total_tokens}")
    
    if response.error:
        print(f"Error: {response.error}")
    
    if response.result:
        print(f"Result: {response.result}")
    
    if response.artifacts:
        print(f"Artifacts: {len(response.artifacts)}")
        for artifact in response.artifacts:
            print(f"  - {artifact.type}: {artifact.path or artifact.description}")
    
    print("\nConversation:")
    for msg in response.messages:
        print(f"  [{msg.role.value}] {msg.content[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
