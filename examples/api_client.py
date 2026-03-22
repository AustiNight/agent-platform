#!/usr/bin/env python3
"""
Example: Use the Agent Platform API.

This script demonstrates how to interact with the API
from a client application.
"""

import httpx
import time
import sys


API_BASE = "http://localhost:8000"


def create_task(instructions: str, agent_type: str = "browser", wait: bool = False) -> dict:
    """Create a new task via the API."""
    response = httpx.post(
        f"{API_BASE}/api/v1/tasks",
        json={
            "agent_type": agent_type,
            "instructions": instructions,
            "wait": wait,
            "config": {
                "headless": False,
                "timeout_seconds": 120,
            }
        },
        timeout=300 if wait else 30,
    )
    response.raise_for_status()
    return response.json()


def get_task(task_id: str) -> dict:
    """Get task status."""
    response = httpx.get(f"{API_BASE}/api/v1/tasks/{task_id}")
    response.raise_for_status()
    return response.json()


def get_task_full(task_id: str) -> dict:
    """Get full task details including messages."""
    response = httpx.get(f"{API_BASE}/api/v1/tasks/{task_id}/full")
    response.raise_for_status()
    return response.json()


def list_agents() -> dict:
    """List available agents."""
    response = httpx.get(f"{API_BASE}/api/v1/agents")
    response.raise_for_status()
    return response.json()


def poll_until_complete(task_id: str, interval: float = 2.0, timeout: float = 300) -> dict:
    """Poll task status until completion or timeout."""
    start = time.time()
    
    while time.time() - start < timeout:
        task = get_task(task_id)
        status = task["status"]
        
        print(f"  Status: {status}", end="")
        if task.get("progress"):
            print(f" - {task['progress']}", end="")
        print()
        
        if status in ["completed", "failed", "cancelled"]:
            return task
        
        time.sleep(interval)
    
    raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")


def main():
    """Run example API interactions."""
    
    # Check API is running
    try:
        response = httpx.get(f"{API_BASE}/health")
        response.raise_for_status()
    except httpx.ConnectError:
        print(f"Error: Cannot connect to API at {API_BASE}")
        print("Make sure the server is running:")
        print("  uvicorn src.gateway.main:app --reload")
        sys.exit(1)
    
    print("Agent Platform API Example")
    print("=" * 50)
    
    # List available agents
    print("\n1. Available Agents:")
    agents = list_agents()
    for agent in agents["agents"]:
        print(f"   - {agent['type']}: {agent['description'][:60]}...")
    
    # Create a simple task
    print("\n2. Creating browser task...")
    instructions = """
    Navigate to https://news.ycombinator.com
    Extract the titles of the top 3 stories on the front page
    Complete the task with the extracted titles
    """
    
    task = create_task(instructions, wait=False)
    task_id = task["task_id"]
    print(f"   Task ID: {task_id}")
    print(f"   Status: {task['status']}")
    
    # Poll for completion
    print("\n3. Waiting for completion...")
    result = poll_until_complete(task_id)
    
    # Show results
    print("\n4. Results:")
    print(f"   Status: {result['status']}")
    
    if result.get("error"):
        print(f"   Error: {result['error']}")
    
    if result.get("result"):
        print(f"   Result: {result['result']}")
    
    # Get full details
    print("\n5. Full task details:")
    full = get_task_full(task_id)
    print(f"   LLM Calls: {full.get('llm_calls', 'N/A')}")
    print(f"   Tokens: {full.get('total_tokens', 'N/A')}")
    print(f"   Messages: {len(full.get('messages', []))}")
    print(f"   Artifacts: {len(full.get('artifacts', []))}")


if __name__ == "__main__":
    main()
