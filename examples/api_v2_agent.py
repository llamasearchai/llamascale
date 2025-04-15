#!/usr/bin/env python3
"""
LlamaScale API v2 Example - Agent Capabilities

This example demonstrates how to use the LlamaScale API v2 with agent capabilities.
The example shows how to:
1. Create an agent with specific tools
2. Run the agent on a user query
3. Stream the agent's reasoning steps and actions
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import httpx

# API Settings
API_URL = "http://localhost:8000"
MODEL = "llama3-70b-instruct-q4"  # Replace with your model name
API_KEY = None  # Set this if you have authentication enabled


async def call_api(
    method: str,
    endpoint: str,
    json_data: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the LlamaScale API"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient() as client:
        if method.lower() == "get":
            response = await client.get(
                f"{API_URL}{endpoint}", headers=headers, timeout=60.0
            )
        else:  # POST
            response = await client.post(
                f"{API_URL}{endpoint}", json=json_data, headers=headers, timeout=60.0
            )

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            raise Exception(f"API call failed with status {response.status_code}")

        return response.json()


async def stream_agent_run(agent_id: str, prompt: str, max_steps: int = 10):
    """Stream the execution of an agent"""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    url = f"{API_URL}/v2/agents/{agent_id}/run/stream"
    data = {"prompt": prompt, "max_steps": max_steps}

    print(f"Streaming agent run for query: {prompt}")
    print("----------------------------------------")

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=data, headers=headers) as response:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                if line.startswith("data:"):
                    line = line[5:].strip()

                try:
                    event = json.loads(line)

                    # Format and print the event
                    event_type = event.get("type", "unknown")

                    if event_type == "thinking":
                        print("\n🤔 Thinking: " + event.get("content", ""))
                    elif event_type == "tool_use":
                        print("\n🔧 Using Tool: " + event.get("content", ""))
                        if "tool_calls" in event:
                            for tool_call in event.get("tool_calls", []):
                                func_name = tool_call["function"]["name"]
                                args = tool_call["function"]["arguments"]
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except:
                                        pass
                                print(f"  - {func_name}({json.dumps(args, indent=2)})")
                    elif event_type == "tool_result":
                        print("\n📊 Tool Result: ")
                        result = event.get("result", {})
                        print(f"  {json.dumps(result, indent=2)}")
                    elif event_type == "final_answer":
                        print("\n✅ Final Answer: " + event.get("content", ""))
                    elif event_type == "error":
                        print("\n❌ Error: " + event.get("error", "Unknown error"))
                    else:
                        print(f"\nEvent ({event_type}): {json.dumps(event, indent=2)}")
                except json.JSONDecodeError:
                    print(f"Error parsing event: {line}")

    print("\nAgent run completed")


async def main():
    """Run the example"""
    print("LlamaScale API v2 Agent Example")
    print("-------------------------------")

    # Check if the API is running
    try:
        health = await call_api("GET", "/v2/health")
        print(f"API status: {health['status']}")
        print(f"API version: {health['version']}")

        # Check if any models are loaded
        if health.get("models_loaded"):
            print(f"Models loaded: {', '.join(health['models_loaded'])}")
    except Exception as e:
        print(f"Failed to connect to API: {e}")
        print(f"Make sure the API server is running at {API_URL}")
        print("You can start it with: llamascale_api --v2")
        return

    # Get available tools
    print("\nRetrieving available tools...")
    tools_response = await call_api("GET", "/v2/tools", api_key=API_KEY)
    available_tools = tools_response.get("tools", [])

    if not available_tools:
        print("No tools available. Make sure the API server is configured correctly.")
        return

    print(f"Found {len(available_tools)} available tools:")
    for tool in available_tools:
        print(f"  - {tool['function']['name']}: {tool['function']['description']}")

    # Create a weather assistant agent
    print("\nCreating a weather assistant agent...")

    agent_request = {
        "name": "WeatherAssistant",
        "model": MODEL,
        "reasoning_strategy": "react",
        "tools": ["get_current_weather", "calculate"],
        "system_prompt": "You are a helpful assistant that provides accurate weather information and calculations.",
    }

    agent_response = await call_api("POST", "/v2/agents/create", agent_request, API_KEY)
    agent_id = agent_response.get("agent_id")

    print(f"Created agent with ID: {agent_id}")
    print(f"Agent details: {json.dumps(agent_response, indent=2)}")

    # Example 1: Basic weather query
    user_query = (
        "What's the weather like in San Francisco and is it warmer than New York?"
    )

    print("\nExample 1: Running agent with weather comparison query")
    print("------------------------------------------------------")
    print(f"Query: {user_query}")

    # Run the agent with streaming
    await stream_agent_run(agent_id, user_query)

    # Example 2: Multi-tool query with calculations
    user_query = "What's the average temperature between Tokyo and Paris, and what's the square root of that value?"

    print("\nExample 2: Running agent with multi-tool query")
    print("----------------------------------------------")
    print(f"Query: {user_query}")

    # Run the agent with streaming
    await stream_agent_run(agent_id, user_query)

    # Non-streaming example
    print("\nExample 3: Non-streaming agent run")
    print("----------------------------------")
    user_query = "Is it currently raining in London?"
    print(f"Query: {user_query}")

    agent_run_response = await call_api(
        "POST",
        f"/v2/agents/{agent_id}/run",
        {"prompt": user_query, "max_steps": 5},
        API_KEY,
    )

    print(f"Final answer: {agent_run_response['final_answer']}")
    print("\nReasoning steps:")

    for i, step in enumerate(agent_run_response["steps"]):
        step_type = step["type"]
        if step_type == "thinking":
            print(f"\nStep {i+1}: Thinking")
            print(step["content"])
        elif step_type == "tool_use":
            print(f"\nStep {i+1}: Using Tool")
            print(step["content"])
            for tool_call in step.get("tool_calls", []):
                func_name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]
                print(f"- {func_name}({args})")
        elif step_type == "final_answer":
            print(f"\nStep {i+1}: Final Answer")
            print(step["content"])


if __name__ == "__main__":
    asyncio.run(main())
