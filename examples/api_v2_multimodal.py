#!/usr/bin/env python3
"""
LlamaScale API v2 Example - Multimodal Capabilities

This example demonstrates how to use the LlamaScale API v2 with multimodal inputs.
The example shows how to:
1. Process images alongside text inputs
2. Use multimodal agents to analyze images
3. Stream multimodal results
"""

import asyncio
import json
import httpx
import time
import os
import argparse
from typing import Dict, List, Any, Optional

# API Settings
API_URL = "http://localhost:8000"
MODEL = "llama3-70b-instruct-multimodal-q4"  # Replace with your multimodal model name
API_KEY = None  # Set this if you have authentication enabled

async def call_api(method: str, endpoint: str, 
                  json_data: Optional[Dict[str, Any]] = None, 
                  api_key: Optional[str] = None) -> Dict[str, Any]:
    """Call the LlamaScale API"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    async with httpx.AsyncClient() as client:
        if method.lower() == "get":
            response = await client.get(
                f"{API_URL}{endpoint}",
                headers=headers,
                timeout=60.0
            )
        else:  # POST
            response = await client.post(
                f"{API_URL}{endpoint}",
                json=json_data,
                headers=headers,
                timeout=60.0
            )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            raise Exception(f"API call failed with status {response.status_code}")
            
        return response.json()

def print_colorful(title: str, content: str, color: str = "blue"):
    """Print colorful output if rich is available"""
    try:
        from rich import print as rprint
        from rich.panel import Panel
        
        colors = {
            "blue": "cyan",
            "green": "green",
            "red": "red",
            "yellow": "yellow",
            "purple": "magenta"
        }
        
        rprint(Panel(content, title=title, border_style=colors.get(color, "blue")))
    except ImportError:
        print(f"\n=== {title} ===")
        print(content)
        print("=" * (len(title) + 8))

async def main():
    """Run the example"""
    parser = argparse.ArgumentParser(description="LlamaScale Multimodal Example")
    parser.add_argument("--image", type=str, help="Path to an image to analyze")
    parser.add_argument("--prompt", type=str, default="What's in this image?", help="Text prompt to send with the image")
    parser.add_argument("--agent", action="store_true", help="Use an agent for more complex reasoning")
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    
    args = parser.parse_args()
    
    if not args.image:
        print("Please specify an image with --image")
        return
    
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return
    
    print_colorful("🦙 LlamaScale Multimodal Example", 
                  "Demonstrating multimodal capabilities with image and text processing", 
                  "purple")
    
    # Check if the API is running
    try:
        health = await call_api("GET", "/v2/health")
        print_colorful("API Status", f"Status: {health['status']}\nVersion: {health['version']}", "green")
        
        # Check if any models are loaded
        if health.get("models_loaded"):
            print_colorful("Models Loaded", ", ".join(health["models_loaded"]), "blue")
    except Exception as e:
        print(f"Failed to connect to API: {e}")
        print(f"Make sure the API server is running at {API_URL}")
        print("You can start it with: llamascale_api --v2")
        return
    
    # Prepare the image input
    image_input = {
        "url": args.image,
        "type": "url"  # URL here can be a local file path
    }
    
    if args.agent:
        # Example 1: Use an agent for complex multimodal reasoning
        print_colorful("Creating Multimodal Agent", "Setting up a multimodal-capable agent", "blue")
        
        # Create a multimodal agent
        agent_request = {
            "name": "MultimodalAssistant",
            "model": MODEL,
            "reasoning_strategy": "multimodal_reasoning",
            "tools": [],
            "system_prompt": "You are a helpful assistant that analyzes images and text. Provide detailed observations about images.",
            "multimodal_capable": True
        }
        
        try:
            agent_response = await call_api("POST", "/v2/agents/create", agent_request, API_KEY)
            agent_id = agent_response.get("agent_id")
            
            print_colorful("Agent Created", f"ID: {agent_id}", "green")
            
            # Prepare the agent request with image
            run_request = {
                "prompt": args.prompt,
                "max_steps": 5,
                "images": [image_input]
            }
            
            print_colorful("Processing Image", f"Image: {args.image}\nPrompt: {args.prompt}", "blue")
            
            if args.stream:
                # Stream the agent run
                print_colorful("Streaming Agent Response", "Processing image and generating response...", "yellow")
                
                headers = {"Content-Type": "application/json"}
                if API_KEY:
                    headers["Authorization"] = f"Bearer {API_KEY}"
                
                url = f"{API_URL}/v2/agents/{agent_id}/run/stream"
                
                async with httpx.AsyncClient() as client:
                    async with client.stream("POST", url, json=run_request, headers=headers) as response:
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
                                    print(f"\n🤔 Thinking: {event.get('content', '')}")
                                elif event_type == "tool_use":
                                    print(f"\n🔧 Using Tool: {event.get('content', '')}")
                                elif event_type == "tool_result":
                                    print(f"\n📊 Tool Result: {event.get('result', {})}")
                                elif event_type == "final_answer":
                                    print(f"\n✅ Final Answer: {event.get('content', '')}")
                                elif event_type == "error":
                                    print(f"\n❌ Error: {event.get('error', 'Unknown error')}")
                                else:
                                    print(f"\nEvent ({event_type}): {json.dumps(event, indent=2)}")
                            except json.JSONDecodeError:
                                print(f"Error parsing event: {line}")
            else:
                # Run the agent
                run_response = await call_api("POST", f"/v2/agents/{agent_id}/run", run_request, API_KEY)
                
                print_colorful("Agent Analysis", run_response["final_answer"], "green")
                
                # Print token usage
                if "usage" in run_response:
                    usage = run_response["usage"]
                    print_colorful("Token Usage", 
                                  f"Prompt: {usage['prompt_tokens']}\n"
                                  f"Completion: {usage['completion_tokens']}\n"
                                  f"Total: {usage['total_tokens']}", 
                                  "yellow")
        
        except Exception as e:
            print_colorful("Error", f"Agent run failed: {e}", "red")
    
    else:
        # Example 2: Direct multimodal generation
        try:
            print_colorful("Direct Multimodal Generation", f"Processing image with prompt: {args.prompt}", "blue")
            
            # Prepare the generation request
            generation_request = {
                "model": MODEL,
                "prompt": args.prompt,
                "max_tokens": 300,
                "temperature": 0.7,
                "images": [image_input]
            }
            
            # Make the API request
            response = await call_api("POST", "/v2/completions", generation_request, API_KEY)
            
            print_colorful("Multimodal Analysis", response["choices"][0]["text"], "green")
            
            # Print token usage
            if "usage" in response:
                usage = response["usage"]
                print_colorful("Token Usage", 
                              f"Prompt: {usage['prompt_tokens']}\n"
                              f"Completion: {usage['completion_tokens']}\n"
                              f"Total: {usage['total_tokens']}", 
                              "yellow")
                
        except Exception as e:
            print_colorful("Error", f"Multimodal generation failed: {e}", "red")

if __name__ == "__main__":
    asyncio.run(main()) 