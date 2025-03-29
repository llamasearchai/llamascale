#!/usr/bin/env python3
"""
LlamaScale API v2 Example - Function Calling

This example demonstrates how to use the LlamaScale API v2 with function calling capabilities.
The example shows how to:
1. Define functions that the model can call
2. Make a request to generate text with function calling
3. Execute the function call and continue the conversation
"""

import asyncio
import json
import httpx
import time
from typing import Dict, List, Any, Optional

# API Settings
API_URL = "http://localhost:8000"
MODEL = "mixtral-8x7b-instruct-q4"  # Replace with your model name
API_KEY = None  # Set this if you have authentication enabled

async def call_api(endpoint: str, json_data: Dict[str, Any], api_key: Optional[str] = None) -> Dict[str, Any]:
    """Call the LlamaScale API"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    async with httpx.AsyncClient() as client:
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

async def execute_function(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the function call"""
    function_name = tool_call["function"]["name"]
    
    try:
        arguments_str = tool_call["function"]["arguments"]
        arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
    except json.JSONDecodeError:
        return {
            "error": f"Invalid JSON arguments: {tool_call['function']['arguments']}"
        }
    
    # Execute the tool call via the API
    return await call_api(
        "/v2/tools/execute",
        tool_call,
        API_KEY
    )

async def main():
    """Run the example"""
    print("LlamaScale API v2 Function Calling Example")
    print("------------------------------------------")
    
    # Check if the API is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/v2/health")
            if response.status_code != 200:
                print(f"API server is not running at {API_URL}")
                return
            
            health = response.json()
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
    
    # Define the weather function
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string", 
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use"
                    }
                },
                "required": ["location"]
            }
        }
    }
    
    # Define the calculator function
    calculator_tool = {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a calculation using a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
    
    # Example 1: Weather query
    print("\nExample 1: Weather Query")
    print("------------------------")
    
    user_query = "What's the weather like in San Francisco?"
    print(f"User query: {user_query}")
    
    # Make the function calling API request
    response = await call_api(
        "/v2/completions",
        {
            "model": MODEL,
            "prompt": user_query,
            "max_tokens": 300,
            "temperature": 0.7,
            "tools": [weather_tool]
        },
        API_KEY
    )
    
    # Check if the model made a function call
    if "tool_calls" in response and response["tool_calls"]:
        print("\nModel made a function call:")
        tool_call = response["tool_calls"][0]
        
        # Print the model's initial response
        print(f"Model: {response['choices'][0]['text']}")
        
        # Print the function call details
        function_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        print(f"Function: {function_name}")
        print(f"Arguments: {arguments}")
        
        # Execute the function call
        print("\nExecuting function call...")
        function_result = await execute_function(tool_call)
        
        print(f"Function result: {json.dumps(function_result, indent=2)}")
        
        # Continue the conversation with the function result
        print("\nContinuing conversation with function result...")
        function_name = function_result["function"]["name"]
        result = function_result["function"]["result"]
        
        follow_up_prompt = f"""
User query: {user_query}

Function call: {function_name}({arguments})
Function result: {json.dumps(result, indent=2)}

Please provide a helpful response to the user based on this information.
"""
        
        follow_up_response = await call_api(
            "/v2/completions",
            {
                "model": MODEL,
                "prompt": follow_up_prompt,
                "max_tokens": 300,
                "temperature": 0.7
            },
            API_KEY
        )
        
        print(f"Final response: {follow_up_response['choices'][0]['text']}")
    else:
        print(f"Model response: {response['choices'][0]['text']}")
    
    # Example 2: Calculator
    print("\nExample 2: Calculator")
    print("--------------------")
    
    user_query = "What is the square root of 169 plus 15?"
    print(f"User query: {user_query}")
    
    # Make the function calling API request
    response = await call_api(
        "/v2/completions",
        {
            "model": MODEL,
            "prompt": user_query,
            "max_tokens": 300,
            "temperature": 0.7,
            "tools": [calculator_tool]
        },
        API_KEY
    )
    
    # Check if the model made a function call
    if "tool_calls" in response and response["tool_calls"]:
        print("\nModel made a function call:")
        tool_call = response["tool_calls"][0]
        
        # Print the model's initial response
        print(f"Model: {response['choices'][0]['text']}")
        
        # Print the function call details
        function_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        print(f"Function: {function_name}")
        print(f"Arguments: {arguments}")
        
        # Execute the function call
        print("\nExecuting function call...")
        function_result = await execute_function(tool_call)
        
        print(f"Function result: {json.dumps(function_result, indent=2)}")
        
        # Continue the conversation with the function result
        print("\nContinuing conversation with function result...")
        function_name = function_result["function"]["name"]
        result = function_result["function"]["result"]
        
        follow_up_prompt = f"""
User query: {user_query}

Function call: {function_name}({arguments})
Function result: {json.dumps(result, indent=2)}

Please provide a helpful response to the user based on this information.
"""
        
        follow_up_response = await call_api(
            "/v2/completions",
            {
                "model": MODEL,
                "prompt": follow_up_prompt,
                "max_tokens": 300,
                "temperature": 0.7
            },
            API_KEY
        )
        
        print(f"Final response: {follow_up_response['choices'][0]['text']}")
    else:
        print(f"Model response: {response['choices'][0]['text']}")

if __name__ == "__main__":
    asyncio.run(main()) 