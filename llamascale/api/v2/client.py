#!/usr/bin/env python3
"""
LlamaScale API v2 Client - Python client for LlamaScale API v2

This module provides a Python client for interacting with the LlamaScale API v2,
making it easy to generate text with function calling and use agents.
"""

import os
import json
import time
import asyncio
import httpx
from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator

class LlamaScaleClient:
    """Python client for the LlamaScale API v2"""
    
    def __init__(self, 
                base_url: str = "http://localhost:8000", 
                api_key: Optional[str] = None,
                timeout: float = 60.0):
        """Initialize the LlamaScale API client
        
        Args:
            base_url: The base URL of the LlamaScale API
            api_key: API key for authentication (optional)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def _make_request(self, 
                           method: str, 
                           endpoint: str, 
                           data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a request to the API
        
        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint
            data: Request data (for POST)
            
        Returns:
            Response data
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        async with httpx.AsyncClient() as client:
            if method.upper() == "GET":
                response = await client.get(url, headers=headers, timeout=self.timeout)
            else:  # POST
                response = await client.post(url, json=data, headers=headers, timeout=self.timeout)
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
            return response.json()
    
    async def generate(self, 
                      model: str, 
                      prompt: str, 
                      max_tokens: int = 256, 
                      temperature: float = 0.7, 
                      top_p: float = 0.95, 
                      tools: Optional[List[Dict[str, Any]]] = None, 
                      tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                      seed: Optional[int] = None,
                      stream: bool = False) -> Dict[str, Any]:
        """Generate text from a model
        
        Args:
            model: Model ID to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            tools: List of tools available to the model
            tool_choice: Tool choice specification
            seed: Random seed for generation
            stream: Whether to stream the response
            
        Returns:
            Generation response
        """
        data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": stream
        }
        
        if tools:
            data["tools"] = tools
        
        if tool_choice:
            data["tool_choice"] = tool_choice
        
        endpoint = "/v2/completions"
        if stream:
            endpoint = "/v2/completions/stream"
            raise ValueError("Streaming is not supported in the generate method. Use generate_stream instead.")
        
        return await self._make_request("POST", endpoint, data)
    
    async def generate_stream(self, 
                             model: str, 
                             prompt: str, 
                             max_tokens: int = 256, 
                             temperature: float = 0.7, 
                             top_p: float = 0.95, 
                             tools: Optional[List[Dict[str, Any]]] = None, 
                             tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                             seed: Optional[int] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream text generation from a model
        
        Args:
            model: Model ID to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            tools: List of tools available to the model
            tool_choice: Tool choice specification
            seed: Random seed for generation
            
        Yields:
            Generation chunks
        """
        data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": True
        }
        
        if tools:
            data["tools"] = tools
        
        if tool_choice:
            data["tool_choice"] = tool_choice
        
        url = f"{self.base_url}/v2/completions/stream"
        headers = self._get_headers()
        
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=data, headers=headers, timeout=self.timeout) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"API request failed with status {response.status_code}: {error_text.decode()}")
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    
    async def execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call
        
        Args:
            tool_call: Tool call to execute
            
        Returns:
            Result of the tool execution
        """
        endpoint = "/v2/tools/execute"
        return await self._make_request("POST", endpoint, tool_call)
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools
        
        Returns:
            List of available tools
        """
        response = await self._make_request("GET", "/v2/tools")
        return response.get("tools", [])
    
    async def create_agent(self, 
                          name: str, 
                          model: str, 
                          reasoning_strategy: str = "react", 
                          tools: Optional[List[str]] = None,
                          system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Create an agent
        
        Args:
            name: Agent name
            model: Model to use
            reasoning_strategy: Reasoning strategy (react, chain_of_thought, etc.)
            tools: Tool names to make available to the agent
            system_prompt: System prompt for the agent
            
        Returns:
            Agent details including agent_id
        """
        data = {
            "name": name,
            "model": model,
            "reasoning_strategy": reasoning_strategy
        }
        
        if tools:
            data["tools"] = tools
        
        if system_prompt:
            data["system_prompt"] = system_prompt
        
        return await self._make_request("POST", "/v2/agents/create", data)
    
    async def run_agent(self, 
                       agent_id: str, 
                       prompt: str, 
                       max_steps: int = 10,
                       tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run an agent with a prompt
        
        Args:
            agent_id: Agent ID to run
            prompt: User prompt for the agent
            max_steps: Maximum number of steps the agent can take
            tools: Optional tools to override agent's default tools
            
        Returns:
            Agent run results including final answer and steps
        """
        data = {
            "prompt": prompt,
            "max_steps": max_steps
        }
        
        if tools:
            data["tools"] = tools
        
        return await self._make_request("POST", f"/v2/agents/{agent_id}/run", data)
    
    async def stream_agent_run(self, 
                              agent_id: str, 
                              prompt: str, 
                              max_steps: int = 10,
                              tools: Optional[List[Dict[str, Any]]] = None,
                              callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream an agent run
        
        Args:
            agent_id: Agent ID to run
            prompt: User prompt for the agent
            max_steps: Maximum number of steps the agent can take
            tools: Optional tools to override agent's default tools
            callback: Optional callback for each event
            
        Yields:
            Agent run events
        """
        data = {
            "prompt": prompt,
            "max_steps": max_steps
        }
        
        if tools:
            data["tools"] = tools
        
        url = f"{self.base_url}/v2/agents/{agent_id}/run/stream"
        headers = self._get_headers()
        
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=data, headers=headers, timeout=self.timeout) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"API request failed with status {response.status_code}: {error_text.decode()}")
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    
                    try:
                        event = json.loads(line)
                        
                        if callback:
                            callback(event)
                        
                        yield event
                    except json.JSONDecodeError:
                        continue
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models
        
        Returns:
            List of available models
        """
        response = await self._make_request("GET", "/v2/models")
        return response.get("models", [])
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health
        
        Returns:
            Health status information
        """
        return await self._make_request("GET", "/v2/health")

# Synchronous client wrapper
class LlamaScaleClientSync:
    """Synchronous wrapper for the LlamaScale API v2 client"""
    
    def __init__(self, 
                base_url: str = "http://localhost:8000", 
                api_key: Optional[str] = None,
                timeout: float = 60.0):
        """Initialize the synchronous LlamaScale API client"""
        self.async_client = LlamaScaleClient(base_url, api_key, timeout)
        self.loop = asyncio.get_event_loop()
    
    def generate(self, **kwargs) -> Dict[str, Any]:
        """Generate text (synchronous wrapper)"""
        return self.loop.run_until_complete(self.async_client.generate(**kwargs))
    
    def execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call (synchronous wrapper)"""
        return self.loop.run_until_complete(self.async_client.execute_tool_call(tool_call))
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools (synchronous wrapper)"""
        return self.loop.run_until_complete(self.async_client.list_tools())
    
    def create_agent(self, **kwargs) -> Dict[str, Any]:
        """Create an agent (synchronous wrapper)"""
        return self.loop.run_until_complete(self.async_client.create_agent(**kwargs))
    
    def run_agent(self, **kwargs) -> Dict[str, Any]:
        """Run an agent (synchronous wrapper)"""
        return self.loop.run_until_complete(self.async_client.run_agent(**kwargs))
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models (synchronous wrapper)"""
        return self.loop.run_until_complete(self.async_client.list_models())
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health (synchronous wrapper)"""
        return self.loop.run_until_complete(self.async_client.health_check()) 