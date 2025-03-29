#!/usr/bin/env python3
"""
LlamaScale API v2 Server - FastAPI server with function calling and agent capabilities
"""

import os
import json
import time
import uuid
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sse_starlette.sse import EventSourceResponse

from llamascale.drivers.mlx.engine import MLXModelEngine, MLXConfig
from llamascale.orchestrator.routing.smart import InferenceRouter, RequestPriority
from llamascale.orchestrator.caching.hybrid import HybridCache
from llamascale.tools.cli.init import init_llamascale_dirs
from llamascale.tools.cli.config import ConfigManager, get_config, LlamaScaleConfig
from llamascale.agents.framework import Agent, Tool, ReasoningStrategy

from .schemas import (
    GenerationRequest, 
    GenerationResponse,
    AgentRequest,
    AgentResponse,
    ToolCall,
    ToolDefinition,
    ModelInfo,
    ModelsResponse,
    HealthStatus,
    UsageInfo,
    ImageInput
)
from .functions import registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llamascale.api.v2")

# Initialize directories
init_llamascale_dirs()

# Security scheme
security = HTTPBearer(auto_error=False)

# API lifespan setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize components on startup
    await startup_event()
    yield
    # Cleanup on shutdown
    await shutdown_event()

# Initialize FastAPI app
app = FastAPI(
    title="LlamaScale API v2",
    description="Advanced API for LlamaScale with function calling and agent capabilities",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
config = None
config_manager = None
router = None
cache = None
models = {}
agents = {}
api_keys = {}

async def startup_event():
    """Initialize server components on startup"""
    global config, config_manager, router, cache, api_keys
    
    # Load configuration using ConfigManager
    config_path = os.environ.get("LLAMASCALE_CONFIG")
    config_manager = ConfigManager(config_path)
    llamascale_config = config_manager.load()
    
    # Convert to dict for backward compatibility with existing code
    config = config_manager._config_to_dict(llamascale_config)
    
    # Initialize cache
    cache_config = llamascale_config.cache_config
    cache = HybridCache(config["cache_config"])
    
    # Initialize router (if mlx_models is configured)
    if llamascale_config.mlx_models:
        router = InferenceRouter(config)
        await router.start()
    
    # Load API keys if available
    api_keys_file = os.path.join(os.path.dirname(config_manager.config_path), "api_keys.json")
    if os.path.exists(api_keys_file):
        try:
            with open(api_keys_file, "r") as f:
                api_keys = json.load(f)
            logger.info(f"Loaded {len(api_keys)} API keys")
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
    
    # Set FastAPI app host and port from config
    logger.info(f"API v2 server will listen on {llamascale_config.api_host}:{llamascale_config.api_port}")
    logger.info("LlamaScale API v2 server initialized")

async def shutdown_event():
    """Cleanup on server shutdown"""
    if router:
        await router.shutdown()
    logger.info("LlamaScale API v2 server shutdown")

async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """Verify API key and return user ID if valid"""
    if not api_keys:  # If no API keys are configured, authentication is disabled
        return "anonymous"
        
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    if token not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_keys[token]

async def get_model_engine(model_name: str) -> MLXModelEngine:
    """Get or initialize model engine for the specified model"""
    global models, config, config_manager
    
    if model_name in models:
        return models[model_name]
    
    # Initialize model
    model_path = os.path.join(config["models_dir"], model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Find model-specific configuration if available
    model_config = None
    for model in config_manager.config.models:
        if model.name == model_name:
            model_config = model
            break
    
    # Use model-specific quantization if available
    mlx_config_dict = config.get("mlx_config", {}).copy()
    if model_config and model_config.quantization:
        mlx_config_dict["quantization"] = model_config.quantization
    
    mlx_config = MLXConfig(**mlx_config_dict)
    engine = MLXModelEngine(model_path, mlx_config)
    models[model_name] = engine
    logger.info(f"Initialized model engine for {model_name}")
    
    return engine

async def _process_image_inputs(images: List[ImageInput]) -> List[Dict[str, Any]]:
    """Process images for multimodal input
    
    Args:
        images: List of image inputs
        
    Returns:
        Processed image data
    """
    if not images:
        return []
    
    processed_images = []
    for img in images:
        if img.type == "url":
            # Check if it's a local file
            if os.path.exists(img.url):
                # Process local file
                try:
                    processed_images.append({
                        "url": img.url,
                        "type": "local_file"
                    })
                except Exception as e:
                    logger.error(f"Error processing local image: {e}")
            else:
                # Download remote URL
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(img.url, timeout=30.0)
                        if response.status_code == 200:
                            # Save to temp file
                            temp_dir = os.path.join(os.path.expanduser("~"), ".llamascale", "temp")
                            os.makedirs(temp_dir, exist_ok=True)
                            
                            filename = f"temp_image_{uuid.uuid4()}.png"
                            filepath = os.path.join(temp_dir, filename)
                            
                            with open(filepath, "wb") as f:
                                f.write(response.content)
                                
                            processed_images.append({
                                "url": filepath,
                                "type": "downloaded",
                                "original_url": img.url
                            })
                except Exception as e:
                    logger.error(f"Error downloading image: {e}")
    
    return processed_images

@app.post("/v2/completions", response_model=GenerationResponse)
async def generate_completion(
    request: GenerationRequest,
    user_id: Optional[str] = Depends(verify_api_key)
):
    """Generate a completion with optional function calling"""
    start_time = time.time()
    model = request.model
    
    # Check if model exists
    engine = await get_model_engine(model)
    
    # Process images if present for multimodal inputs
    images = []
    if request.images:
        images = await _process_image_inputs(request.images)
        
        # Check if model supports multimodal
        model_info = engine.get_model_info() if hasattr(engine, "get_model_info") else {}
        if not model_info.get("supports_multimodal", False):
            raise HTTPException(
                status_code=400,
                detail=f"Model {model} does not support multimodal inputs"
            )
    
    # Format prompt for tools if needed
    prompt = request.prompt
    if request.tools:
        prompt = _format_prompt_with_tools(prompt, request.tools)
        
    # Add image information to prompt if needed
    if images:
        image_descriptions = []
        for i, img in enumerate(images):
            image_descriptions.append(f"[Image {i+1}: {img.get('original_url', img['url'])}]")
            
        if image_descriptions:
            prompt = f"{prompt}\n\nThe following images are attached to this message:\n" + "\n".join(image_descriptions)
    
    # Generate the completion
    gen_id = f"gen_{uuid.uuid4()}"
    
    result = await engine.generate(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        seed=request.seed
    )
    
    # Extract the completion text
    completion_text = result.get("text", "")
    
    # Check for tool calls
    tool_calls = _parse_tool_calls_from_text(completion_text)
    
    # Remove the tool calls from the completion text if found
    if tool_calls:
        completion_text = _remove_tool_calls_from_text(completion_text)
    
    # Format response 
    response = {
        "id": gen_id,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": completion_text,
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }
        ],
        "usage": {
            "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": result.get("usage", {}).get("total_tokens", 0)
        },
        "created": int(time.time()),
    }
    
    # Add tool calls if present
    if tool_calls:
        response["tool_calls"] = tool_calls
    
    return response

@app.post("/v2/completions/stream")
async def stream_completion(
    request: GenerationRequest,
    user_id: Optional[str] = Depends(verify_api_key)
):
    """Stream text generation from a model"""
    if not request.stream:
        return await generate_completion(request, user_id)
    
    # Get model engine
    engine = await get_model_engine(request.model)
    gen_id = str(uuid.uuid4())
    
    # Check if the engine supports streaming natively
    if hasattr(engine, "generate_stream"):
        # Create streaming response generator
        async def stream_generator():
            async for chunk in engine.generate_stream(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                seed=request.seed
            ):
                # Add required fields for SSE
                chunk_data = {
                    "id": gen_id,
                    "model": request.model,
                    "created": int(time.time()),
                    "object": "completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "text": chunk.get("token", ""),
                            "delta": {"text": chunk.get("token", "")},
                            "finish_reason": "stop" if chunk.get("type") == "end" else None
                        }
                    ]
                }
                
                # If the chunk has tool calls, include them
                if "tool_calls" in chunk:
                    chunk_data["tool_calls"] = chunk["tool_calls"]
                
                # Yield as SSE event
                yield json.dumps(chunk_data)
                
        return EventSourceResponse(stream_generator())
    else:
        # Fall back to simulated streaming
        return await _simulate_streaming(request, engine, gen_id)

async def _simulate_streaming(request: GenerationRequest, engine: MLXModelEngine, gen_id: str):
    """Simulate streaming for engines that don't support native streaming"""
    async def stream_generator():
        # Generate the full response first
        result = await engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed
        )
        
        # Parse tool calls if applicable
        tool_calls = None
        if request.tools:
            tool_calls = _parse_tool_calls_from_text(result["text"])
            result["text"] = _remove_tool_calls_from_text(result["text"])
        
        # Split text into chunks to simulate streaming
        text = result["text"]
        chunk_size = max(1, len(text) // 10)  # Aim for ~10 chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Stream text chunks
        for i, chunk in enumerate(chunks):
            # Simulate generation delay
            await asyncio.sleep(0.1)
            
            # Create chunk data
            chunk_data = {
                "id": gen_id,
                "model": request.model,
                "created": int(time.time()),
                "object": "completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "text": chunk,
                        "delta": {"text": chunk},
                        "finish_reason": "stop" if i == len(chunks) - 1 else None
                    }
                ]
            }
            
            # Include tool calls in the final chunk if present
            if i == len(chunks) - 1 and tool_calls:
                chunk_data["tool_calls"] = tool_calls
            
            # Yield as SSE event
            yield json.dumps(chunk_data)
    
    return EventSourceResponse(stream_generator())

def _format_prompt_with_tools(prompt: str, tools: List[ToolDefinition]) -> str:
    """Format the prompt with function calling information"""
    # Create a description of available tools
    tools_desc = "You have access to the following tools:\n\n"
    
    for i, tool in enumerate(tools):
        func = tool.function
        tools_desc += f"{i+1}. {func.name}: {func.description}\n"
        tools_desc += f"   Parameters: {json.dumps(func.parameters, indent=2)}\n\n"
    
    # Add instructions for tool use
    tools_desc += """
When you need to use a tool, respond in the following format:
<tool>
{
  "name": "tool_name",
  "arguments": {
    "param1": "value1",
    "param2": "value2"
  }
}
</tool>

Then continue with your response after the tool call.
You can make multiple tool calls if needed.
"""
    
    # Combine the tool description with the original prompt
    return f"{tools_desc}\n\nUser Request: {prompt}\n\nResponse:"

def _parse_tool_calls_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse tool calls from generated text"""
    import re
    
    # Look for tool calls enclosed in <tool>...</tool> tags
    pattern = r"<tool>(.*?)</tool>"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if not matches:
        return None
    
    tool_calls = []
    for i, match in enumerate(matches):
        try:
            # Parse the JSON inside the tool tags
            tool_data = json.loads(match.strip())
            
            # Create a ToolCall object
            tool_call = {
                "id": f"call_{str(uuid.uuid4())[:8]}",
                "type": "function",
                "function": {
                    "name": tool_data.get("name", ""),
                    "arguments": json.dumps(tool_data.get("arguments", {}))
                }
            }
            
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool call: {match}")
    
    return tool_calls if tool_calls else None

def _remove_tool_calls_from_text(text: str) -> str:
    """Remove tool calls from generated text"""
    import re
    
    # Remove tool calls enclosed in <tool>...</tool> tags
    clean_text = re.sub(r"<tool>.*?</tool>", "", text, flags=re.DOTALL)
    
    # Clean up any extra whitespace
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    
    return clean_text.strip()

@app.post("/v2/tools/execute")
async def execute_tool(
    tool_call: ToolCall,
    user_id: Optional[str] = Depends(verify_api_key)
):
    """Execute a tool call and return the result"""
    try:
        # Use the function registry to execute the tool
        result = await registry.execute_tool_call(tool_call)
        
        # Format the response
        return {
            "id": f"result_{tool_call.id}",
            "type": "function_result",
            "function": {
                "name": tool_call.function.get("name"),
                "result": result
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing tool: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

@app.get("/v2/tools")
async def list_tools(
    user_id: Optional[str] = Depends(verify_api_key)
):
    """List available tools"""
    tools = registry.get_tool_definitions()
    return {"tools": [t.dict() for t in tools]}

@app.post("/v2/agents/create")
async def create_agent(
    request: Dict[str, Any],
    user_id: Optional[str] = Depends(verify_api_key)
):
    """Create a new agent"""
    # Required fields
    name = request.get("name")
    model_name = request.get("model")
    
    if not name or not model_name:
        raise HTTPException(status_code=400, detail="Name and model are required")
    
    # Optional fields
    reasoning_strategy = request.get("reasoning_strategy", "react")
    system_prompt = request.get("system_prompt")
    
    # Validate reasoning strategy
    try:
        strategy = ReasoningStrategy(reasoning_strategy)
    except ValueError:
        valid_strategies = [s.value for s in ReasoningStrategy]
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid reasoning strategy. Valid options: {', '.join(valid_strategies)}"
        )
    
    # Create agent tools from registry tools
    tools = []
    if "tools" in request:
        # Get tools from registry
        registry_tools = {t.function.name: t for t in registry.get_tool_definitions()}
        
        for tool_name in request["tools"]:
            if tool_name in registry_tools:
                # Convert registry tool to agent tool
                registry_tool = registry_tools[tool_name]
                
                # Create async handler for the tool
                async def handler(**kwargs):
                    tool_call = ToolCall(
                        id=str(uuid.uuid4()),
                        type="function",
                        function={
                            "name": tool_name,
                            "arguments": kwargs
                        }
                    )
                    result = await registry.execute_tool_call(tool_call)
                    return result
                
                # Create agent tool
                agent_tool = Tool(
                    name=tool_name,
                    description=registry_tool.function.description,
                    parameters=registry_tool.function.parameters,
                    async_handler=handler
                )
                
                tools.append(agent_tool)
    
    # Create the agent
    agent = Agent(
        name=name,
        model_name=model_name,
        reasoning_strategy=strategy,
        tools=tools,
        system_prompt=system_prompt
    )
    
    # Store the agent
    agents[agent.id] = agent
    
    return {
        "agent_id": agent.id,
        "name": agent.name,
        "model": agent.model_name,
        "reasoning_strategy": agent.reasoning_strategy.value,
        "tools": [t.name for t in agent.tools]
    }

@app.post("/v2/agents/{agent_id}/run", response_model=AgentResponse)
async def run_agent(
    agent_id: str,
    request: AgentRequest,
    user_id: Optional[str] = Depends(verify_api_key)
):
    """Run an agent with the given prompt"""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    agent = agents[agent_id]
    
    # Process images if present
    images = []
    if request.images:
        images = await _process_image_inputs(request.images)
        
        # Check if agent is multimodal capable
        if not agent.multimodal_capable:
            raise HTTPException(
                status_code=400,
                detail=f"Agent {agent_id} does not support multimodal inputs"
            )
    
    # Override system prompt if provided
    if request.system_prompt:
        agent.custom_system_prompt = request.system_prompt
    
    # Override tools if provided
    if request.tools:
        # Convert API tools to agent tools
        agent_tools = []
        for tool_def in request.tools:
            function_def = tool_def.function
            
            # Create tool
            async def handler(**kwargs):
                return {"message": "Tool execution not implemented in test mode"}
            
            tool = Tool(
                name=function_def.name,
                description=function_def.description or f"Call {function_def.name}",
                parameters=function_def.parameters,
                async_handler=handler,
                requires_auth=False
            )
            
            agent_tools.append(tool)
        
        agent.tools = agent_tools
    
    # Run the agent
    if images:
        # Use multimodal processing
        image_paths = [img["url"] for img in images]
        result = await agent.process_multimodal_input(request.prompt, image_paths)
    else:
        # Use regular processing
        result = await agent.run(request.prompt, max_iterations=request.max_steps)
    
    # Format response
    run_id = f"run_{uuid.uuid4()}"
    
    response = {
        "id": run_id,
        "model": agent.model_name,
        "final_answer": result.get("final_answer", ""),
        "steps": result.get("steps", []),
        "usage": {
            "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": result.get("usage", {}).get("total_tokens", 0)
        },
        "created": int(time.time())
    }
    
    return response

@app.post("/v2/agents/{agent_id}/run/stream")
async def stream_agent_run(
    agent_id: str,
    request: AgentRequest,
    user_id: Optional[str] = Depends(verify_api_key)
):
    """Stream the execution of an agent"""
    # Get the agent
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    agent = agents[agent_id]
    
    # Convert tools from the request to agent tools
    if request.tools:
        # Clear existing tools
        agent.tools = []
        
        for tool_def in request.tools:
            # Create async handler for the tool
            async def handler(**kwargs):
                tool_call = ToolCall(
                    id=str(uuid.uuid4()),
                    type="function",
                    function={
                        "name": tool_def.function.name,
                        "arguments": kwargs
                    }
                )
                result = await registry.execute_tool_call(tool_call)
                return result
            
            # Create agent tool
            agent_tool = Tool(
                name=tool_def.function.name,
                description=tool_def.function.description,
                parameters=tool_def.function.parameters,
                async_handler=handler
            )
            
            agent.tools.append(agent_tool)
    
    # Stream the agent execution
    run_id = str(uuid.uuid4())
    
    async def stream_generator():
        try:
            async for event in agent.run_stream(request.prompt, max_iterations=request.max_steps):
                # Format the event
                formatted_event = {
                    **event,
                    "agent_id": agent_id,
                    "run_id": run_id
                }
                
                yield json.dumps(formatted_event)
        except Exception as e:
            logger.error(f"Error streaming agent: {e}")
            yield json.dumps({
                "type": "error",
                "agent_id": agent_id,
                "run_id": run_id,
                "error": str(e),
                "timestamp": time.time()
            })
    
    return EventSourceResponse(stream_generator())

@app.get("/v2/models", response_model=ModelsResponse)
async def list_models(
    user_id: Optional[str] = Depends(verify_api_key)
):
    """List available models with v2 format"""
    global config, config_manager
    
    models_info = []
    
    # Use the model definitions from config
    for model_config in config_manager.config.models:
        # Determine model capabilities
        capabilities = ["completion"]
        
        if model_config.type == "llm":
            capabilities.extend(["function_calling", "agents"])
        
        info = ModelInfo(
            id=model_config.name,
            name=model_config.name,
            type=model_config.type,
            parameters=model_config.params,
            context_length=model_config.max_seq_len,
            quantization=model_config.quantization,
            capabilities=capabilities
        )
        models_info.append(info)
    
    # Also add any models found in the models directory
    models_dir = config["models_dir"]
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item)
            if os.path.isdir(model_path):
                # Check if already in the list
                if not any(m.id == item for m in models_info):
                    # Try to load config file
                    config_path = os.path.join(model_path, "config.json")
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r') as f:
                                model_data = json.load(f)
                                info = ModelInfo(
                                    id=item,
                                    name=item,
                                    type=model_data.get("model_type", "llm"),
                                    parameters=model_data.get("params"),
                                    context_length=model_data.get("max_seq_len", 8192),
                                    quantization=model_data.get("quantization"),
                                    capabilities=["completion"]
                                )
                                models_info.append(info)
                        except Exception:
                            # Add basic info if config can't be loaded
                            models_info.append(ModelInfo(
                                id=item,
                                name=item,
                                type="llm",
                                context_length=8192,
                                capabilities=["completion"]
                            ))
                    else:
                        # Add basic info if no config
                        models_info.append(ModelInfo(
                            id=item,
                            name=item,
                            type="llm",
                            context_length=8192,
                            capabilities=["completion"]
                        ))
    
    return ModelsResponse(models=models_info)

@app.get("/v2/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint for API v2"""
    return HealthStatus(
        status="healthy",
        timestamp=int(time.time()),
        version="2.0.0",
        components={
            "cache": cache is not None,
            "router": router is not None,
            "functions": len(registry.tools) > 0
        },
        models_loaded=list(models.keys())
    )

def start():
    """Start the FastAPI server"""
    import uvicorn
    
    # Load config to get host and port
    llamascale_config = get_config()
    
    # Start server
    uvicorn.run(
        "llamascale.api.v2.server:app", 
        host=llamascale_config.api_host,
        port=llamascale_config.api_port,
        log_level="info"
    )

if __name__ == "__main__":
    start() 