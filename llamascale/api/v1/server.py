#!/usr/bin/env python3
"""
LlamaScale API Server - REST API for LlamaScale functionality
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from llamascale.drivers.mlx.engine import MLXConfig, MLXModelEngine
from llamascale.orchestrator.caching.hybrid import HybridCache
from llamascale.orchestrator.routing.smart import InferenceRouter, RequestPriority
from llamascale.tools.cli.config import ConfigManager, LlamaScaleConfig, get_config
from llamascale.tools.cli.init import init_llamascale_dirs

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("llamascale.api")

# Initialize directories
init_llamascale_dirs()

# Initialize FastAPI app
app = FastAPI(
    title="LlamaScale API",
    description="API for LlamaScale Ultra 6.0 - Mac-Native Enterprise LLM Orchestration Platform",
    version="6.0.0",
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


class GenerationRequest(BaseModel):
    """Model for generation request"""

    model: str = Field(..., description="Model to use for generation")
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature (0.0 to 1.0)")
    top_p: float = Field(0.9, description="Top-p sampling parameter (0.0 to 1.0)")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    stream: bool = Field(False, description="Whether to stream the response")
    priority: Optional[int] = Field(
        RequestPriority.NORMAL, description="Request priority (0-3)"
    )


class ModelInfo(BaseModel):
    """Model for model information"""

    name: str = Field(..., description="Model name")
    type: Optional[str] = Field(None, description="Model type")
    parameters: Optional[str] = Field(None, description="Model parameters")
    context_length: Optional[int] = Field(None, description="Context length")
    quantization: Optional[str] = Field(None, description="Quantization level")


class ModelsResponse(BaseModel):
    """Model for models list response"""

    models: List[ModelInfo] = Field(..., description="List of available models")


class CacheStats(BaseModel):
    """Model for cache statistics"""

    memory_hits: int
    redis_hits: int
    disk_hits: int
    semantic_hits: int
    misses: int
    sets: int
    total_hits: int
    total_ops: int
    hit_rate: float
    memory_items: int
    memory_size_limit: int
    disk_items: Optional[int] = None
    disk_size: Optional[int] = None
    redis_memory_used: Optional[str] = None
    redis_keys: Optional[int] = None


@app.on_event("startup")
async def startup_event():
    """Initialize server components on startup"""
    global config, config_manager, router, cache

    # Load configuration using the new ConfigManager
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

    # Set FastAPI app host and port from config
    logger.info(
        f"API server will listen on {llamascale_config.api_host}:{llamascale_config.api_port}"
    )

    logger.info("LlamaScale API server initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    if router:
        await router.shutdown()
    logger.info("LlamaScale API server shutdown")


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


@app.post("/v1/generate")
async def generate(request: GenerationRequest):
    """Generate text from a model"""
    start_time = time.time()

    # Use router if available, otherwise use direct model access
    if router and request.model in config.get("mlx_models", []):
        # Format request for router
        router_request = {
            "model": request.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "seed": request.seed,
            "priority": request.priority,
        }

        # Dispatch to router
        result = await router.dispatch(router_request)
    else:
        # Check cache first
        cache_key = f"{request.model}:{request.prompt}:{request.max_tokens}:{request.temperature}:{request.top_p}:{request.seed}"
        cached_result = await cache.get(cache_key, request.model)

        if cached_result:
            logger.info(f"Cache hit for {request.model}")
            return cached_result

        # Get model engine
        engine = await get_model_engine(request.model)

        # Generate text
        result = await engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
        )

        # Cache result
        await cache.set(cache_key, result, request.model)

    # Add timing information
    result["api_timing"] = {"request_time": time.time() - start_time}

    return result


@app.post("/v1/generate/stream")
async def generate_stream(request: GenerationRequest):
    """Generate text from a model with streaming response"""
    if not request.stream:
        return await generate(request)

    # Get model engine
    engine = await get_model_engine(request.model)

    # Check if the engine supports streaming natively
    if hasattr(engine, "generate_stream"):
        # Use native streaming
        return StreamingResponse(
            engine.generate_stream(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                seed=request.seed,
            ),
            media_type="application/x-ndjson",
        )

    # Fall back to simulated streaming
    async def stream_generator():
        """Stream generation results"""
        # In a full implementation, this would use a real streaming API
        # For this demonstration, we'll simulate streaming

        # Header
        yield json.dumps({"event": "start"}) + "\n"

        prompt_tokens = len(engine.tokenizer.encode(request.prompt))
        generated_tokens = 0
        start_time = time.time()

        # Generate full response first (in a real system, this would be streamed token by token)
        result = await engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
        )

        # Split the result text into chunks to simulate streaming
        text = result["text"]
        chunk_size = max(1, len(text) // 10)
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        # Stream chunks
        for i, chunk in enumerate(chunks):
            # Simulate token-by-token generation delay
            await asyncio.sleep(0.1)

            # Calculate token count
            tokens_in_chunk = min(5, request.max_tokens - generated_tokens)
            generated_tokens += tokens_in_chunk

            # Create chunk response
            chunk_data = {
                "event": "chunk",
                "text": chunk,
                "chunk_index": i,
                "tokens": tokens_in_chunk,
            }

            yield json.dumps(chunk_data) + "\n"

        # End of generation
        total_time = time.time() - start_time
        final_data = {
            "event": "done",
            "total_tokens": generated_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generated_tokens,
            "total_time": total_time,
        }

        yield json.dumps(final_data) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models"""
    global config, config_manager

    models_info = []

    # Use the model definitions from config
    for model_config in config_manager.config.models:
        info = ModelInfo(
            name=model_config.name,
            type=model_config.type,
            parameters=model_config.params,
            context_length=model_config.max_seq_len,
            quantization=model_config.quantization,
        )
        models_info.append(info)

    # Also add any models found in the models directory
    models_dir = config["models_dir"]
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item)
            if os.path.isdir(model_path):
                # Check if already in the list
                if not any(m.name == item for m in models_info):
                    # Try to load config file
                    config_path = os.path.join(model_path, "config.json")
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, "r") as f:
                                model_data = json.load(f)
                                info = ModelInfo(
                                    name=item,
                                    type=model_data.get("model_type"),
                                    parameters=model_data.get("params"),
                                    context_length=model_data.get("max_seq_len"),
                                    quantization=model_data.get("quantization"),
                                )
                                models_info.append(info)
                        except Exception:
                            # Add basic info if config can't be loaded
                            models_info.append(ModelInfo(name=item))
                    else:
                        # Add basic info if no config
                        models_info.append(ModelInfo(name=item))

    return ModelsResponse(models=models_info)


@app.get("/v1/cache/stats", response_model=CacheStats)
async def get_cache_stats():
    """Get cache statistics"""
    stats = await cache.get_stats()
    return stats


@app.post("/v1/cache/clear")
async def clear_cache(memory: bool = True, disk: bool = True, redis: bool = False):
    """Clear cache"""
    if memory:
        await cache.clear("memory")

    if disk:
        await cache.clear("disk")

    if redis and cache.redis:
        await cache.clear("redis")

    return {
        "status": "success",
        "message": "Cache cleared",
        "components": {
            "memory": memory,
            "disk": disk,
            "redis": redis and cache.redis is not None,
        },
    }


@app.get("/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {
            "cache": cache is not None,
            "router": router is not None,
            "models_loaded": len(models),
        },
    }


@app.get("/v1/config")
async def get_server_config():
    """Get current server configuration"""
    global config_manager

    # Return a safe subset of the configuration
    safe_config = {
        "mlx_config": {
            k: v for k, v in config["mlx_config"].items() if k not in ["api_keys"]
        },
        "cache_config": {
            k: v for k, v in config["cache_config"].items() if k not in ["auth_token"]
        },
        "api_port": config.get("api_port", 8000),
        "api_host": config.get("api_host", "127.0.0.1"),
        "enable_mlx": config.get("enable_mlx", True),
        "enable_cuda": config.get("enable_cuda", False),
        "models_available": [m.name for m in config_manager.config.models],
    }

    return safe_config


def start():
    """Start the FastAPI server"""
    import uvicorn

    # Load config to get host and port
    llamascale_config = get_config()

    # Start server
    uvicorn.run(
        "llamascale.api.v1.server:app",
        host=llamascale_config.api_host,
        port=llamascale_config.api_port,
        log_level="info",
    )


if __name__ == "__main__":
    start()
