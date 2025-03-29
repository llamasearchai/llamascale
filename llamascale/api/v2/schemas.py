#!/usr/bin/env python3
"""
LlamaScale API v2 Schemas - Data models for the API with function calling support
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator

class RequestPriority(str, Enum):
    """Priority levels for request processing"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class ImageInput(BaseModel):
    """Image input for multimodal processing"""
    url: str = Field(..., description="URL or path to the image")
    type: str = Field("url", description="Type of image reference (url, base64, etc.)")

class FunctionDefinition(BaseModel):
    """Definition of a function that can be called by the model"""
    name: str = Field(..., description="The name of the function")
    description: Optional[str] = Field(None, description="A description of what the function does")
    parameters: Dict[str, Any] = Field(..., description="The parameters the function accepts")

class ToolDefinition(BaseModel):
    """Definition of a tool that can be used by the model"""
    type: str = Field("function", description="The type of tool, currently only 'function' is supported")
    function: FunctionDefinition = Field(..., description="The function definition")

class ToolChoice(BaseModel):
    """Specification for which tool to use"""
    type: str = Field("function", description="The type of tool to use")
    function: Dict[str, str] = Field(..., description="The function to use")

class ToolCall(BaseModel):
    """A call to a tool"""
    id: str = Field(..., description="A unique identifier for the tool call")
    type: str = Field("function", description="The type of tool, currently only 'function' is supported")
    function: Dict[str, Any] = Field(..., description="The function call details")

class GenerationRequest(BaseModel):
    """Request to generate text from a model with function calling support"""
    model: str = Field(..., description="The ID of the model to use for generation")
    prompt: str = Field(..., description="The prompt to generate from")
    max_tokens: Optional[int] = Field(100, description="The maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, description="The sampling temperature")
    top_p: Optional[float] = Field(0.95, description="The nucleus sampling probability")
    seed: Optional[int] = Field(None, description="The random seed for generation")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    tools: Optional[List[ToolDefinition]] = Field(None, description="The tools available to the model")
    tool_choice: Optional[Union[str, ToolChoice]] = Field(None, description="Which tool to use")
    priority: Optional[RequestPriority] = Field(RequestPriority.NORMAL, description="Priority for request processing")
    images: Optional[List[ImageInput]] = Field(None, description="Images for multimodal processing")

    @field_validator("temperature")
    def validate_temperature(cls, value):
        """Validate that temperature is in a reasonable range"""
        if value < 0 or value > 2:
            raise ValueError("Temperature must be between 0 and 2")
        return value

    @field_validator("top_p")
    def validate_top_p(cls, value):
        """Validate that top_p is in a reasonable range"""
        if value < 0 or value > 1:
            raise ValueError("Top-p must be between 0 and 1")
        return value

class StreamingResponse(BaseModel):
    """A streaming response chunk"""
    id: str = Field(..., description="A unique identifier for the request")
    model: str = Field(..., description="The model used for generation")
    created: int = Field(..., description="The timestamp of creation")
    object: str = Field("completion.chunk", description="The type of object")
    choices: List[Dict[str, Any]] = Field(..., description="The completion choices")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Any tool calls made by the model")

class UsageInfo(BaseModel):
    """Token usage information"""
    prompt_tokens: int = Field(..., description="The number of tokens in the prompt")
    completion_tokens: int = Field(..., description="The number of tokens in the completion")
    total_tokens: int = Field(..., description="The total number of tokens")

class GenerationResponse(BaseModel):
    """Response from generating text"""
    id: str = Field(..., description="A unique identifier for the request")
    model: str = Field(..., description="The model used for generation")
    choices: List[Dict[str, Any]] = Field(..., description="The completion choices")
    usage: UsageInfo = Field(..., description="Token usage information")
    created: int = Field(..., description="The timestamp of creation")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Any tool calls made by the model")

class ToolCallResult(BaseModel):
    """Result of executing a tool call"""
    id: str = Field(..., description="A unique identifier for the result")
    type: str = Field("function_result", description="The type of result")
    function: Dict[str, Any] = Field(..., description="The function result details")

class ModelInfo(BaseModel):
    """Information about a model"""
    id: str = Field(..., description="The model identifier")
    name: str = Field(..., description="The model name")
    type: Optional[str] = Field("llm", description="The model type (llm, embedding, etc.)")
    parameters: Optional[int] = Field(None, description="The number of parameters in the model")
    context_length: Optional[int] = Field(None, description="The maximum context length")
    quantization: Optional[str] = Field(None, description="The quantization method used")
    capabilities: Optional[List[str]] = Field(None, description="The model's capabilities")
    supports_multimodal: Optional[bool] = Field(False, description="Whether the model supports multimodal inputs")

class ModelsResponse(BaseModel):
    """Response listing available models"""
    models: List[ModelInfo] = Field(..., description="List of available models")

class AgentAction(BaseModel):
    """An action taken by an agent"""
    type: str = Field(..., description="The type of action")
    content: Optional[str] = Field(None, description="The content of the action")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Any tool calls made during the action")

class AgentRequest(BaseModel):
    """Request to run an agent"""
    prompt: str = Field(..., description="The user prompt for the agent")
    max_steps: Optional[int] = Field(10, description="Maximum number of steps the agent can take")
    tools: Optional[List[ToolDefinition]] = Field(None, description="Tools available to the agent")
    system_prompt: Optional[str] = Field(None, description="System prompt to override agent's default")
    images: Optional[List[ImageInput]] = Field(None, description="Images for multimodal processing")

class AgentResponse(BaseModel):
    """Response from running an agent"""
    id: str = Field(..., description="A unique identifier for the run")
    model: str = Field(..., description="The model used by the agent")
    final_answer: str = Field(..., description="The final answer from the agent")
    steps: List[AgentAction] = Field(..., description="The steps taken by the agent")
    usage: UsageInfo = Field(..., description="Token usage information")
    created: int = Field(..., description="The timestamp of creation")

class HealthStatus(BaseModel):
    """Health status of the API"""
    status: str = Field(..., description="The status of the API")
    timestamp: int = Field(..., description="The current timestamp")
    version: str = Field(..., description="The API version")
    components: Optional[Dict[str, bool]] = Field(None, description="Status of API components")
    models_loaded: Optional[List[str]] = Field(None, description="Models currently loaded in memory") 