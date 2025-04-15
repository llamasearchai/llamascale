"""
LlamaScale Agent Framework - Advanced reasoning and tool use capabilities
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from llamascale.drivers.mlx.engine import MLXConfig, MLXModelEngine
from llamascale.tools.cli.config import ConfigManager, get_config

logger = logging.getLogger(__name__)


class ReasoningStrategy(str, Enum):
    """Types of reasoning strategies available to agents"""

    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    REACT = "react"
    REFLECTION = "reflection"
    VERIFICATION = "verification"
    MULTIMODAL_REASONING = "multimodal_reasoning"


class ToolResult(BaseModel):
    """Result from a tool execution"""

    success: bool = Field(..., description="Whether the tool execution succeeded")
    result: Any = Field(None, description="Result of the tool execution")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    meta: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata about execution"
    )


class Tool(BaseModel):
    """Tool that can be used by an agent"""

    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters the tool accepts"
    )
    async_handler: Callable = Field(
        ..., description="Async function that implements the tool"
    )
    requires_auth: bool = Field(
        False, description="Whether the tool requires authentication"
    )

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given parameters"""
        try:
            result = await self.async_handler(**kwargs)
            return ToolResult(success=True, result=result)
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}")
            return ToolResult(success=False, error=str(e))


class ImageTool(Tool):
    """Tool for handling and analyzing images"""

    supported_formats: List[str] = Field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp"],
        description="Supported image formats",
    )
    max_image_size: int = Field(
        default=10 * 1024 * 1024, description="Maximum image size in bytes"
    )
    use_mlx_vision: bool = Field(
        default=True, description="Whether to use MLX Vision for image analysis"
    )

    async def analyze_image(
        self, image_path: str, analysis_type: str = "caption"
    ) -> Dict[str, Any]:
        """Analyze an image using MLX Vision or other frameworks

        Args:
            image_path: Path to the image file
            analysis_type: Type of analysis to perform (caption, objects, features)

        Returns:
            Analysis results
        """
        try:
            # Try to import image handling libraries
            try:
                from PIL import Image

                has_pil = True
            except ImportError:
                has_pil = False
                return ToolResult(
                    success=False,
                    error="PIL not installed. Install with 'pip install pillow'",
                )

            # Check if the file exists and is an image
            if not os.path.exists(image_path):
                return ToolResult(success=False, error=f"Image not found: {image_path}")

            if not has_pil:
                return ToolResult(
                    success=False,
                    error="PIL not installed. Install with 'pip install pillow'",
                )

            # Load and validate the image
            image = Image.open(image_path)

            # Perform the requested analysis
            if analysis_type == "caption":
                return await self._generate_caption(image)
            elif analysis_type == "objects":
                return await self._detect_objects(image)
            elif analysis_type == "features":
                return await self._extract_features(image)
            else:
                return ToolResult(
                    success=False, error=f"Unknown analysis type: {analysis_type}"
                )

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return ToolResult(success=False, error=f"Error analyzing image: {e}")

    async def _generate_caption(self, image) -> Dict[str, Any]:
        """Generate a caption for an image"""
        try:
            # Try to use MLX Vision if available
            try:
                import mlx.vision

                has_mlx_vision = True
            except ImportError:
                has_mlx_vision = False

            if has_mlx_vision and self.use_mlx_vision:
                # Use MLX Vision for captioning
                caption = "Image caption generated with MLX Vision"  # Placeholder
                return {"caption": caption, "confidence": 0.95}
            else:
                # Fallback to simpler analysis
                width, height = image.size
                format_name = image.format
                mode = image.mode

                return {
                    "caption": f"An image of format {format_name}, dimensions {width}x{height}, mode {mode}",
                    "metadata": {
                        "width": width,
                        "height": height,
                        "format": format_name,
                        "mode": mode,
                    },
                }

        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return {"error": str(e)}

    async def _detect_objects(self, image) -> Dict[str, Any]:
        """Detect objects in an image"""
        # Placeholder for object detection
        return {"objects": ["object1", "object2"], "confidence": 0.8}

    async def _extract_features(self, image) -> Dict[str, Any]:
        """Extract features from an image"""
        # Placeholder for feature extraction
        return {"features": [0.1, 0.2, 0.3, 0.4], "dimensions": 4}


@dataclass
class AgentMemory:
    """Memory for an agent to store information across steps"""

    conversations: List[Dict[str, Any]] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    long_term_memory: Dict[str, Any] = field(default_factory=dict)
    reflections: List[str] = field(default_factory=list)

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.conversations.append({"role": role, "content": content})

    def get_recent_messages(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the n most recent messages"""
        return self.conversations[-n:] if len(self.conversations) > 0 else []

    def store(self, key: str, value: Any):
        """Store a value in working memory"""
        self.working_memory[key] = value

    def retrieve(self, key: str) -> Any:
        """Retrieve a value from working memory"""
        return self.working_memory.get(key)

    def archive(self, key: str, value: Any):
        """Store a value in long-term memory"""
        self.long_term_memory[key] = value

    def add_reflection(self, reflection: str):
        """Add a reflection on the agent's performance"""
        self.reflections.append(reflection)


class Agent:
    """LlamaScale Agent with advanced reasoning capabilities"""

    def __init__(
        self,
        name: str,
        model_name: str,
        reasoning_strategy: ReasoningStrategy = ReasoningStrategy.REACT,
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        multimodal_capable: bool = False,
    ):
        """Initialize the agent

        Args:
            name: Name of the agent
            model_name: Name of the model to use for reasoning
            reasoning_strategy: Strategy to use for reasoning
            tools: List of tools available to the agent
            system_prompt: Custom system prompt
            multimodal_capable: Whether the agent can handle multimodal inputs
        """
        self.name = name
        self.model_name = model_name
        self.reasoning_strategy = reasoning_strategy
        self.tools = tools or []
        self.custom_system_prompt = system_prompt
        self.memory = AgentMemory()
        self.multimodal_capable = multimodal_capable
        self.image_tool = None

        # Find any image tools
        for tool in self.tools:
            if isinstance(tool, ImageTool):
                self.image_tool = tool
                break

        # Initialize model engine lazily
        self._model_engine = None
        self._initialized = False
        self.id = str(uuid.uuid4())

    async def initialize(self):
        """Initialize the agent's model engine"""
        config = get_config()

        # Find model-specific configuration
        model_config = None
        for model in config.models:
            if model.name == self.model_name:
                model_config = model
                break

        if not model_config:
            raise ValueError(f"Model {self.model_name} not found in configuration")

        # Initialize the model engine
        model_path = os.path.join(config.models_dir, self.model_name)
        if not os.path.exists(model_path):
            raise ValueError(f"Model {self.model_name} not found at {model_path}")

        logger.info(f"Initializing model engine for agent {self.name}")
        from llamascale.drivers.mlx.engine import MLXConfig, MLXModelEngine

        mlx_config = MLXConfig()
        if model_config.quantization:
            mlx_config.quantization = model_config.quantization

        self._model_engine = MLXModelEngine(model_path, mlx_config)
        self._initialized = True
        logger.info(f"Agent {self.name} initialized with model {self.model_name}")

    def _default_system_prompt(self) -> str:
        """Generate the default system prompt based on reasoning strategy"""
        base_prompt = f"You are {self.name}, an AI assistant with {self.reasoning_strategy.value} reasoning capabilities."

        if self.reasoning_strategy == ReasoningStrategy.REACT:
            base_prompt += """
You solve problems by thinking step by step, using the following process:
1. Think: Analyze the problem and break it down into smaller steps
2. Act: Choose a tool to gather information or perform an action
3. Observe: Review the results of your actions
4. Repeat until you can provide a final answer

Use the format:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <tool parameters as JSON>
Observation: <result of the action>
... (repeat as needed)
Final Answer: <your final response>
"""
        elif self.reasoning_strategy == ReasoningStrategy.REFLECTION:
            base_prompt += """
You solve problems by thinking step by step and reflecting on your progress:
1. Think: Analyze the problem and break it down
2. Act: Choose a tool or reasoning step
3. Observe: Review the results
4. Reflect: Consider if your approach is working and how to improve
5. Repeat until you can provide a final answer

Use the format:
Thought: <your reasoning>
Action: <tool_name or reasoning step>
Action Input: <tool parameters or reasoning>
Observation: <result of the action>
Reflection: <analysis of progress and strategy adjustment>
... (repeat as needed)
Final Answer: <your final response>
"""
        elif self.reasoning_strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            base_prompt += """
You solve problems by thinking step-by-step:
1. Break down the problem into logical steps
2. Work through each step sequentially
3. Articulate your reasoning at each step
4. Arrive at a detailed final answer

Use the format:
Step 1: <description of the first step and your reasoning>
Step 2: <description of the second step and your reasoning>
... (continue with all required steps)
Final Answer: <your final response>
"""
        elif self.reasoning_strategy == ReasoningStrategy.TREE_OF_THOUGHT:
            base_prompt += """
You solve problems by exploring multiple reasoning paths:
1. Break down the problem into key decision points
2. At each decision point, consider 2-3 alternative approaches
3. Evaluate each approach briefly
4. Choose the most promising approach and continue
5. Arrive at a final answer through the best path

Use the format:
Decision Point: <description of the decision point>
Option 1: <first approach>
Option 2: <second approach>
Option 3: <third approach>
Evaluation: <assessment of the options>
Selected Approach: <the option you're choosing>
... (repeat for next decision point)
Final Answer: <your final response>
"""
        elif self.reasoning_strategy == ReasoningStrategy.VERIFICATION:
            base_prompt += """
You solve problems by carefully verifying your work:
1. Break down the problem into steps
2. Solve each step
3. Verify your solution for each step
4. If you find errors, correct them
5. Provide a final answer with confidence assessment

Use the format:
Problem Analysis: <break down of the problem>
Initial Solution: <your first attempt at solving>
Verification: <checking your work>
Corrections: <fixing any errors found>
Final Answer: <your final response with confidence level>
"""

        # Add available tools to the prompt
        if self.tools:
            base_prompt += "\n\nYou have access to the following tools:\n"
            for tool in self.tools:
                base_prompt += f"- {tool.name}: {tool.description}\n"
                if tool.parameters:
                    base_prompt += f"  Parameters: {json.dumps(tool.parameters)}\n"

        return base_prompt

    async def _get_next_action(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get the next action from the model"""
        if not self._model_engine:
            await self.initialize()

        # Format messages for the model
        formatted_messages = [
            {
                "role": "system",
                "content": self.custom_system_prompt or self._default_system_prompt(),
            }
        ]
        formatted_messages.extend(messages)

        # Generate a response
        prompt = self._format_messages_as_prompt(formatted_messages)
        response = await self._model_engine.generate(
            prompt=prompt, max_tokens=1024, temperature=0.7, top_p=0.95
        )

        return self._parse_model_response(response["text"])

    def _format_messages_as_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages as a prompt for the model"""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
            elif role == "tool":
                prompt += f"Tool Result: {content}\n\n"

        prompt += "Assistant: "
        return prompt

    async def _parse_model_response(self, response: str) -> Dict[str, Any]:
        """Parse the model response to extract the next action"""
        # Default structure
        parsed = {"type": "thinking", "content": response}

        # Extract tool calls based on the reasoning strategy
        if self.reasoning_strategy == ReasoningStrategy.REACT:
            # Look for tool calls in ReAct format
            if "Action:" in response and "Action Input:" in response:
                action_match = (
                    response.split("Action:")[1].split("Action Input:")[0].strip()
                )
                action_input_match = response.split("Action Input:")[1].strip()

                # Parse as a tool call
                parsed = {
                    "type": "tool_use",
                    "content": f"I'll use the {action_match} tool",
                    "tool_calls": [
                        {
                            "id": f"call_{uuid.uuid4()}",
                            "type": "function",
                            "function": {
                                "name": action_match,
                                "arguments": action_input_match,
                            },
                        }
                    ],
                }
            elif "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[1].strip()
                parsed = {"type": "final_answer", "content": final_answer}

        return parsed

    async def process_multimodal_input(
        self, text: str, image_paths: List[str] = None
    ) -> Dict[str, Any]:
        """Process a multimodal input with both text and images

        Args:
            text: Text input from the user
            image_paths: Paths to images to process

        Returns:
            Processing results
        """
        if not self.multimodal_capable:
            return {
                "error": "This agent is not configured for multimodal inputs. Set multimodal_capable=True when creating the agent."
            }

        if not image_paths:
            # If no images, just use the regular run method
            return await self.run(text)

        # Check if we have an image tool
        if not self.image_tool:
            return {
                "error": "No image tool available. Add an ImageTool to the agent's tools."
            }

        # Process the images
        image_analysis_results = []
        for image_path in image_paths:
            # Analyze the image using the image tool
            analysis = await self.image_tool.analyze_image(image_path, "caption")
            if isinstance(analysis, ToolResult):
                if not analysis.success:
                    image_analysis_results.append(
                        {"path": image_path, "error": analysis.error}
                    )
                    continue
                analysis = analysis.result

            image_analysis_results.append({"path": image_path, "analysis": analysis})

        # Create a prompt that includes the image descriptions
        image_descriptions = []
        for idx, result in enumerate(image_analysis_results):
            if "analysis" in result and "caption" in result["analysis"]:
                image_descriptions.append(
                    f"Image {idx+1}: {result['analysis']['caption']}"
                )
            else:
                image_descriptions.append(f"Image {idx+1}: [Failed to analyze]")

        multimodal_prompt = f"""
User query: {text}

The user has also provided the following images:
{'\n'.join(image_descriptions)}

Please respond to the user's query taking into account both the text and the images.
"""

        # Run the agent with the enhanced prompt
        return await self.run(multimodal_prompt)

    async def run(self, query: str, max_iterations: int = 10) -> Dict[str, Any]:
        """Run the agent on a query

        Args:
            query: User query to process
            max_iterations: Maximum number of reasoning steps

        Returns:
            Dict containing the final answer and reasoning trace
        """
        # Initialize if needed
        if not self._model_engine:
            await self.initialize()

        # Add the user query to memory
        self.memory.add_message("user", query)

        # Initialize the reasoning trace
        reasoning_trace = []

        # Main reasoning loop
        iteration = 0
        while iteration < max_iterations:
            # Get the next action from the model
            next_action = await self._get_next_action(self.memory.get_recent_messages())

            # Record the action in the trace
            reasoning_trace.append(next_action)

            # If it's a final answer, we're done
            if next_action["type"] == "final_answer":
                final_answer = next_action["content"]
                self.memory.add_message("assistant", final_answer)
                return {"answer": final_answer, "reasoning_trace": reasoning_trace}

            # If it's an action, execute it
            if next_action["type"] == "action":
                action_name = next_action["content"]

                # Find the tool
                tool = next((t for t in self.tools if t.name == action_name), None)

                if tool:
                    # Execute the tool
                    result = await tool.execute()

                    # Add the result to memory
                    result_str = json.dumps(result.dict())
                    self.memory.add_message("tool", result_str)
                else:
                    # Tool not found
                    error_msg = f"Tool {action_name} not found. Available tools: {', '.join(t.name for t in self.tools)}"
                    self.memory.add_message("tool", error_msg)

            # Increment the iteration counter
            iteration += 1

        # If we reach here, we've exceeded the maximum number of iterations
        return {
            "answer": "I apologize, but I was unable to find a satisfactory answer within the allowed number of reasoning steps.",
            "reasoning_trace": reasoning_trace,
        }

    async def run_stream(
        self, query: str, max_iterations: int = 10
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the agent on a query and stream the results

        Args:
            query: User query to process
            max_iterations: Maximum number of reasoning steps

        Yields:
            Dict containing step information
        """
        # Initialize if needed
        if not self._model_engine:
            await self.initialize()

        # Add the user query to memory
        self.memory.add_message("user", query)

        # Initial event
        yield {
            "type": "start",
            "agent_id": self.id,
            "model": self.model_name,
            "timestamp": time.time(),
            "max_iterations": max_iterations,
        }

        # Main reasoning loop
        iteration = 0
        while iteration < max_iterations:
            # Get the next action from the model
            next_action = await self._get_next_action(self.memory.get_recent_messages())

            # Stream the action
            yield {
                "type": "step",
                "iteration": iteration,
                "action": next_action,
                "timestamp": time.time(),
            }

            # If it's a final answer, we're done
            if next_action["type"] == "final_answer":
                final_answer = next_action["content"]
                self.memory.add_message("assistant", final_answer)

                yield {
                    "type": "end",
                    "answer": final_answer,
                    "iterations": iteration + 1,
                    "timestamp": time.time(),
                }
                return

            # If it's an action, execute it
            if next_action["type"] == "action":
                action_name = next_action["content"]

                # Find the tool
                tool = next((t for t in self.tools if t.name == action_name), None)

                if tool:
                    # Stream tool execution start
                    yield {
                        "type": "tool_start",
                        "tool": action_name,
                        "timestamp": time.time(),
                    }

                    # Execute the tool
                    result = await tool.execute()

                    # Stream tool execution result
                    yield {
                        "type": "tool_result",
                        "tool": action_name,
                        "result": result.dict(),
                        "timestamp": time.time(),
                    }

                    # Add the result to memory
                    result_str = json.dumps(result.dict())
                    self.memory.add_message("tool", result_str)
                else:
                    # Tool not found
                    error_msg = f"Tool {action_name} not found. Available tools: {', '.join(t.name for t in self.tools)}"
                    self.memory.add_message("tool", error_msg)

                    # Stream tool error
                    yield {
                        "type": "tool_error",
                        "tool": action_name,
                        "error": error_msg,
                        "timestamp": time.time(),
                    }

            # Increment the iteration counter
            iteration += 1

        # If we reach here, we've exceeded the maximum number of iterations
        yield {
            "type": "end",
            "answer": "I apologize, but I was unable to find a satisfactory answer within the allowed number of reasoning steps.",
            "iterations": max_iterations,
            "timestamp": time.time(),
        }
