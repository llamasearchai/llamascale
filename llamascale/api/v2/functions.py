#!/usr/bin/env python3
"""
LlamaScale API v2 Functions Module - Support for function calling
"""

import asyncio
import inspect
import json
import logging
import os
import uuid
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, Field, create_model

from .schemas import FunctionDefinition, ToolCall, ToolDefinition

logger = logging.getLogger("llamascale.api.v2.functions")


class ToolRegistry:
    """Registry for tools that can be called by models"""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Register a function as a tool"""
        func_name = name or func.__name__
        func_desc = description or func.__doc__ or f"Call {func_name}"

        # Get function signature
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Build parameters schema
        parameters = {"type": "object", "properties": {}, "required": []}

        for param_name, param in sig.parameters.items():
            # Skip self/cls for methods
            if param_name in ("self", "cls"):
                continue

            # Get parameter type
            param_type = type_hints.get(param_name, type(None))

            # Add to schema
            param_schema = self._type_to_json_schema(param_type)
            parameters["properties"][param_name] = param_schema

            # Check if required
            if param.default is inspect.Parameter.empty:
                parameters["required"].append(param_name)

            # Add description from default field if available
            if param.default is not inspect.Parameter.empty and isinstance(
                param.default, Field
            ):
                parameters["properties"][param_name][
                    "description"
                ] = param.default.description

        # Store the function
        self.tools[func_name] = {
            "func": func,
            "description": func_desc,
            "parameters": parameters,
            "is_async": asyncio.iscoroutinefunction(func),
        }

        logger.info(f"Registered tool: {func_name}")

    def _type_to_json_schema(self, typ: Type) -> Dict[str, Any]:
        """Convert Python type to JSON Schema"""
        origin = get_origin(typ)
        args = get_args(typ)

        # Handle Union types (Optional is Union[T, None])
        if origin is Union:
            # If one of the types is NoneType, treat as optional
            if type(None) in args:
                non_none_types = [t for t in args if t is not type(None)]
                if len(non_none_types) == 1:
                    schema = self._type_to_json_schema(non_none_types[0])
                    return schema

            # Otherwise, use oneOf for union types
            return {"oneOf": [self._type_to_json_schema(arg) for arg in args]}

        # Handle List types
        elif origin is list:
            item_type = args[0] if args else Any
            return {"type": "array", "items": self._type_to_json_schema(item_type)}

        # Handle Dict types
        elif origin is dict:
            return {"type": "object"}

        # Handle primitive types
        elif typ is str:
            return {"type": "string"}
        elif typ is int:
            return {"type": "integer"}
        elif typ is float:
            return {"type": "number"}
        elif typ is bool:
            return {"type": "boolean"}

        # Handle Pydantic models
        elif isinstance(typ, type) and issubclass(typ, BaseModel):
            schema = typ.schema()
            return {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }

        # Default to object for complex types
        else:
            return {"type": "object"}

    def get_tool_definitions(self) -> List[ToolDefinition]:
        """Get tool definitions for API consumption"""
        tool_defs = []

        for name, tool_info in self.tools.items():
            tool_def = ToolDefinition(
                type="function",
                function=FunctionDefinition(
                    name=name,
                    description=tool_info["description"],
                    parameters=tool_info["parameters"],
                ),
            )
            tool_defs.append(tool_def)

        return tool_defs

    async def execute_tool_call(self, tool_call: ToolCall) -> Any:
        """Execute a tool call"""
        if tool_call.type != "function":
            raise ValueError(f"Unsupported tool type: {tool_call.type}")

        func_name = tool_call.function.get("name")
        if not func_name:
            raise ValueError("Function name is required")

        if func_name not in self.tools:
            raise ValueError(f"Function {func_name} not found")

        tool_info = self.tools[func_name]
        func = tool_info["func"]

        # Parse arguments
        func_args = {}
        if "arguments" in tool_call.function:
            args_str = tool_call.function["arguments"]
            if isinstance(args_str, str):
                try:
                    func_args = json.loads(args_str)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid function arguments: {args_str}")
            elif isinstance(args_str, dict):
                func_args = args_str

        # Execute the function
        try:
            if tool_info["is_async"]:
                result = await func(**func_args)
            else:
                result = func(**func_args)

            return result
        except Exception as e:
            logger.error(f"Error executing tool {func_name}: {e}")
            raise


# Create registry instance
registry = ToolRegistry()


# Function to register tools
def register_tool(name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to register a function as a tool"""

    def decorator(func):
        registry.register(func, name, description)
        return func

    return decorator


# Built-in tools


@register_tool(description="Get current weather in a given location")
async def get_current_weather(
    location: str = Field(
        ..., description="The city and state, e.g. San Francisco, CA"
    ),
    unit: str = Field(
        "celsius", description="The temperature unit to use: 'celsius' or 'fahrenheit'"
    ),
) -> Dict[str, Any]:
    """Get the current weather in a location"""
    # This is a mock implementation
    await asyncio.sleep(0.5)  # Simulate API call

    # Generate mock weather data
    import random

    temps = {
        "San Francisco": {"celsius": (15, 25), "fahrenheit": (59, 77)},
        "New York": {"celsius": (10, 30), "fahrenheit": (50, 86)},
        "London": {"celsius": (8, 20), "fahrenheit": (46, 68)},
        "Tokyo": {"celsius": (12, 28), "fahrenheit": (54, 82)},
        "Sydney": {"celsius": (18, 32), "fahrenheit": (64, 90)},
    }

    conditions = ["sunny", "partly cloudy", "cloudy", "rainy", "stormy", "windy"]

    # Find closest known city
    chosen_city = None
    for city in temps.keys():
        if city.lower() in location.lower():
            chosen_city = city
            break

    if not chosen_city:
        chosen_city = random.choice(list(temps.keys()))

    # Normalize unit
    unit_key = "celsius"
    if unit.lower() in ("f", "fahrenheit"):
        unit_key = "fahrenheit"

    # Generate temperature
    min_temp, max_temp = temps[chosen_city][unit_key]
    temp = round(random.uniform(min_temp, max_temp), 1)

    # Generate condition
    condition = random.choice(conditions)

    return {
        "location": location,
        "temperature": temp,
        "unit": unit_key,
        "condition": condition,
        "humidity": round(random.uniform(30, 95)),
        "wind_speed": round(random.uniform(0, 20), 1),
        "forecast": f"The weather in {location} is {condition} with a temperature of {temp}°{unit_key[0].upper()}.",
        "timestamp": datetime.now().isoformat(),
    }


@register_tool(description="Perform a calculation using a mathematical expression")
def calculate(
    expression: str = Field(..., description="The mathematical expression to evaluate")
) -> Dict[str, Any]:
    """Evaluate a mathematical expression"""
    import math
    import operator

    # Define safe operations
    safe_dict = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        # Math functions
        "ceil": math.ceil,
        "floor": math.floor,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        # Constants
        "pi": math.pi,
        "e": math.e,
        # Operators
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "//": operator.floordiv,
        "%": operator.mod,
        "**": operator.pow,
    }

    try:
        # Sanitize input
        if any(
            keyword in expression
            for keyword in ["__", "import", "eval", "exec", "open"]
        ):
            raise ValueError("Expression contains forbidden keywords")

        # Evaluate expression
        # Note: This is still potentially unsafe and should be improved in production
        result = eval(expression, {"__builtins__": {}}, safe_dict)

        return {"expression": expression, "result": result, "isError": False}
    except Exception as e:
        return {"expression": expression, "result": str(e), "isError": True}
