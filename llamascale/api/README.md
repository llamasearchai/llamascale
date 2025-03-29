# LlamaScale API

The LlamaScale API provides HTTP endpoints to interact with LlamaScale's LLM capabilities through a REST API interface. This enables developers to integrate LlamaScale's Mac-native inference into web applications, services, and other systems.

## API Versions

LlamaScale supports multiple API versions:

- **v1 API**: Basic text generation API compatible with standard LLM interfaces
- **v2 API**: Advanced API with function calling and agent capabilities

## Running the API Server

```bash
# Running v1 API
llamascale_api

# Running v2 API (with advanced capabilities)
llamascale_api --v2
```

## API v1 Endpoints

### Text Generation

```
POST /v1/completions
```

Generate text completions from a model.

**Request Body:**

```json
{
  "model": "llama2-7b-q4",
  "prompt": "What is the capital of France?",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.95,
  "seed": 42
}
```

**Response:**

```json
{
  "id": "gen_1234567890",
  "model": "llama2-7b-q4",
  "choices": [
    {
      "index": 0,
      "text": "The capital of France is Paris. Paris is known as the 'City of Light' and is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 30,
    "total_tokens": 37
  },
  "created": 1678901234
}
```

### Streaming Generation

```
POST /v1/completions/stream
```

Stream text completions from a model.

**Request Body:**

```json
{
  "model": "llama2-7b-q4",
  "prompt": "Write a poem about AI",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.95,
  "seed": 42,
  "stream": true
}
```

**Response:**

Server-sent events (SSE) stream with JSON chunks:

```
data: {"id":"gen_1234567890","model":"llama2-7b-q4","choices":[{"index":0,"text":"Sil","delta":{"text":"Sil"},"finish_reason":null}],"created":1678901234}

data: {"id":"gen_1234567890","model":"llama2-7b-q4","choices":[{"index":0,"text":"icon","delta":{"text":"icon"},"finish_reason":null}],"created":1678901234}

...
```

### Model List

```
GET /v1/models
```

List available models.

**Response:**

```json
{
  "data": [
    {
      "id": "llama2-7b-q4",
      "created": 1678901234,
      "owned_by": "llamascale"
    },
    {
      "id": "mistral-7b-q4",
      "created": 1678901234,
      "owned_by": "llamascale"
    }
  ]
}
```

### Health Check

```
GET /v1/health
```

Check the health of the API server.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": 1678901234,
  "version": "1.0.0"
}
```

## API v2 Endpoints

The v2 API includes all v1 endpoints plus the following advanced features:

### Function Calling

```
POST /v2/completions
```

Generate text completions with function calling capability.

**Request Body:**

```json
{
  "model": "mistral-7b-q4",
  "prompt": "What's the weather like in San Francisco?",
  "max_tokens": 100,
  "temperature": 0.7,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get current weather in a given location",
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
  ]
}
```

**Response:**

```json
{
  "id": "gen_1234567890",
  "model": "mistral-7b-q4",
  "choices": [
    {
      "index": 0,
      "text": "I'll check the weather in San Francisco for you.",
      "finish_reason": "tool_calls"
    }
  ],
  "tool_calls": [
    {
      "id": "call_1234567890",
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "arguments": "{\"location\":\"San Francisco, CA\",\"unit\":\"celsius\"}"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 30,
    "total_tokens": 45
  },
  "created": 1678901234
}
```

### Tool Execution

```
POST /v2/tools/execute
```

Execute a tool call.

**Request Body:**

```json
{
  "id": "call_1234567890",
  "type": "function",
  "function": {
    "name": "get_current_weather",
    "arguments": "{\"location\":\"San Francisco, CA\",\"unit\":\"celsius\"}"
  }
}
```

**Response:**

```json
{
  "id": "result_call_1234567890",
  "type": "function_result",
  "function": {
    "name": "get_current_weather",
    "result": {
      "location": "San Francisco, CA",
      "temperature": 18.5,
      "unit": "celsius",
      "condition": "sunny",
      "humidity": 65,
      "wind_speed": 10.2,
      "forecast": "The weather in San Francisco, CA is sunny with a temperature of 18.5°C.",
      "timestamp": "2023-06-01T12:00:00Z"
    }
  }
}
```

### Agent Creation

```
POST /v2/agents/create
```

Create an agent with reasoning capabilities.

**Request Body:**

```json
{
  "name": "WeatherAssistant",
  "model": "mistral-7b-q4",
  "reasoning_strategy": "react",
  "tools": ["get_current_weather", "calculate"],
  "system_prompt": "You are a helpful weather assistant who provides accurate weather information."
}
```

**Response:**

```json
{
  "agent_id": "agent_1234567890",
  "name": "WeatherAssistant",
  "model": "mistral-7b-q4",
  "reasoning_strategy": "react",
  "tools": ["get_current_weather", "calculate"]
}
```

### Agent Execution

```
POST /v2/agents/{agent_id}/run
```

Run an agent on a request.

**Request Body:**

```json
{
  "prompt": "What will the temperature in celsius be in Paris tomorrow?",
  "max_steps": 5
}
```

**Response:**

```json
{
  "id": "run_1234567890",
  "model": "mistral-7b-q4",
  "final_answer": "The temperature in Paris tomorrow will be around 20°C with partly cloudy conditions.",
  "steps": [
    {
      "type": "thinking",
      "content": "I need to check the weather for Paris. I should use the get_current_weather tool."
    },
    {
      "type": "tool_use",
      "content": "I'll check the current weather in Paris.",
      "tool_calls": [
        {
          "id": "call_1234567890",
          "type": "function",
          "function": {
            "name": "get_current_weather",
            "arguments": "{\"location\":\"Paris, France\",\"unit\":\"celsius\"}"
          }
        }
      ]
    },
    {
      "type": "thinking",
      "content": "Now I have the current weather, but I need to provide information about tomorrow."
    },
    {
      "type": "final_answer",
      "content": "The temperature in Paris tomorrow will be around 20°C with partly cloudy conditions."
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 120,
    "total_tokens": 132
  },
  "created": 1678901234
}
```

## API Authentication

The API server supports API key authentication. To use authentication:

1. Create an API key file at `~/.llamascale/api_keys.json`:

```json
{
  "sk-llamascale-12345": "user1",
  "sk-llamascale-67890": "user2"
}
```

2. Include the API key in your requests:

```bash
curl -X POST http://localhost:8000/v2/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-llamascale-12345" \
  -d '{
    "model": "llama2-7b-q4",
    "prompt": "Hello, world!"
  }'
```

## Configuration

The API server can be configured using the LlamaScale configuration system:

```yaml
# ~/.llamascale/config.yaml

api_host: "127.0.0.1"  # Listen address
api_port: 8000         # Listen port

# API rate limits
rate_limits:
  requests_per_minute: 60
  tokens_per_minute: 100000

# Authentication settings
require_auth: false    # Set to true to require API keys

# Logging
log_level: "info"
``` 