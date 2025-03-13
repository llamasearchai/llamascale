# 🦙 LlamaScale Ultra 6.0

**Mac-Native Enterprise LLM Orchestration Platform**  
`v6.0.0 | Apple Silicon | FastAPI 1.0 | MLX 2.0 | vLLM 0.5 | Kubernetes-Native`

LlamaScale is a high-performance LLM orchestration platform optimized for Apple Silicon, providing enterprise-grade features for deploying, scaling, and managing large language models.

## 🌟 Key Features

- **Mac-Native Performance**: Optimized for Apple Silicon with MLX 2.0 integration
- **Intelligent Load Balancing**: Smart request routing with cost optimization
- **Multi-Level Caching**: Memory, Redis, and disk caching with semantic similarity
- **Dynamic Scaling**: Kubernetes-native autoscaling for cloud deployments
- **Comprehensive Monitoring**: Prometheus, Grafana, and OpenTelemetry integration
- **Function Calling**: Advanced function calling and tool usage capabilities
- **Agent Framework**: ReAct, Chain-of-Thought, and other reasoning strategies for complex tasks
- **Cost Optimization**: Token-level cost tracking and optimization
- **Advanced Configuration**: Flexible configuration system with hardware-optimized presets
- **Multimodal Capabilities**: Process and analyze images alongside text with MLX Vision
- **Interactive Llama Experience**: Beautiful llama-themed CLI with colorful output and fun llama facts
- **Full FastAPI Integration**: Complete REST API with both synchronous and asynchronous endpoints

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install llamascale

# Install with optional components
pip install llamascale[mlx,redis,monitoring,agents]

# Install from source
git clone https://github.com/llamascale/llamascale.git
cd llamascale
pip install -e .
```

### Basic Usage

1. **Initialize Configuration**

```bash
# Create default configuration
llamascale config-init

# Apply a hardware-optimized preset (m1, m2, m3, performance, memory)
llamascale config-preset m3
```

2. **Download a Model**

```bash
# Download a model (replace MODEL_NAME with actual model)
llamascale download --model MODEL_NAME
```

3. **Generate Text**

```bash
# Generate text from the model
llamascale generate --model MODEL_NAME --prompt "Hello, I am a large language model."
```

4. **Start the API Server**

```bash
# Start the basic API server (v1)
llamascale_api

# Start the advanced API with function calling and agents (v2)
llamascale_api --v2
```

## 💻 Command Line Interface

LlamaScale provides a comprehensive command-line interface:

```
Usage: llamascale [OPTIONS] COMMAND [ARGS]...

Commands:
  generate            Generate text from LLM
  list                List available models
  download            Download LLM model
  cache               Cache management commands
  benchmark           Benchmark model performance
  config-init         Initialize configuration file
  config-view         View configuration
  config-preset       Apply a configuration preset
  config-set          Set configuration value
  config-add-model    Add a model to the configuration
  config-remove-model Remove a model from the configuration
  version             Show version information
```

### Generate Text

```bash
llamascale generate --model mistral-7b --prompt "Explain quantum computing" --max-tokens 1024
```

Options:
- `--model, -m`: Model to use
- `--prompt, -p`: Input prompt
- `--max-tokens`: Maximum tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Top-p sampling parameter (default: 0.9)
- `--seed`: Random seed for reproducibility
- `--output, -o`: Output file path (stdout if not specified)
- `--stream`: Stream output tokens

### List Models

```bash
# List available models
llamascale list

# Show detailed model information
llamascale list --detailed
```

### Download Models

```bash
# Download a model
llamascale download --model mistral-7b

# Force redownload if model exists
llamascale download --model mistral-7b --force
```

### Cache Management

```bash
# Show cache statistics
llamascale cache stats

# Clear all caches
llamascale cache clear --all

# Clear only memory cache
llamascale cache clear --memory

# Clear only disk cache
llamascale cache clear --disk
```

### Configuration Management

```bash
# Initialize configuration
llamascale config-init

# View configuration
llamascale config-view
llamascale config-view --format json

# Apply a hardware-optimized preset
llamascale config-preset m3

# Set configuration values
llamascale config-set mlx_config.quantization int4
llamascale config-set cache_config.memory_size 2000
llamascale config-set api_port 8080

# Add a model to configuration
llamascale config-add-model llama3-8b-instruct --hf-repo meta-llama/Meta-Llama-3-8B-Instruct --params 8B

# Remove a model
llamascale config-remove-model llama3-8b-instruct
```

### Benchmark Models

```bash
# Benchmark model performance
llamascale benchmark --model mistral-7b --tokens 1024 --runs 3
```

## 🏗️ Architecture

LlamaScale follows a modular architecture:

```
llamascale/
├── orchestrator/            # Core Orchestration Engine
│   ├── routing/             # Intelligent Load Balancing
│   ├── caching/             # Multi-Level Caching
│   └── autoscaler/          # Dynamic Resource Scaling
├── api/                     # API Interfaces
│   ├── v1/                  # Basic API
│   └── v2/                  # Advanced API with Function Calling
├── agents/                  # Agent Framework
│   └── framework.py         # Advanced reasoning capabilities 
├── drivers/                 # Model Backends
│   ├── mlx/                 # Apple Silicon Support
│   ├── cuda/                # NVIDIA GPU Support
│   └── multimodal/          # Multimodal Support
├── monitoring/              # Observability
└── tools/                   # Developer Tooling
    ├── cli/                 # Command Line Interface
    └── benchmarks/          # Benchmarking Suite
```

## 🔧 Configuration System

LlamaScale provides a comprehensive configuration system for customizing and optimizing your LLM deployment.

### Configuration Storage

Configuration is stored in JSON format at `~/.llamascale/config.json` and can be managed through the CLI or by directly editing the file.

### Hardware-Optimized Presets

LlamaScale includes presets optimized for different hardware:

- `m1`: Optimized for M1 Macs
- `m2`: Optimized for M2 Macs
- `m3`: Optimized for M3 Macs
- `performance`: Optimized for maximum performance
- `memory`: Optimized for low memory usage

### Configuration Sections

The configuration includes multiple sections:

- **General Settings**: Directories, ports, and feature flags
- **MLX Configuration**: Metal GPU settings, quantization, threading
- **Cache Configuration**: Memory/disk cache settings, semantic threshold
- **Models Configuration**: Model-specific settings and locations
- **Routing Configuration**: Load balancing and smart routing parameters

## 🤖 Function Calling and Agent Framework

LlamaScale v6.0 introduces advanced function calling and agent capabilities for building sophisticated AI systems.

### Function Calling

Function calling allows models to invoke functions (tools) to accomplish specific tasks:

```python
from llamascale.api.v2.client import LlamaScaleClient

# Initialize client
client = LlamaScaleClient()

# Define a tool
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}

# Generate text with function calling
response = client.generate(
    model="llama3-70b-instruct-q4",
    prompt="What's the weather like in San Francisco?",
    tools=[weather_tool]
)

# Execute tool call and continue conversation
if response.tool_calls:
    tool_result = client.execute_tool_call(response.tool_calls[0])
    
    # Continue conversation with tool result
    final_response = client.generate(
        model="llama3-70b-instruct-q4",
        prompt=f"User asked about the weather in San Francisco.\nTool result: {tool_result}"
    )
```

### Agent Framework

The agent framework allows models to break down complex tasks through multi-step reasoning:

```python
from llamascale.agents import Agent, ReasoningStrategy

# Create an agent
agent = Agent(
    name="WeatherExpert",
    model_name="llama3-70b-instruct-q4",
    reasoning_strategy=ReasoningStrategy.REACT,
    tools=[weather_tool, calculator_tool]
)

# Run the agent
result = await agent.run(
    "What's the average temperature between Tokyo and Paris, and what's the square root of that value?"
)

print(f"Answer: {result['answer']}")
print(f"Reasoning steps: {len(result['reasoning_trace'])} steps taken")
```

### Available Reasoning Strategies

LlamaScale supports multiple reasoning strategies:

- **REACT**: Reasoning and Acting - observe, think, act cycle
- **CHAIN_OF_THOUGHT**: Step-by-step reasoning
- **TREE_OF_THOUGHT**: Exploring multiple reasoning paths
- **REFLECTION**: Self-critique and refine answers
- **VERIFICATION**: Verify answers with additional checks

## 📡 API Server

LlamaScale includes both a basic API (v1) and an advanced API (v2) with function calling and agent support:

```bash
# Start basic API server
llamascale_api

# Start advanced API server
llamascale_api --v2
```

See the [API documentation](./llamascale/api/README.md) for details on endpoints and request formats.

## 📄 License

LlamaScale is released under the MIT License. See the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
# Updated in commit 1 - 2025-04-04 17:32:29

# Updated in commit 9 - 2025-04-04 17:32:30

# Updated in commit 17 - 2025-04-04 17:32:30

# Updated in commit 25 - 2025-04-04 17:32:31

# Updated in commit 1 - 2025-04-05 14:36:07

# Updated in commit 9 - 2025-04-05 14:36:07

# Updated in commit 17 - 2025-04-05 14:36:08

# Updated in commit 25 - 2025-04-05 14:36:08

# Updated in commit 1 - 2025-04-05 15:22:37

# Updated in commit 9 - 2025-04-05 15:22:37

# Updated in commit 17 - 2025-04-05 15:22:37

# Updated in commit 25 - 2025-04-05 15:22:38

# Updated in commit 1 - 2025-04-05 15:56:57

# Updated in commit 9 - 2025-04-05 15:56:57

# Updated in commit 17 - 2025-04-05 15:56:57

# Updated in commit 25 - 2025-04-05 15:56:57

# Updated in commit 1 - 2025-04-05 17:02:22

# Updated in commit 9 - 2025-04-05 17:02:22

# Updated in commit 17 - 2025-04-05 17:02:22

# Updated in commit 25 - 2025-04-05 17:02:23

# Updated in commit 1 - 2025-04-05 17:34:23

# Updated in commit 9 - 2025-04-05 17:34:23

# Updated in commit 17 - 2025-04-05 17:34:24

# Updated in commit 25 - 2025-04-05 17:34:24

# Updated in commit 1 - 2025-04-05 18:21:06

# Updated in commit 9 - 2025-04-05 18:21:06

# Updated in commit 17 - 2025-04-05 18:21:06

# Updated in commit 25 - 2025-04-05 18:21:06
