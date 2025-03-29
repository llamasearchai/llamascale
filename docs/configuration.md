# LlamaScale Configuration System

LlamaScale provides a comprehensive configuration system to customize and optimize your LLM deployment. This guide explains how to use the configuration system effectively.

## Configuration Files

LlamaScale configuration is stored in JSON format and can be managed through the CLI or by directly editing the configuration file. By default, the configuration file is located at:

```
~/.llamascale/config.json
```

## Configuration Commands

LlamaScale provides several CLI commands to manage your configuration:

### Initialize Configuration

Create a new configuration file with default settings:

```bash
llamascale config-init
```

### View Configuration

View your current configuration:

```bash
llamascale config-view
```

For JSON output:

```bash
llamascale config-view --format json
```

### Apply Configuration Preset

Apply a pre-defined configuration preset optimized for your hardware:

```bash
llamascale config-preset m3
```

Available presets:
- `m1`: Optimized for M1 Macs
- `m2`: Optimized for M2 Macs
- `m3`: Optimized for M3 Macs
- `performance`: Optimized for maximum performance
- `memory`: Optimized for low memory usage

### Set Configuration Values

Set individual configuration values:

```bash
llamascale config-set mlx_config.quantization int4
llamascale config-set cache_config.memory_size 2000
llamascale config-set api_port 8080
```

### Manage Models

Add a model to your configuration:

```bash
llamascale config-add-model llama3-8b-instruct --hf-repo meta-llama/Meta-Llama-3-8B-Instruct --params 8B
```

Remove a model from your configuration:

```bash
llamascale config-remove-model llama3-8b-instruct
```

## Configuration Structure

The main configuration object contains the following sections:

### General Settings

- `models_dir`: Directory for storing models
- `cache_dir`: Directory for disk cache storage
- `log_dir`: Directory for log files
- `enable_mlx`: Enable MLX backend
- `enable_cuda`: Enable CUDA backend
- `api_port`: Port for the API server
- `api_host`: Host for the API server

### MLX Configuration

The `mlx_config` section contains settings for the MLX backend:

- `use_metal`: Use Metal GPU acceleration
- `quantization`: Quantization method (int8, int4, etc.)
- `max_batch_size`: Maximum batch size for inference
- `max_tokens`: Maximum tokens to generate
- `context_len`: Maximum context length
- `cpu_offload`: Offload computation to CPU when needed
- `thread_count`: Number of CPU threads (0 for automatic)

### Cache Configuration

The `cache_config` section contains settings for the caching system:

- `memory_size`: Memory cache size (number of entries)
- `disk_path`: Path to disk cache
- `redis_url`: Optional Redis URL for distributed caching
- `ttl`: Time-to-live for cache entries (seconds)
- `semantic_threshold`: Similarity threshold for semantic caching

### Models Configuration

The `models` section contains a list of available models:

- `name`: Model name
- `type`: Model type (llm, embedding)
- `params`: Model parameter size (e.g., "7B")
- `max_seq_len`: Maximum sequence length
- `quantization`: Model-specific quantization
- `local_path`: Local path to model files
- `hf_repo`: Hugging Face repository for downloading

### Routing Configuration

The `routing_weights` section controls how requests are routed to different backends:

- `latency`: Weight for latency optimization
- `cost`: Weight for cost optimization
- `reliability`: Weight for reliability optimization

## Example Configuration

```json
{
  "models_dir": "~/.llamascale/models",
  "cache_dir": "~/.llamascale/cache",
  "log_dir": "~/.llamascale/logs",
  "mlx_config": {
    "use_metal": true,
    "quantization": "int4",
    "max_batch_size": 2,
    "max_tokens": 4096,
    "context_len": 8192,
    "cpu_offload": false,
    "thread_count": 0
  },
  "cache_config": {
    "memory_size": 2000,
    "disk_path": "~/.llamascale/cache",
    "redis_url": null,
    "ttl": 300,
    "semantic_threshold": 0.95
  },
  "mlx_models": [
    "llama3-8b-instruct"
  ],
  "enable_mlx": true,
  "enable_cuda": false,
  "api_port": 8000,
  "api_host": "127.0.0.1",
  "models": [
    {
      "name": "llama3-8b-instruct",
      "type": "llm",
      "params": "8B",
      "max_seq_len": 8192,
      "quantization": "int4",
      "local_path": "~/.llamascale/models/llama3-8b-instruct",
      "hf_repo": "meta-llama/Meta-Llama-3-8B-Instruct"
    }
  ]
}
```

## Programmatic Usage

You can also access the configuration system programmatically:

```python
from llamascale.tools.cli.config import ConfigManager, ModelConfig

# Load configuration
manager = ConfigManager()
config = manager.load()

# Add a model
model = ModelConfig(
    name="llama3-8b-instruct",
    type="llm",
    params="8B",
    max_seq_len=8192,
    quantization="int4",
    local_path="~/.llamascale/models/llama3-8b-instruct",
    hf_repo="meta-llama/Meta-Llama-3-8B-Instruct"
)
manager.add_model(model)

# Apply a preset
manager.apply_preset("m3")

# Save changes
manager.save()
```

## Using Configuration with API Server

When starting the API server, the configuration is automatically loaded:

```bash
llamascale_api
```

You can also specify a custom configuration file:

```bash
LLAMASCALE_CONFIG=/path/to/config.json llamascale_api
```

## Best Practices

1. **Use Presets**: Start with a preset that matches your hardware, then customize as needed
2. **Optimize Quantization**: Use int8 for better quality or int4 for better performance
3. **Adjust Cache Size**: Increase memory cache size for frequently used prompts
4. **Set Thread Count**: Use 0 to automatically use all available cores, or set a specific number to avoid resource contention
5. **Enable Redis**: For multi-instance deployments, configure Redis for shared caching 