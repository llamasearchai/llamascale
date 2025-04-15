#!/usr/bin/env python3
"""
LlamaScale Configuration Manager
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration"""

    name: str
    type: str = "llm"
    params: str = "7B"
    max_seq_len: int = 8192
    quantization: Optional[str] = None
    local_path: Optional[str] = None
    hf_repo: Optional[str] = None


@dataclass
class MLXConfig:
    """MLX configuration"""

    use_metal: bool = True
    quantization: Optional[str] = None
    max_batch_size: int = 1
    max_tokens: int = 4096
    context_len: int = 8192
    cpu_offload: bool = False
    thread_count: int = 0
    kv_cache_config: Optional[Dict[str, Any]] = None


@dataclass
class CacheConfig:
    """Cache configuration"""

    memory_size: int = 1000
    disk_path: Optional[str] = None
    redis_url: Optional[str] = None
    ttl: int = 300
    semantic_threshold: float = 0.95


@dataclass
class NodeConfig:
    """Backend node configuration"""

    backend_type: str = "mlx"
    capacity: Dict[str, Any] = field(default_factory=lambda: {"max_batch_size": 1})
    cost_per_token: float = 0.0


@dataclass
class RoutingConfig:
    """Routing configuration"""

    weights: Dict[str, float] = field(
        default_factory=lambda: {"latency": 0.6, "cost": 0.3, "reliability": 0.1}
    )


@dataclass
class LlamaScaleConfig:
    """LlamaScale configuration"""

    models_dir: str = "~/.llamascale/models"
    cache_dir: str = "~/.llamascale/cache"
    log_dir: str = "~/.llamascale/logs"
    mlx_config: MLXConfig = field(default_factory=MLXConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    mlx_models: List[str] = field(default_factory=list)
    cuda_models: List[str] = field(default_factory=list)
    mlx_nodes: List[NodeConfig] = field(default_factory=lambda: [NodeConfig()])
    cuda_nodes: List[NodeConfig] = field(default_factory=list)
    routing_weights: Dict[str, float] = field(
        default_factory=lambda: {"latency": 0.6, "cost": 0.3, "reliability": 0.1}
    )
    enable_mlx: bool = True
    enable_cuda: bool = False
    api_port: int = 8000
    api_host: str = "127.0.0.1"
    models: List[ModelConfig] = field(default_factory=list)


class ConfigManager:
    """LlamaScale configuration manager"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager

        Args:
            config_path: Path to configuration file (default: ~/.llamascale/config.json)
        """
        self.home_dir = os.path.expanduser("~")
        self.llamascale_dir = os.path.join(self.home_dir, ".llamascale")

        if config_path:
            self.config_path = config_path
        else:
            self.config_path = os.path.join(self.llamascale_dir, "config.json")

        self.default_config = self._create_default_config()
        self.config = self.load()

    def _create_default_config(self) -> LlamaScaleConfig:
        """Create default configuration"""
        default_config = LlamaScaleConfig(
            models_dir=os.path.join(self.llamascale_dir, "models"),
            cache_dir=os.path.join(self.llamascale_dir, "cache"),
            log_dir=os.path.join(self.llamascale_dir, "logs"),
            mlx_config=MLXConfig(),
            cache_config=CacheConfig(
                disk_path=os.path.join(self.llamascale_dir, "cache")
            ),
        )

        return default_config

    def load(self) -> LlamaScaleConfig:
        """Load configuration from file"""
        # Start with default config
        config = self.default_config

        # Try to load from file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config_dict = json.load(f)

                # Update default config with loaded values
                self._update_config_from_dict(config, config_dict)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Could not load configuration: {e}")

        return config

    def _update_config_from_dict(
        self, config: LlamaScaleConfig, config_dict: Dict[str, Any]
    ):
        """Update configuration from dictionary"""
        # Update top-level fields
        for key, value in config_dict.items():
            if key == "mlx_config" and value:
                self._update_mlx_config(config.mlx_config, value)
            elif key == "cache_config" and value:
                self._update_cache_config(config.cache_config, value)
            elif key == "mlx_nodes" and value:
                config.mlx_nodes = []
                for node_dict in value:
                    node = NodeConfig()
                    for k, v in node_dict.items():
                        setattr(node, k, v)
                    config.mlx_nodes.append(node)
            elif key == "cuda_nodes" and value:
                config.cuda_nodes = []
                for node_dict in value:
                    node = NodeConfig(backend_type="cuda")
                    for k, v in node_dict.items():
                        setattr(node, k, v)
                    config.cuda_nodes.append(node)
            elif key == "models" and value:
                config.models = []
                for model_dict in value:
                    model = ModelConfig(name=model_dict["name"])
                    for k, v in model_dict.items():
                        setattr(model, k, v)
                    config.models.append(model)
            elif hasattr(config, key):
                setattr(config, key, value)

    def _update_mlx_config(self, mlx_config: MLXConfig, config_dict: Dict[str, Any]):
        """Update MLX configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(mlx_config, key):
                setattr(mlx_config, key, value)

    def _update_cache_config(
        self, cache_config: CacheConfig, config_dict: Dict[str, Any]
    ):
        """Update cache configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(cache_config, key):
                setattr(cache_config, key, value)

    def save(self, config: LlamaScaleConfig = None):
        """Save configuration to file"""
        if config is None:
            config = self.config

        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        # Convert dataclasses to dictionaries
        config_dict = self._config_to_dict(config)

        # Save to file
        with open(self.config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Saved configuration to {self.config_path}")

    def _config_to_dict(self, config: LlamaScaleConfig) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {}

        # Convert main config
        for key, value in asdict(config).items():
            if key == "mlx_config":
                config_dict[key] = asdict(config.mlx_config)
            elif key == "cache_config":
                config_dict[key] = asdict(config.cache_config)
            elif key == "mlx_nodes":
                config_dict[key] = [asdict(node) for node in config.mlx_nodes]
            elif key == "cuda_nodes":
                config_dict[key] = [asdict(node) for node in config.cuda_nodes]
            elif key == "models":
                config_dict[key] = [asdict(model) for model in config.models]
            else:
                config_dict[key] = value

        return config_dict

    def apply_preset(self, preset_name: str):
        """Apply a configuration preset"""
        presets = {
            "m1": self._preset_m1,
            "m2": self._preset_m2,
            "m3": self._preset_m3,
            "performance": self._preset_performance,
            "memory": self._preset_memory,
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}")

        # Apply preset
        presets[preset_name](self.config)

        # Save updated config
        self.save()

    def _preset_m1(self, config: LlamaScaleConfig):
        """Preset optimized for M1 Macs"""
        config.mlx_config.quantization = "int8"
        config.mlx_config.max_batch_size = 1
        config.mlx_config.thread_count = 0  # Use all cores

    def _preset_m2(self, config: LlamaScaleConfig):
        """Preset optimized for M2 Macs"""
        config.mlx_config.quantization = "int8"
        config.mlx_config.max_batch_size = 1
        config.mlx_config.thread_count = 0  # Use all cores

    def _preset_m3(self, config: LlamaScaleConfig):
        """Preset optimized for M3 Macs"""
        config.mlx_config.quantization = "int4"
        config.mlx_config.max_batch_size = 2
        config.mlx_config.thread_count = 0  # Use all cores

    def _preset_performance(self, config: LlamaScaleConfig):
        """Preset optimized for performance"""
        config.mlx_config.quantization = "int4"
        config.mlx_config.max_batch_size = 2
        config.mlx_config.thread_count = 0  # Use all cores
        config.cache_config.memory_size = 2000
        config.routing_weights = {"latency": 0.9, "cost": 0.05, "reliability": 0.05}

    def _preset_memory(self, config: LlamaScaleConfig):
        """Preset optimized for low memory usage"""
        config.mlx_config.quantization = "int4"
        config.mlx_config.max_batch_size = 1
        config.mlx_config.cpu_offload = True
        config.cache_config.memory_size = 500

    def add_model(self, model_config: ModelConfig):
        """Add or update a model configuration"""
        # Check if model already exists
        for i, model in enumerate(self.config.models):
            if model.name == model_config.name:
                # Update existing model
                self.config.models[i] = model_config
                self.save()
                return

        # Add new model
        self.config.models.append(model_config)

        # Add to appropriate model list
        if model_config.type == "llm":
            if model_config.name not in self.config.mlx_models:
                self.config.mlx_models.append(model_config.name)

        self.save()

    def remove_model(self, model_name: str):
        """Remove a model configuration"""
        # Remove from models list
        self.config.models = [m for m in self.config.models if m.name != model_name]

        # Remove from mlx_models list
        if model_name in self.config.mlx_models:
            self.config.mlx_models.remove(model_name)

        # Remove from cuda_models list
        if model_name in self.config.cuda_models:
            self.config.cuda_models.remove(model_name)

        self.save()


def get_config() -> LlamaScaleConfig:
    """Get LlamaScale configuration"""
    manager = ConfigManager()
    return manager.load()
