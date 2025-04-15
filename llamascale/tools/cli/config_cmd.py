#!/usr/bin/env python3
"""
LlamaScale Configuration CLI Commands
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from .config import CacheConfig, ConfigManager, LlamaScaleConfig, MLXConfig, ModelConfig

logger = logging.getLogger(__name__)


def config_init(args):
    """Initialize configuration file"""
    manager = ConfigManager(args.config_path)

    # Create default config
    config = manager.load()

    # Save to file
    manager.save(config)

    print(f"Initialized configuration at {manager.config_path}")


def config_view(args):
    """View configuration"""
    manager = ConfigManager(args.config_path)
    config = manager.load()

    # Convert to dict for display
    config_dict = manager._config_to_dict(config)

    if args.format == "json":
        print(json.dumps(config_dict, indent=2))
    else:
        print("\nLlamaScale Configuration:")
        print("------------------------")
        _print_config_recursive(config_dict, indent=2)


def _print_config_recursive(config: Dict[str, Any], indent: int = 0, prefix: str = ""):
    """Print configuration recursively"""
    for key, value in config.items():
        # Form the full key path
        full_key = f"{prefix}.{key}" if prefix else key

        # Handle nested dictionaries
        if isinstance(value, dict) and value:
            print(f"{' ' * indent}{full_key}:")
            _print_config_recursive(value, indent + 2, full_key)
        # Handle lists of dictionaries
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            print(f"{' ' * indent}{full_key}: [")
            for i, item in enumerate(value):
                print(f"{' ' * (indent + 2)}Item {i}:")
                _print_config_recursive(item, indent + 4)
            print(f"{' ' * indent}]")
        # Handle basic values
        else:
            print(f"{' ' * indent}{full_key}: {value}")


def config_preset(args):
    """Apply a configuration preset"""
    manager = ConfigManager(args.config_path)

    try:
        manager.apply_preset(args.preset)
        print(f"Applied preset '{args.preset}'")
    except ValueError as e:
        print(f"Error: {e}")
        valid_presets = ["m1", "m2", "m3", "performance", "memory"]
        print(f"Valid presets: {', '.join(valid_presets)}")
        sys.exit(1)


def config_add_model(args):
    """Add a model to the configuration"""
    manager = ConfigManager(args.config_path)

    # Create model config
    model = ModelConfig(
        name=args.name,
        type=args.type,
        params=args.params,
        max_seq_len=args.max_seq_len,
        quantization=args.quantization,
        local_path=args.local_path,
        hf_repo=args.hf_repo,
    )

    # Add model to config
    manager.add_model(model)

    print(f"Added model '{args.name}' to configuration")


def config_remove_model(args):
    """Remove a model from the configuration"""
    manager = ConfigManager(args.config_path)

    # Remove model
    manager.remove_model(args.name)

    print(f"Removed model '{args.name}' from configuration")


def config_set(args):
    """Set configuration values"""
    manager = ConfigManager(args.config_path)
    config = manager.load()

    # Parse key path
    parts = args.key.split(".")

    # Set value for specific configuration sections
    if parts[0] == "mlx_config":
        if len(parts) != 2:
            print(f"Error: Invalid key path '{args.key}'")
            sys.exit(1)

        if not hasattr(config.mlx_config, parts[1]):
            print(f"Error: MLXConfig has no attribute '{parts[1]}'")
            sys.exit(1)

        value = _convert_value(args.value, getattr(config.mlx_config, parts[1]))
        setattr(config.mlx_config, parts[1], value)

    elif parts[0] == "cache_config":
        if len(parts) != 2:
            print(f"Error: Invalid key path '{args.key}'")
            sys.exit(1)

        if not hasattr(config.cache_config, parts[1]):
            print(f"Error: CacheConfig has no attribute '{parts[1]}'")
            sys.exit(1)

        value = _convert_value(args.value, getattr(config.cache_config, parts[1]))
        setattr(config.cache_config, parts[1], value)

    else:
        # Set top-level config value
        if not hasattr(config, parts[0]):
            print(f"Error: LlamaScaleConfig has no attribute '{parts[0]}'")
            sys.exit(1)

        if len(parts) != 1:
            print(f"Error: Invalid key path '{args.key}'")
            sys.exit(1)

        value = _convert_value(args.value, getattr(config, parts[0]))
        setattr(config, parts[0], value)

    # Save updated config
    manager.save(config)

    print(f"Set '{args.key}' to '{args.value}'")


def _convert_value(value_str: str, current_value: Any) -> Any:
    """Convert string value to appropriate type based on current value"""
    if isinstance(current_value, bool):
        return value_str.lower() in ["true", "yes", "y", "1"]
    elif isinstance(current_value, int):
        return int(value_str)
    elif isinstance(current_value, float):
        return float(value_str)
    elif isinstance(current_value, list):
        return json.loads(value_str)
    elif isinstance(current_value, dict):
        return json.loads(value_str)
    else:
        return value_str


def add_config_args(subparsers):
    """Add configuration command arguments to parser"""
    # Init
    parser_init = subparsers.add_parser(
        "config-init", help="Initialize configuration file"
    )
    parser_init.add_argument("--config-path", help="Path to configuration file")
    parser_init.set_defaults(func=config_init)

    # View
    parser_view = subparsers.add_parser("config-view", help="View configuration")
    parser_view.add_argument("--config-path", help="Path to configuration file")
    parser_view.add_argument(
        "--format", choices=["pretty", "json"], default="pretty", help="Output format"
    )
    parser_view.set_defaults(func=config_view)

    # Preset
    parser_preset = subparsers.add_parser(
        "config-preset", help="Apply a configuration preset"
    )
    parser_preset.add_argument(
        "preset",
        choices=["m1", "m2", "m3", "performance", "memory"],
        help="Preset to apply",
    )
    parser_preset.add_argument("--config-path", help="Path to configuration file")
    parser_preset.set_defaults(func=config_preset)

    # Set
    parser_set = subparsers.add_parser("config-set", help="Set configuration value")
    parser_set.add_argument(
        "key", help="Configuration key (e.g. 'mlx_config.thread_count')"
    )
    parser_set.add_argument("value", help="Configuration value")
    parser_set.add_argument("--config-path", help="Path to configuration file")
    parser_set.set_defaults(func=config_set)

    # Add model
    parser_add_model = subparsers.add_parser(
        "config-add-model", help="Add a model to the configuration"
    )
    parser_add_model.add_argument("name", help="Model name")
    parser_add_model.add_argument(
        "--type", default="llm", choices=["llm", "embedding"], help="Model type"
    )
    parser_add_model.add_argument(
        "--params", default="7B", help="Model parameters (e.g. '7B')"
    )
    parser_add_model.add_argument(
        "--max-seq-len", type=int, default=8192, help="Maximum sequence length"
    )
    parser_add_model.add_argument("--quantization", help="Quantization method")
    parser_add_model.add_argument("--local-path", help="Local model path")
    parser_add_model.add_argument("--hf-repo", help="Hugging Face repository")
    parser_add_model.add_argument("--config-path", help="Path to configuration file")
    parser_add_model.set_defaults(func=config_add_model)

    # Remove model
    parser_remove_model = subparsers.add_parser(
        "config-remove-model", help="Remove a model from the configuration"
    )
    parser_remove_model.add_argument("name", help="Model name")
    parser_remove_model.add_argument("--config-path", help="Path to configuration file")
    parser_remove_model.set_defaults(func=config_remove_model)
