#!/usr/bin/env python3
"""
LlamaScale Initialization Script
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def init_llamascale_dirs():
    """
    Initialize LlamaScale directories and configuration
    """
    # Create default directories
    home_dir = os.path.expanduser("~")
    llamascale_dir = os.path.join(home_dir, ".llamascale")
    
    dirs = [
        llamascale_dir,
        os.path.join(llamascale_dir, "models"),
        os.path.join(llamascale_dir, "cache"),
        os.path.join(llamascale_dir, "logs")
    ]
    
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    
    # Create default config if it doesn't exist
    config_path = os.path.join(llamascale_dir, "config.json")
    if not os.path.exists(config_path):
        default_config = {
            "models_dir": os.path.join(llamascale_dir, "models"),
            "cache_dir": os.path.join(llamascale_dir, "cache"),
            "log_dir": os.path.join(llamascale_dir, "logs"),
            "mlx_config": {
                "use_metal": True,
                "quantization": "int8",
                "max_batch_size": 1,
                "max_tokens": 4096,
                "context_len": 8192
            },
            "cache_config": {
                "memory_size": 1000,
                "disk_path": os.path.join(llamascale_dir, "cache"),
                "ttl": 300,
                "semantic_threshold": 0.95
            },
            "mlx_models": [],
            "mlx_nodes": [
                {
                    "capacity": {
                        "max_batch_size": 1
                    },
                    "cost_per_token": 0.0
                }
            ]
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
            logger.info(f"Created default config: {config_path}")
    
    return llamascale_dir

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_llamascale_dirs()
    print("LlamaScale directories and configuration initialized successfully.") 