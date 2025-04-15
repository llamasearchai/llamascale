#!/usr/bin/env python3
"""
LlamaScale CLI - Command Line Interface for LlamaScale Ultra 6.0
Mac-Native Enterprise LLM Orchestration Platform
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pkg_resources

# Add the parent directory to the path so we can import our modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.append(parent_dir)

# Import rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("For a better experience, install rich: pip install rich")

from llamascale.drivers.mlx.engine import MLXConfig, MLXModelEngine
from llamascale.orchestrator.caching.hybrid import HybridCache
from llamascale.orchestrator.routing.smart import InferenceRouter
from llamascale.tools.cli.config import ConfigManager, get_config
from llamascale.tools.cli.config_cmd import add_config_args
from llamascale.tools.cli.init import init_llamascale_dirs

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("llamascale")

# Improved ASCII Art Logo
LOGO = r"""
                    /\             
                   /  \    
                  /    \      [bold magenta]🦙 LlamaScale Ultra 6.0[/bold magenta]
                 /      \     [italic cyan]Mac-Native Enterprise LLM Orchestration[/italic cyan]
                /        \    
               /    /\    \   
              /    /  \    \  
             /    /    \    \ 
            /____/      \____\

    [bold yellow]L[/bold yellow][bold green]l[/bold green][bold blue]a[/bold blue][bold purple]m[/bold purple][bold red]a[/bold red][bold yellow]S[/bold yellow][bold green]c[/bold green][bold blue]a[/bold blue][bold purple]l[/bold purple][bold red]e[/bold red]  [bold yellow]U[/bold yellow][bold green]l[/bold green][bold blue]t[/bold blue][bold purple]r[/bold purple][bold red]a[/bold red]  [bold yellow]6[/bold yellow][bold green].[/bold green][bold blue]0[/bold blue]
"""

# Fun llama facts for a more engaging experience
LLAMA_FACTS = [
    "Llamas can grow up to 6 feet tall and weigh up to 450 pounds!",
    "Llamas are social animals and live in herds.",
    "Llamas communicate by humming, just like your ML models!",
    "Llamas can spit up to 10 feet when they're angry or threatened.",
    "Llamas have been used as pack animals in South America for over 4,000 years.",
    "Llamas have three stomach compartments to help them digest tough plants.",
    "Llama wool is water-repellent and hypoallergenic.",
    "Llamas can carry 25-30% of their body weight for 8-13 miles.",
    "The ancient Incas called llamas 'the silent brothers'.",
    "Llamas can live up to 20 years, almost as long as your cached models!",
    "Llamas are related to camels, but don't have humps.",
    "Baby llamas are called 'crias' and can walk within an hour of birth.",
    "Llamas are eco-friendly and have soft padded feet that don't damage terrain.",
    "Llamas are sometimes used as guard animals for sheep flocks.",
    "Llamas don't bite, they just spit when upset - unlike some ML libraries!",
]

# Create a console for rich output if available
console = Console() if HAS_RICH else None


class LlamaScaleCLI:
    """Command Line Interface for LlamaScale Ultra 6.0"""

    def __init__(self):
        """Initialize the CLI with configuration"""
        # Ensure directories exist
        init_llamascale_dirs()

        self.config = self._load_config()
        self.args = None
        self.router = None
        self.cache = None
        self.show_llama_facts = True
        self.model_engines = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from default locations"""
        # Try to load config using the new ConfigManager
        try:
            config_manager = ConfigManager()
            llamascale_config = config_manager.load()

            # Convert to dictionary for backward compatibility
            config_dict = config_manager._config_to_dict(llamascale_config)
            logger.info("Loaded configuration using ConfigManager")
            return config_dict

        except Exception as e:
            logger.warning(f"Could not load configuration using ConfigManager: {e}")

            # Fall back to the old method
            # Default config
            default_config = {
                "models_dir": os.path.expanduser("~/.llamascale/models"),
                "cache_dir": os.path.expanduser("~/.llamascale/cache"),
                "log_dir": os.path.expanduser("~/.llamascale/logs"),
                "mlx_config": {
                    "use_metal": True,
                    "quantization": "int8",
                    "max_batch_size": 1,
                    "max_tokens": 4096,
                    "context_len": 8192,
                },
                "cache_config": {
                    "memory_size": 1000,
                    "disk_path": os.path.expanduser("~/.llamascale/cache"),
                    "ttl": 300,
                    "semantic_threshold": 0.95,
                },
            }

            # Try to load from user config
            config_path = os.path.expanduser("~/.llamascale/config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        user_config = json.load(f)
                        default_config.update(user_config)
                        logger.info(f"Loaded configuration from {config_path}")
                except Exception as e:
                    logger.warning(f"Could not load user config: {e}")

            return default_config

    def _ensure_directories(self):
        """Ensure all required directories exist"""
        dirs = [
            self.config["models_dir"],
            self.config["cache_dir"],
            self.config["log_dir"],
        ]

        for directory in dirs:
            os.makedirs(directory, exist_ok=True)

    def _setup_parser(self) -> argparse.ArgumentParser:
        """Set up command line argument parser"""
        parser = argparse.ArgumentParser(
            description="LlamaScale Ultra 6.0 - Mac-Native Enterprise LLM Orchestration Platform",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        subparsers = parser.add_subparsers(dest="command", help="Command to execute")

        # Generate command
        generate_parser = subparsers.add_parser(
            "generate", help="Generate text from LLM"
        )
        generate_parser.add_argument(
            "--model", "-m", required=True, help="Model to use"
        )
        generate_parser.add_argument(
            "--prompt", "-p", required=True, help="Input prompt"
        )
        generate_parser.add_argument(
            "--max-tokens", type=int, default=512, help="Maximum tokens to generate"
        )
        generate_parser.add_argument(
            "--temperature", type=float, default=0.7, help="Sampling temperature"
        )
        generate_parser.add_argument(
            "--top-p", type=float, default=0.9, help="Top-p sampling parameter"
        )
        generate_parser.add_argument(
            "--seed", type=int, help="Random seed for reproducibility"
        )
        generate_parser.add_argument(
            "--output", "-o", help="Output file path (stdout if not specified)"
        )
        generate_parser.add_argument(
            "--stream", action="store_true", help="Stream output tokens"
        )

        # List models command
        list_parser = subparsers.add_parser("list", help="List available models")
        list_parser.add_argument(
            "--detailed",
            "-d",
            action="store_true",
            help="Show detailed model information",
        )

        # Download model command
        download_parser = subparsers.add_parser("download", help="Download LLM model")
        download_parser.add_argument(
            "--model", "-m", required=True, help="Model to download"
        )
        download_parser.add_argument(
            "--force",
            "-f",
            action="store_true",
            help="Force redownload if model exists",
        )

        # Cache command
        cache_parser = subparsers.add_parser("cache", help="Cache management")
        cache_subparsers = cache_parser.add_subparsers(
            dest="cache_command", help="Cache command"
        )

        cache_clear_parser = cache_subparsers.add_parser("clear", help="Clear cache")
        cache_clear_parser.add_argument(
            "--all", "-a", action="store_true", help="Clear all caches"
        )
        cache_clear_parser.add_argument(
            "--memory", action="store_true", help="Clear memory cache"
        )
        cache_clear_parser.add_argument(
            "--disk", action="store_true", help="Clear disk cache"
        )

        cache_stats_parser = cache_subparsers.add_parser(
            "stats", help="Show cache statistics"
        )

        # Benchmark command
        benchmark_parser = subparsers.add_parser(
            "benchmark", help="Benchmark model performance"
        )
        benchmark_parser.add_argument(
            "--model", "-m", required=True, help="Model to benchmark"
        )
        benchmark_parser.add_argument(
            "--tokens",
            "-t",
            type=int,
            default=1024,
            help="Number of tokens to generate",
        )
        benchmark_parser.add_argument(
            "--runs", "-r", type=int, default=3, help="Number of benchmark runs"
        )

        # Init command
        init_parser = subparsers.add_parser(
            "init", help="Initialize LlamaScale configuration"
        )

        # Version command
        subparsers.add_parser("version", help="Show version information")

        # Add configuration commands
        add_config_args(subparsers)

        return parser

    def _init_cache(self):
        """Initialize cache system"""
        from llamascale.orchestrator.caching.hybrid import HybridCache

        self.cache = HybridCache(self.config["cache_config"])
        logger.info("Cache system initialized")

    async def _init_model(self, model_name: str) -> MLXModelEngine:
        """Initialize model engine for the specified model"""
        if model_name in self.model_engines:
            return self.model_engines[model_name]

        model_path = os.path.join(self.config["models_dir"], model_name)
        if not os.path.exists(model_path):
            raise ValueError(f"Model {model_name} not found at {model_path}")

        logger.info(f"Initializing model engine for {model_name}")
        config = MLXConfig(**self.config["mlx_config"])
        engine = MLXModelEngine(model_path, config)
        self.model_engines[model_name] = engine

        return engine

    async def _handle_generate(self):
        """Handle generate command"""
        model_name = self.args.model
        prompt = self.args.prompt

        try:
            # Initialize model
            engine = await self._init_model(model_name)

            # Check cache first
            cache_key = f"{model_name}:{prompt}:{self.args.max_tokens}:{self.args.temperature}:{self.args.top_p}:{self.args.seed}"
            cached_result = None
            if self.cache:
                cached_result = await self.cache.get(cache_key, model_name)

            if cached_result:
                result = cached_result
                logger.info("Result retrieved from cache")
                print("Result from cache:", file=sys.stderr)
            else:
                # Generate text
                logger.info(f"Generating text with model {model_name}")
                result = await engine.generate(
                    prompt=prompt,
                    max_tokens=self.args.max_tokens,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    seed=self.args.seed,
                )

                # Cache result
                if self.cache:
                    await self.cache.set(cache_key, result, model_name)
                    logger.info("Result cached")

            # Output result
            if self.args.output:
                with open(self.args.output, "w") as f:
                    if self.args.stream:
                        f.write(result["text"])
                    else:
                        json.dump(result, f, indent=2)
                logger.info(f"Output written to {self.args.output}")
            else:
                if self.args.stream:
                    print(result["text"])
                else:
                    print(json.dumps(result, indent=2))

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            print(f"Error generating text: {e}", file=sys.stderr)
            return 1

        return 0

    def _handle_list(self):
        """Handle list models command"""
        models_dir = self.config["models_dir"]

        if not os.path.exists(models_dir):
            logger.error(f"Models directory {models_dir} does not exist")
            print(f"Models directory {models_dir} does not exist", file=sys.stderr)
            return 1

        models = []
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item)
            if os.path.isdir(model_path):
                models.append(item)

        if not models:
            print("No models found")
            return 0

        logger.info(f"Found {len(models)} models")
        print(f"Available models in {models_dir}:")
        for model in sorted(models):
            if self.args.detailed:
                # Get detailed model information
                model_path = os.path.join(models_dir, model)
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r") as f:
                            config = json.load(f)
                        print(f"  - {model}:")
                        print(f"      Type: {config.get('model_type', 'unknown')}")
                        print(f"      Parameters: {config.get('params', 'unknown')}")
                        print(
                            f"      Context Length: {config.get('max_seq_len', 'unknown')}"
                        )
                    except Exception as e:
                        print(f"  - {model} (Error reading config: {e})")
                else:
                    print(f"  - {model} (No config found)")
            else:
                print(f"  - {model}")

        return 0

    def _handle_download(self):
        """Handle download model command"""
        model_name = self.args.model
        model_path = os.path.join(self.config["models_dir"], model_name)

        if os.path.exists(model_path) and not self.args.force:
            logger.warning(f"Model {model_name} already exists")
            print(
                f"Model {model_name} already exists. Use --force to redownload.",
                file=sys.stderr,
            )
            return 1

        logger.info(f"Downloading model {model_name}")
        print(f"Downloading model {model_name}...")
        # Implement model downloading logic here
        # For now, just create a placeholder directory
        os.makedirs(model_path, exist_ok=True)

        # Create a simple config file
        config = {"model_type": "llm", "params": "7B", "max_seq_len": 8192}

        with open(os.path.join(model_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model {model_name} downloaded to {model_path}")
        print(f"Model {model_name} downloaded to {model_path}")
        return 0

    async def _handle_cache_clear(self):
        """Handle cache clear command"""
        if not self.cache:
            self._init_cache()

        if self.args.all or self.args.memory:
            logger.info("Clearing memory cache")
            print("Clearing memory cache...")
            await self.cache.clear("memory")

        if self.args.all or self.args.disk:
            logger.info("Clearing disk cache")
            print("Clearing disk cache...")
            await self.cache.clear("disk")

        return 0

    async def _handle_cache_stats(self):
        """Handle cache stats command"""
        if not self.cache:
            self._init_cache()

        logger.info("Retrieving cache statistics")
        print("Cache Statistics:")
        stats = await self.cache.get_stats()
        for key, value in stats.items():
            print(f"  - {key}: {value}")

        return 0

    def _handle_init(self):
        """Handle init command"""
        llamascale_dir = init_llamascale_dirs()
        print(f"LlamaScale initialized at: {llamascale_dir}")
        return 0

    async def _handle_benchmark(self):
        """Handle benchmark command"""
        model_name = self.args.model
        tokens = self.args.tokens
        runs = self.args.runs

        try:
            logger.info(f"Benchmarking model {model_name}")
            engine = await self._init_model(model_name)

            print(f"Benchmarking {model_name} for {tokens} tokens, {runs} runs:")

            total_time = 0
            total_tokens = 0

            for i in range(runs):
                print(f"Run {i+1}/{runs}...")

                # Generate text
                start_time = asyncio.get_event_loop().time()
                result = await engine.generate(
                    prompt="This is a benchmark test of the LlamaScale system.",
                    max_tokens=tokens,
                    temperature=0.7,
                    top_p=0.9,
                )
                end_time = asyncio.get_event_loop().time()

                run_time = end_time - start_time
                gen_tokens = result["usage"]["completion_tokens"]
                tokens_per_second = gen_tokens / run_time

                print(
                    f"  - Time: {run_time:.2f}s, Tokens: {gen_tokens}, Tokens/sec: {tokens_per_second:.2f}"
                )

                total_time += run_time
                total_tokens += gen_tokens

            avg_time = total_time / runs
            avg_tokens = total_tokens / runs
            avg_tokens_per_second = avg_tokens / avg_time

            print("\nBenchmark Results:")
            print(f"  - Average time: {avg_time:.2f}s")
            print(f"  - Average tokens: {avg_tokens:.1f}")
            print(f"  - Average tokens/sec: {avg_tokens_per_second:.2f}")

            logger.info(f"Benchmark complete: {avg_tokens_per_second:.2f} tokens/sec")

        except Exception as e:
            logger.error(f"Error during benchmark: {e}")
            print(f"Error during benchmark: {e}", file=sys.stderr)
            return 1

        return 0

    def _handle_version(self):
        """Handle version command"""
        from llamascale import __version__

        print("LlamaScale Ultra 6.0")
        print("Mac-Native Enterprise LLM Orchestration Platform")
        print(f"Version: {__version__}")
        print(f"Python: {sys.version}")

        try:
            import mlx.core

            print(f"MLX: {mlx.core.__version__}")
        except ImportError:
            print("MLX: Not installed")

        return 0

    async def run(self) -> int:
        """Run the CLI with the given arguments"""
        self._print_logo()

        # Parse arguments
        parser = self._setup_parser()
        self.args = parser.parse_args()

        if not self.args.command:
            parser.print_help()
            return 1

        # Ensure directories exist
        self._ensure_directories()

        # Initialize components based on command
        if self.args.command in ["generate", "benchmark"]:
            # Initialize cache
            self._init_cache()

        # Handle commands
        if self.args.command == "generate":
            return await self._handle_generate()
        elif self.args.command == "list":
            return self._handle_list()
        elif self.args.command == "download":
            return self._handle_download()
        elif self.args.command == "cache":
            if self.args.cache_command == "clear":
                return await self._handle_cache_clear()
            elif self.args.cache_command == "stats":
                return await self._handle_cache_stats()
            else:
                logger.error("Unknown cache command")
                print("Unknown cache command", file=sys.stderr)
                return 1
        elif self.args.command == "benchmark":
            return await self._handle_benchmark()
        elif self.args.command == "init":
            return self._handle_init()
        elif self.args.command == "version":
            return self._handle_version()
        # Handle configuration commands
        elif self.args.command in [
            "config-init",
            "config-view",
            "config-preset",
            "config-set",
            "config-add-model",
            "config-remove-model",
        ]:
            # These commands use the function directly from the arguments
            return self.args.func(self.args)
        else:
            logger.error(f"Unknown command: {self.args.command}")
            print("Command required. Use --help for more information.", file=sys.stderr)
            return 1

    def _print_logo(self):
        """Print the LlamaScale logo"""
        if HAS_RICH:
            console.print(LOGO)
        else:
            print("\n🦙 LlamaScale Ultra 6.0")
            print("Mac-Native Enterprise LLM Orchestration Platform")
            print("------------------------------------------")

    def _print_llama_fact(self):
        """Print a random llama fact"""
        if not self.show_llama_facts:
            return

        fact = random.choice(LLAMA_FACTS)
        if HAS_RICH:
            console.print(
                Panel(
                    f"[yellow]🦙 Llama Fact:[/yellow] [cyan]{fact}[/cyan]",
                    border_style="green",
                    expand=False,
                )
            )
        else:
            print(f"\n🦙 Llama Fact: {fact}\n")

    def _print_success(self, message: str):
        """Print a success message"""
        if HAS_RICH:
            console.print(f"[bold green]✓[/bold green] {message}")
        else:
            print(f"✓ {message}")

    def _print_error(self, message: str):
        """Print an error message"""
        if HAS_RICH:
            console.print(f"[bold red]✗[/bold red] {message}")
        else:
            print(f"✗ {message}")

    def _print_warning(self, message: str):
        """Print a warning message"""
        if HAS_RICH:
            console.print(f"[bold yellow]![/bold yellow] {message}")
        else:
            print(f"! {message}")

    def _print_info(self, message: str):
        """Print an info message"""
        if HAS_RICH:
            console.print(f"[bold blue]i[/bold blue] {message}")
        else:
            print(f"i {message}")

    def _print_model_table(self, models: List[Dict[str, Any]]):
        """Print a table of models"""
        if HAS_RICH:
            table = Table(title="Available Models")
            table.add_column("Name", style="cyan")
            table.add_column("Parameters", style="magenta")
            table.add_column("Quantization", style="green")
            table.add_column("Status", style="yellow")

            for model in models:
                params = f"{model.get('parameters', 'Unknown')}"
                quant = model.get("quantization", "None")
                status = "✓ Loaded" if model.get("loaded", False) else "Available"
                table.add_row(model["id"], params, quant, status)

            console.print(table)
        else:
            print("\nAvailable Models:")
            print("----------------")
            for model in models:
                params = f"{model.get('parameters', 'Unknown')}"
                quant = model.get("quantization", "None")
                status = "✓ Loaded" if model.get("loaded", False) else "Available"
                print(f"- {model['id']} ({params}, {quant}): {status}")

    def _create_progress(self, description: str):
        """Create a progress bar for long-running operations"""
        if HAS_RICH:
            return Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[bold green]{task.completed} of {task.total}"),
                TimeElapsedColumn(),
            )
        return None


def main():
    """Main entry point"""
    cli = LlamaScaleCLI()
    exit_code = asyncio.run(cli.run())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
