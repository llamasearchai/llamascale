#!/usr/bin/env python3
"""
LlamaScale Ultra 6.0 - Mac-Native Enterprise LLM Orchestration Platform
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llamascale-llamasearch",
    version="6.0.0",
    author="LlamaSearch AI",
    author_email="nikjois@llamasearch.ai",
    description="Mac-Native Enterprise LLM Orchestration Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://llamasearch.ai",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn>=0.18.0",
        "pydantic>=2.0.0",
        "httpx>=0.23.0",
        "numpy>=1.20.0",
        "asyncio>=3.4.3",
        "rich>=12.0.0",
        "sse-starlette>=1.6.1",  # For streaming responses
        "python-multipart>=0.0.6",  # For file uploads
        "email-validator>=2.0.0",  # For email validation
        "typer>=0.9.0",  # For improved CLI
        "aiofiles>=23.2.0",  # For async file operations
    ],
    extras_require={
        "mlx": ["mlx>=2.0.0"],
        "redis": ["redis>=4.3.4"],
        "monitoring": ["prometheus-client>=0.14.1", "opentelemetry-api>=1.11.1", "grafana-client>=0.5.0"],
        "agents": [
            "openai>=1.0.0", 
            "tiktoken>=0.5.0", 
            "numpy>=1.20.0",
            "langchain>=0.1.0",
            "chromadb>=0.4.22"
        ],
        "multimodal": [
            "pillow>=10.0.0",
            "mlx-vision>=0.0.3",
            "transformers>=4.36.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.1",
            "black>=23.9.1",
            "isort>=5.12.0",
            "mypy>=1.5.1",
        ],
        "all": [
            "mlx>=2.0.0",
            "redis>=4.3.4",
            "prometheus-client>=0.14.1",
            "opentelemetry-api>=1.11.1",
            "grafana-client>=0.5.0",
            "openai>=1.0.0",
            "tiktoken>=0.5.0",
            "langchain>=0.1.0",
            "chromadb>=0.4.22",
            "pillow>=10.0.0",
            "mlx-vision>=0.0.3",
            "transformers>=4.36.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llamascale=llamascale.tools.cli.llamascale_cli:main",
            "llamascale_api=llamascale_api:main",
        ],
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
) 
# Updated in commit 5 - 2025-04-04 17:32:29

# Updated in commit 13 - 2025-04-04 17:32:30

# Updated in commit 21 - 2025-04-04 17:32:31

# Updated in commit 29 - 2025-04-04 17:32:31

# Updated in commit 5 - 2025-04-05 14:36:07

# Updated in commit 13 - 2025-04-05 14:36:08

# Updated in commit 21 - 2025-04-05 14:36:08

# Updated in commit 29 - 2025-04-05 14:36:08

# Updated in commit 5 - 2025-04-05 15:22:37

# Updated in commit 13 - 2025-04-05 15:22:37

# Updated in commit 21 - 2025-04-05 15:22:38
