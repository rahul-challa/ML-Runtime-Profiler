#!/usr/bin/env python3
"""
Setup script for ML Runtime Profiler
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [line.strip() for line in requirements_path.read_text().splitlines() 
                   if line.strip() and not line.startswith("#")]

setup(
    name="ml-runtime-profiler",
    version="1.0.0",
    description="A lightweight runtime profiler for transformer inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ML Runtime Profiler Team",
    author_email="",
    url="",
    packages=find_packages(),
    py_modules=[
        "profiler",
        "compare_onnx", 
        "visualize_results",
        "example_usage",
        "test_installation"
    ],
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "full": [
            "onnxruntime>=1.12.0",
            "tensorboard>=2.10.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ml-profiler=profiler:main",
            "ml-compare-onnx=compare_onnx:main",
            "ml-visualize=visualize_results:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    keywords="machine-learning, profiling, transformer, inference, pytorch, onnx",
    project_urls={
        "Bug Reports": "",
        "Source": "",
        "Documentation": "",
    },
) 