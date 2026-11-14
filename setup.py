"""
Setup configuration for CineMatch Movie Recommendation System.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List
from setuptools import find_packages, setup

# Base directory for relative file resolution
BASE_DIR = Path(__file__).resolve().parent


def read_long_description() -> str:
    """Read the long description from README.md with fallback."""
    readme_file = BASE_DIR / "README.md"
    try:
        if readme_file.exists():
            return readme_file.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Warning: Could not read README.md: {exc}")
    return "CineMatch: Production-grade Movie Recommendation Engine"


def parse_requirements(filename: str = "requirements.txt") -> List[str]:
    """
    Parse dependencies from a requirements file.
    Filters out comments, empty lines, and standard library modules.
    """
    req_file = BASE_DIR / filename
    requirements: List[str] = []
    if not req_file.exists():
        # Fallback default requirements matching project specifications
        return [
            "streamlit>=1.28.0",
            "pandas>=1.5.0",
            "numpy>=1.23.0",
            "scikit-learn>=1.3.0",
            "scikit-surprise>=1.1.3",
            "plotly>=5.17.0",
            "scipy>=1.11.0",
            "Pillow>=10.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ]

    try:
        with open(req_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments, blank lines, and stdlib mentions
                if not line or line.startswith("#") or "stdlib" in line.lower():
                    continue
                # Skip difflib if present (Python stdlib)
                if line.lower().startswith("difflib"):
                    continue
                requirements.append(line)
    except Exception as exc:
        print(f"Warning: Error parsing {filename}: {exc}")
        return [
            "streamlit>=1.28.0",
            "pandas>=1.5.0",
            "numpy>=1.23.0",
            "scikit-learn>=1.3.0",
            "scikit-surprise>=1.1.3",
            "plotly>=5.17.0",
            "scipy>=1.11.0",
            "Pillow>=10.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ]

    return requirements


setup(
    name="cinematch",
    version="1.0.0",
    author="CineMatch Engineering Team",
    author_email="dev@cinematch.ai",
    description="Intelligent Movie Recommendation Platform with Content-Based, Collaborative, and Hybrid Models",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/example/movie-recommender",
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cinematch-generate=data.generate_dataset:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="recommendation-system movie-recommender machine-learning collaborative-filtering content-based hybrid streamlit",
)
