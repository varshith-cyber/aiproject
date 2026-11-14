"""
Movie Recommendation System - Dataset Generation Script
========================================================
Generates realistic synthetic datasets for movies, users, and ratings.
Acts as the entrypoint for generating dataset files.
"""

from __future__ import annotations

import os
import sys

# Ensure data directory is on path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from generate_data import main as generate_data_main
except ImportError:
    # Direct import fallback
    import generate_data
    generate_data_main = generate_data.main


def main() -> None:
    """Execute dataset generation."""
    print("🎬 CineMatch Data Pipeline: Generating dataset...")
    generate_data_main()
    print("✅ Dataset generation completed successfully.")


if __name__ == "__main__":
    main()
