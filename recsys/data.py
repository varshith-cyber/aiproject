"""Data loading: MovieLens (auto-download) with a synthetic fallback for offline use."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
          "Drama", "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"]


def download_movielens(data_dir: Path = DATA_DIR) -> Path:
    """Download and unzip MovieLens 'latest-small' (~100k ratings) if not present."""
    target = data_dir / "ml-latest-small"
    if (target / "ratings.csv").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MovieLens from {ML_URL} ...")
    with urllib.request.urlopen(ML_URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(data_dir)
    return target


def load_movielens() -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = download_movielens()
    movies = pd.read_csv(folder / "movies.csv")  # movieId, title, genres
    ratings = pd.read_csv(folder / "ratings.csv")[["userId", "movieId", "rating"]]
    return movies, ratings


def generate_synthetic(n_users: int = 300, n_movies: int = 600, seed: int = 42):
    """Small fake dataset with genre-driven tastes, so every model has signal to learn."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(1, n_movies + 1):
        g = rng.choice(GENRES, size=rng.integers(1, 4), replace=False)
        year = int(rng.integers(1970, 2020))
        rows.append((m, f"Movie {m:04d} ({year})", "|".join(g)))
    movies = pd.DataFrame(rows, columns=["movieId", "title", "genres"])

    quality = rng.normal(0, 0.4, n_movies)
    popularity = rng.pareto(1.5, n_movies) + 1
    popularity /= popularity.sum()
    movie_genres = [set(s.split("|")) for s in movies.genres]

    recs = []
    for u in range(1, n_users + 1):
        liked = set(rng.choice(GENRES, size=3, replace=False))
        n_rated = int(rng.integers(20, 90))
        items = rng.choice(n_movies, size=n_rated, replace=False, p=popularity)
        for i in items:
            match = len(liked & movie_genres[i]) / len(movie_genres[i])
            r = 2.6 + 2.0 * match + quality[i] + rng.normal(0, 0.5)
            recs.append((u, i + 1, float(np.clip(np.round(r * 2) / 2, 0.5, 5.0))))
    ratings = pd.DataFrame(recs, columns=["userId", "movieId", "rating"])
    return movies, ratings


def load_data(synthetic: bool = False):
    if synthetic:
        print("Using synthetic data.")
        return generate_synthetic()
    try:
        return load_movielens()
    except Exception as exc:  # no internet, blocked domain, ...
        print(f"[warn] Could not load MovieLens ({exc}). Falling back to synthetic data.")
        return generate_synthetic()
