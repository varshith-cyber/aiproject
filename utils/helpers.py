"""
Utility helpers for CineMatch Streamlit dashboard.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Optional, Dict


GENRE_EMOJI: Dict[str, str] = {
    "Action": "💥", "Adventure": "🗺️", "Animation": "🎨", "Comedy": "😂",
    "Crime": "🔫", "Documentary": "🎥", "Drama": "🎭", "Fantasy": "🧙",
    "Horror": "👻", "Musical": "🎵", "Mystery": "🔍", "Romance": "❤️",
    "Sci-Fi": "🚀", "Thriller": "😱", "War": "⚔️", "Western": "🤠",
    "Biography": "📖", "Family": "👨‍👩‍👧", "History": "🏛️", "Sport": "🏆",
}

GENRE_COLORS: Dict[str, str] = {
    "Action": "#E50914", "Adventure": "#F5A623", "Animation": "#7ED321",
    "Comedy": "#FFD700", "Crime": "#9B59B6", "Documentary": "#1ABC9C",
    "Drama": "#3498DB", "Fantasy": "#E91E63", "Horror": "#2C3E50",
    "Musical": "#FF6B9D", "Mystery": "#6C3483", "Romance": "#FF4757",
    "Sci-Fi": "#00B4DB", "Thriller": "#C0392B", "War": "#7F8C8D",
    "Western": "#E67E22", "Biography": "#27AE60", "Family": "#F39C12",
    "History": "#8E44AD", "Sport": "#17A589",
}


def format_runtime(minutes: int) -> str:
    """Convert minutes to '2h 15m' format."""
    if minutes is None or pd.isna(minutes):
        return "N/A"
    h = int(minutes) // 60
    m = int(minutes) % 60
    if h == 0:
        return f"{m}m"
    return f"{h}h {m}m" if m else f"{h}h"


def format_number(n) -> str:
    """Format large numbers: 1234567 → '1.2M'."""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "N/A"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n))


def get_genre_emoji(genre: str) -> str:
    """Return emoji for a genre."""
    return GENRE_EMOJI.get(genre, "🎬")


def get_genre_color(genre: str) -> str:
    """Return hex color for a genre badge."""
    return GENRE_COLORS.get(genre, "#555555")


def get_star_rating(rating: float, max_rating: float = 10.0) -> str:
    """Convert a numeric rating to star string (out of 5 stars)."""
    if rating is None or pd.isna(rating):
        return "☆☆☆☆☆"
    stars_5 = rating / max_rating * 5
    full = int(stars_5)
    half = 1 if (stars_5 - full) >= 0.5 else 0
    empty = 5 - full - half
    return "⭐" * full + ("½" if half else "") + "☆" * empty


def compute_bayesian_rating(votes: float, rating: float,
                             mean_votes: float, mean_rating: float) -> float:
    """IMDB-style Bayesian average."""
    if mean_votes <= 0:
        return rating
    return (votes / (votes + mean_votes)) * rating + (mean_votes / (votes + mean_votes)) * mean_rating


def filter_movies(
    df: pd.DataFrame,
    genres: Optional[List[str]] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    min_rating: Optional[float] = None,
    language: Optional[str] = None,
) -> pd.DataFrame:
    """Filter movies DataFrame by multiple criteria."""
    result = df.copy()
    if genres:
        mask = result["genres"].apply(
            lambda g: any(genre in str(g).split("|") for genre in genres)
        )
        result = result[mask]
    if min_year is not None:
        result = result[result["year"] >= min_year]
    if max_year is not None:
        result = result[result["year"] <= max_year]
    if min_rating is not None:
        result = result[result["avg_rating"] >= min_rating]
    if language and language != "All":
        result = result[result["language"] == language]
    return result


def search_movies(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Search movies by title, director, cast, or description."""
    if not query:
        return df
    q = query.lower()
    mask = (
        df["title"].str.lower().str.contains(q, na=False) |
        df["director"].str.lower().str.contains(q, na=False) |
        df["cast"].str.lower().str.contains(q, na=False) |
        df["description"].str.lower().str.contains(q, na=False)
    )
    return df[mask]


def paginate_dataframe(df: pd.DataFrame, page_size: int = 20, page_num: int = 1) -> pd.DataFrame:
    """Return a single page of a DataFrame."""
    start = (page_num - 1) * page_size
    end = start + page_size
    return df.iloc[start:end]


def get_trending_movies(movies_df: pd.DataFrame, ratings_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top-n trending movies using Bayesian average."""
    agg = ratings_df.groupby("movie_id")["rating"].agg(
        vote_count="count", mean_rating="mean"
    ).reset_index()
    C = agg["vote_count"].mean()
    global_mean = agg["mean_rating"].mean()
    agg["trend_score"] = agg.apply(
        lambda r: compute_bayesian_rating(r["vote_count"], r["mean_rating"], C, global_mean), axis=1
    )
    top = agg.nlargest(n, "trend_score")
    result = top.merge(movies_df, on="movie_id", how="left")
    result = result.rename(columns={"mean_rating": "user_rating"})
    return result.reset_index(drop=True)


def get_genre_list(movies_df: pd.DataFrame) -> List[str]:
    """Return sorted list of unique genres from movies DataFrame."""
    all_genres = set()
    for genres_str in movies_df["genres"].dropna():
        for g in genres_str.split("|"):
            g = g.strip()
            if g:
                all_genres.add(g)
    return sorted(all_genres)


def get_user_genre_profile(user_id: int, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Dict[str, float]:
    """Return genre preference scores for a user based on their ratings."""
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    if user_ratings.empty:
        return {}
    merged = user_ratings.merge(movies_df[["movie_id", "genres"]], on="movie_id")
    genre_scores: Dict[str, List[float]] = {}
    for _, row in merged.iterrows():
        for g in str(row["genres"]).split("|"):
            g = g.strip()
            if g:
                genre_scores.setdefault(g, []).append(float(row["rating"]))
    return {g: round(float(np.mean(v)), 2) for g, v in genre_scores.items()}
