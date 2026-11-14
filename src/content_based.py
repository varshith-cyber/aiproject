"""
Content-Based Filtering Recommender
=====================================
Recommends movies based on their intrinsic features:
genres, description, director, and cast — using TF-IDF
vectorization and cosine similarity.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Dict


class ContentBasedRecommender:
    """
    Content-Based Filtering using TF-IDF + Cosine Similarity.

    The recommender builds a combined textual feature string from each
    movie's genres, description, director and cast.  It then computes
    a full pairwise cosine-similarity matrix, which is used at query
    time to rank candidates.

    Parameters
    ----------
    movies_df : pd.DataFrame
        Must contain columns: movie_id, title, genres, description,
        director, cast (pipe-separated strings).
    """

    def __init__(self, movies_df: pd.DataFrame) -> None:
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self._title_to_idx: Dict[str, int] = {}
        self._preprocess()
        self._compute_similarity()
        self._title_to_idx = {
            title.lower(): idx
            for idx, title in enumerate(self.movies_df["title"])
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_feature_string(self, row: pd.Series) -> str:
        """Combine movie attributes into a single text blob."""
        genres = row.get("genres", "").replace("|", " ")
        description = row.get("description", "")
        director = row.get("director", "").replace(" ", "_")  # treat as token
        cast = row.get("cast", "").replace("|", " ").replace(" ", "_")
        language = row.get("language", "")
        # Repeat genres 3x to give them extra weight
        return f"{genres} {genres} {genres} {description} {director} {cast} {language}"

    def _preprocess(self) -> None:
        """Build TF-IDF matrix from combined feature strings."""
        self.movies_df["_features"] = self.movies_df.apply(
            self._build_feature_string, axis=1
        )
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df["_features"])

    def _compute_similarity(self) -> None:
        """Compute full pairwise cosine-similarity matrix."""
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def _resolve_index(self, movie_title: str) -> Optional[int]:
        """Return row index for a movie title (case-insensitive)."""
        return self._title_to_idx.get(movie_title.lower())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_recommendations(self, movie_title: str, n: int = 10) -> pd.DataFrame:
        """
        Return the top-N movies most similar to *movie_title*.

        Parameters
        ----------
        movie_title : str
            Exact (case-insensitive) title of the seed movie.
        n : int
            Number of recommendations to return.

        Returns
        -------
        pd.DataFrame
            Columns: title, genres, year, avg_rating, similarity_score
        """
        idx = self._resolve_index(movie_title)
        if idx is None:
            raise ValueError(f"Movie '{movie_title}' not found in catalogue.")

        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Exclude the query movie itself
        sim_scores = [s for s in sim_scores if s[0] != idx][:n]

        indices = [s[0] for s in sim_scores]
        scores = [round(s[1], 4) for s in sim_scores]

        result = self.movies_df.iloc[indices][
            ["title", "genres", "year", "avg_rating", "director"]
        ].copy()
        result["similarity_score"] = scores
        result = result.reset_index(drop=True)
        result.index += 1
        return result

    def get_recommendations_by_profile(
        self, liked_genres: List[str], n: int = 10
    ) -> pd.DataFrame:
        """
        Recommend movies matching a user's genre preferences.

        Builds a pseudo-document from the preferred genres, transforms
        it via the fitted TF-IDF, then scores every movie.

        Parameters
        ----------
        liked_genres : list of str
            Genres the user enjoys, e.g. ["Action", "Sci-Fi"].
        n : int
            Number of recommendations to return.

        Returns
        -------
        pd.DataFrame
        """
        profile_text = " ".join(liked_genres * 3)
        profile_vec = self.tfidf.transform([profile_text])
        scores = cosine_similarity(profile_vec, self.tfidf_matrix).flatten()

        top_indices = scores.argsort()[::-1][:n]
        result = self.movies_df.iloc[top_indices][
            ["title", "genres", "year", "avg_rating", "director"]
        ].copy()
        result["match_score"] = [round(scores[i], 4) for i in top_indices]
        result = result.reset_index(drop=True)
        result.index += 1
        return result

    def explain_recommendation(self, movie1: str, movie2: str) -> str:
        """
        Explain why *movie2* was recommended after watching *movie1*.

        Parameters
        ----------
        movie1, movie2 : str
            Movie titles (case-insensitive).

        Returns
        -------
        str
            Human-readable explanation.
        """
        idx1 = self._resolve_index(movie1)
        idx2 = self._resolve_index(movie2)
        if idx1 is None:
            return f"'{movie1}' not found."
        if idx2 is None:
            return f"'{movie2}' not found."

        row1 = self.movies_df.iloc[idx1]
        row2 = self.movies_df.iloc[idx2]
        sim = round(self.similarity_matrix[idx1][idx2], 4)

        genres1 = set(row1["genres"].split("|"))
        genres2 = set(row2["genres"].split("|"))
        shared_genres = genres1 & genres2

        reasons = []
        if shared_genres:
            reasons.append(f"shared genres ({', '.join(shared_genres)})")
        if row1["director"] == row2["director"]:
            reasons.append(f"same director ({row1['director']})")
        cast1 = set(row1["cast"].split("|"))
        cast2 = set(row2["cast"].split("|"))
        shared_cast = cast1 & cast2
        if shared_cast:
            reasons.append(f"overlapping cast ({', '.join(shared_cast)})")

        reason_str = "; ".join(reasons) if reasons else "similar thematic content"
        return (
            f"'{movie2}' was recommended after '{movie1}' "
            f"(similarity: {sim}) because of: {reason_str}."
        )

    def get_movie_info(self, movie_title: str) -> Optional[pd.Series]:
        """Return full row for a movie, or None if not found."""
        idx = self._resolve_index(movie_title)
        if idx is None:
            return None
        return self.movies_df.iloc[idx]
