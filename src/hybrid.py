"""
Hybrid Recommender
==================
Combines Content-Based and Collaborative Filtering with weighted
score merging.  Also provides trending and cold-start helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from content_based import ContentBasedRecommender
from collaborative import CollaborativeFilteringRecommender


class HybridRecommender:
    """
    Weighted hybrid of content-based and collaborative filtering.

    Parameters
    ----------
    content_rec : ContentBasedRecommender
    collab_rec  : CollaborativeFilteringRecommender
    movies_df   : pd.DataFrame
    ratings_df  : pd.DataFrame
    """

    def __init__(
        self,
        content_rec: ContentBasedRecommender,
        collab_rec: CollaborativeFilteringRecommender,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
    ) -> None:
        self.content_rec = content_rec
        self.collab_rec = collab_rec
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy()

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
        """Min-max normalize a {movie_id: score} dict to [0, 1]."""
        if not scores:
            return {}
        vals = np.array(list(scores.values()), dtype=float)
        lo, hi = vals.min(), vals.max()
        if hi == lo:
            return {k: 1.0 for k in scores}
        return {k: float((v - lo) / (hi - lo)) for k, v in scores.items()}

    @staticmethod
    def _weighted_merge(
        content_scores: Dict[int, float],
        collab_scores: Dict[int, float],
        content_weight: float,
        collab_weight: float,
    ) -> Dict[int, float]:
        """
        Merge two score dicts with per-source weights.

        Movies present in only one source receive a 0 from the other.
        """
        all_ids = set(content_scores) | set(collab_scores)
        merged: Dict[int, float] = {}
        for mid in all_ids:
            cs = content_scores.get(mid, 0.0)
            ls = collab_scores.get(mid, 0.0)
            merged[mid] = content_weight * cs + collab_weight * ls
        return merged

    # ------------------------------------------------------------------
    # Main recommendation entry-point
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: int,
        movie_title: Optional[str] = None,
        n: int = 10,
        content_weight: float = 0.4,
        collab_weight: float = 0.6,
    ) -> pd.DataFrame:
        """
        Hybrid recommendation for a user, optionally seeded by a movie.

        Strategy
        --------
        * **Collaborative** signal: SVD-predicted ratings for unseen movies.
        * **Content-based** signal (if *movie_title* given): cosine similarity
          to the seed movie; otherwise the user's genre profile is used.
        * Scores are min-max normalised then merged with the given weights.

        Parameters
        ----------
        user_id : int
        movie_title : str, optional  Seed movie for content signal.
        n : int  Number of recommendations.
        content_weight : float  Weight for content signal (default 0.4).
        collab_weight : float  Weight for collab signal (default 0.6).

        Returns
        -------
        pd.DataFrame
            Columns: title, genres, year, avg_rating,
                     content_score, collab_score, hybrid_score
        """
        # ---- Collaborative scores ----------------------------------------
        try:
            collab_df = self.collab_rec.svd_recommendations(user_id, n=50)
            collab_raw: Dict[int, float] = {}
            for _, row in collab_df.iterrows():
                mid = self.movies_df.loc[
                    self.movies_df["title"] == row["title"], "movie_id"
                ].values
                if len(mid):
                    collab_raw[int(mid[0])] = float(row["svd_predicted_rating"])
        except Exception:
            collab_raw = {}

        # ---- Content scores ----------------------------------------------
        try:
            if movie_title:
                content_df = self.content_rec.get_recommendations(movie_title, n=50)
                score_col = "similarity_score"
            else:
                # Build genre profile from user's rated movies
                user_rated = self.ratings_df[self.ratings_df["user_id"] == user_id]
                rated_movies = self.movies_df[
                    self.movies_df["movie_id"].isin(user_rated["movie_id"])
                ]
                liked = user_rated[user_rated["rating"] >= 3.5]
                liked_genres_series = self.movies_df[
                    self.movies_df["movie_id"].isin(liked["movie_id"])
                ]["genres"].str.split("|").explode()
                liked_genres = liked_genres_series.value_counts().head(3).index.tolist()
                content_df = self.content_rec.get_recommendations_by_profile(
                    liked_genres or ["Drama"], n=50
                )
                score_col = "match_score"

            content_raw: Dict[int, float] = {}
            for _, row in content_df.iterrows():
                mid = self.movies_df.loc[
                    self.movies_df["title"] == row["title"], "movie_id"
                ].values
                if len(mid):
                    content_raw[int(mid[0])] = float(row[score_col])
        except Exception:
            content_raw = {}

        # ---- Normalise and merge -----------------------------------------
        norm_content = self._normalize_scores(content_raw)
        norm_collab = self._normalize_scores(collab_raw)
        merged = self._weighted_merge(norm_content, norm_collab, content_weight, collab_weight)

        # Remove movies the user already rated
        already_rated = set(
            self.ratings_df[self.ratings_df["user_id"] == user_id]["movie_id"]
        )
        merged = {k: v for k, v in merged.items() if k not in already_rated}

        sorted_ids = sorted(merged, key=merged.get, reverse=True)[:n]

        rows = []
        for mid in sorted_ids:
            movie_row = self.movies_df[self.movies_df["movie_id"] == mid]
            if movie_row.empty:
                continue
            movie_row = movie_row.iloc[0]
            rows.append({
                "title": movie_row["title"],
                "genres": movie_row["genres"],
                "year": movie_row["year"],
                "avg_rating": movie_row["avg_rating"],
                "content_score": round(norm_content.get(mid, 0.0), 4),
                "collab_score": round(norm_collab.get(mid, 0.0), 4),
                "hybrid_score": round(merged[mid], 4),
            })

        result = pd.DataFrame(rows)
        if result.empty:
            return result
        result = result.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        result.index += 1
        return result

    # ------------------------------------------------------------------
    # Trending
    # ------------------------------------------------------------------

    def get_trending(self, n: int = 10) -> pd.DataFrame:
        """
        Return trending movies: high rating count AND high average rating.

        A Bayesian average (Wilson-like) is computed:
            score = (votes * avg) / (votes + C)  where C = mean vote count

        Parameters
        ----------
        n : int  Number of trending movies to return.

        Returns
        -------
        pd.DataFrame
        """
        agg = (
            self.ratings_df.groupby("movie_id")["rating"]
            .agg(vote_count="count", mean_rating="mean")
            .reset_index()
        )
        C = agg["vote_count"].mean()
        agg["trend_score"] = (agg["vote_count"] * agg["mean_rating"]) / (
            agg["vote_count"] + C
        )
        agg = agg.sort_values("trend_score", ascending=False).head(n)

        result = agg.merge(
            self.movies_df[["movie_id", "title", "genres", "year", "avg_rating"]],
            on="movie_id",
        )
        result = result[["title", "genres", "year", "mean_rating", "vote_count", "trend_score"]]
        result["mean_rating"] = result["mean_rating"].round(2)
        result["trend_score"] = result["trend_score"].round(4)
        result = result.reset_index(drop=True)
        result.index += 1
        return result

    # ------------------------------------------------------------------
    # Cold-start
    # ------------------------------------------------------------------

    def cold_start_recommendations(
        self, preferred_genres: List[str], n: int = 10
    ) -> pd.DataFrame:
        """
        Recommend movies for a brand-new user (no rating history).

        Uses content-based genre matching combined with global popularity.

        Parameters
        ----------
        preferred_genres : list of str  e.g. ["Action", "Sci-Fi"]
        n : int

        Returns
        -------
        pd.DataFrame
        """
        content_df = self.content_rec.get_recommendations_by_profile(
            preferred_genres, n=n * 3
        )
        content_raw: Dict[int, float] = {}
        for _, row in content_df.iterrows():
            mid = self.movies_df.loc[
                self.movies_df["title"] == row["title"], "movie_id"
            ].values
            if len(mid):
                content_raw[int(mid[0])] = float(row["match_score"])

        # Popularity signal
        pop = (
            self.ratings_df.groupby("movie_id")["rating"]
            .agg(vote_count="count", mean_rating="mean")
            .reset_index()
        )
        C = pop["vote_count"].mean()
        pop["pop_score"] = (pop["vote_count"] * pop["mean_rating"]) / (
            pop["vote_count"] + C
        )
        pop_raw: Dict[int, float] = dict(zip(pop["movie_id"], pop["pop_score"]))

        norm_content = self._normalize_scores(content_raw)
        norm_pop = self._normalize_scores(pop_raw)
        merged = self._weighted_merge(norm_content, norm_pop, 0.7, 0.3)

        sorted_ids = sorted(merged, key=merged.get, reverse=True)[:n]

        rows = []
        for mid in sorted_ids:
            movie_row = self.movies_df[self.movies_df["movie_id"] == mid]
            if movie_row.empty:
                continue
            movie_row = movie_row.iloc[0]
            rows.append({
                "title": movie_row["title"],
                "genres": movie_row["genres"],
                "year": movie_row["year"],
                "avg_rating": movie_row["avg_rating"],
                "genre_match": round(norm_content.get(mid, 0.0), 4),
                "popularity": round(norm_pop.get(mid, 0.0), 4),
                "final_score": round(merged[mid], 4),
            })

        result = pd.DataFrame(rows)
        if result.empty:
            return result
        result = result.sort_values("final_score", ascending=False).reset_index(drop=True)
        result.index += 1
        return result
