"""
Collaborative Filtering Recommender
=====================================
Implements three collaborative-filtering strategies:

1. **User-Based CF** – Pearson correlation between user rating vectors
2. **Item-Based CF** – Cosine similarity between item rating vectors
3. **SVD-Based CF** – Matrix factorisation via TruncatedSVD

All methods work on a user-item ratings matrix where missing values
are filled with 0 (mean-centred before similarity computation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple


class CollaborativeFilteringRecommender:
    """
    Collaborative Filtering with User-Based, Item-Based, and SVD approaches.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Columns: user_id, movie_id, rating
    movies_df : pd.DataFrame
        Columns: movie_id, title, genres, avg_rating, ...
    """

    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_similarity: Optional[np.ndarray] = None
        self.item_similarity: Optional[np.ndarray] = None
        self._svd_model: Optional[TruncatedSVD] = None
        self._svd_matrix: Optional[np.ndarray] = None
        self._build_user_item_matrix()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_user_item_matrix(self) -> None:
        """
        Pivot ratings into a user x movie matrix.
        Missing ratings are filled with 0.
        """
        self.user_item_matrix = self.ratings_df.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating",
            aggfunc="mean",
        ).fillna(0)

    def _mean_centre(self) -> pd.DataFrame:
        """Subtract each user's mean rating (ignoring zeros)."""
        matrix = self.user_item_matrix.copy()
        user_means = self.ratings_df.groupby("user_id")["rating"].mean()
        for uid in matrix.index:
            if uid in user_means.index:
                mask = matrix.loc[uid] != 0
                matrix.loc[uid, mask] -= user_means[uid]
        return matrix

    # ------------------------------------------------------------------
    # User-Based CF
    # ------------------------------------------------------------------

    def _compute_user_similarity(self) -> np.ndarray:
        """Pearson correlation matrix between all pairs of users."""
        matrix = self._mean_centre().values
        n_users = matrix.shape[0]
        sim = np.zeros((n_users, n_users))
        for i in range(n_users):
            for j in range(i, n_users):
                # Only compare on movies both users rated
                mask = (self.user_item_matrix.values[i] != 0) & \
                       (self.user_item_matrix.values[j] != 0)
                if mask.sum() < 2:
                    corr = 0.0
                else:
                    try:
                        corr, _ = pearsonr(matrix[i][mask], matrix[j][mask])
                        corr = 0.0 if np.isnan(corr) else corr
                    except Exception:
                        corr = 0.0
                sim[i][j] = sim[j][i] = corr
        return sim

    def user_based_recommendations(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """
        Recommend movies for *user_id* using User-Based CF.

        Finds the most similar users, then aggregates their ratings for
        movies the target user has not yet seen, weighted by similarity.

        Parameters
        ----------
        user_id : int
        n : int  Number of recommendations.

        Returns
        -------
        pd.DataFrame  Columns: title, genres, predicted_rating, year
        """
        if self.user_similarity is None:
            self.user_similarity = self._compute_user_similarity()

        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found.")

        user_idx = list(self.user_item_matrix.index).index(user_id)
        sim_scores = self.user_similarity[user_idx]

        # Movies the user has NOT rated
        user_ratings = self.user_item_matrix.loc[user_id]
        unseen_movies = user_ratings[user_ratings == 0].index.tolist()

        predictions: List[Tuple[int, float]] = []
        for movie_id in unseen_movies:
            if movie_id not in self.user_item_matrix.columns:
                continue
            movie_col = self.user_item_matrix[movie_id]
            raters_mask = movie_col != 0
            if raters_mask.sum() == 0:
                continue
            rater_indices = [
                list(self.user_item_matrix.index).index(uid)
                for uid in self.user_item_matrix.index[raters_mask]
            ]
            weights = np.array([sim_scores[ri] for ri in rater_indices])
            ratings_v = movie_col[raters_mask].values
            denom = np.abs(weights).sum()
            if denom < 1e-9:
                continue
            pred = np.dot(weights, ratings_v) / denom
            predictions.append((movie_id, round(pred, 3)))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_ids = [p[0] for p in predictions[:n]]
        top_scores = {p[0]: p[1] for p in predictions[:n]}

        result = self.movies_df[self.movies_df["movie_id"].isin(top_ids)][
            ["movie_id", "title", "genres", "year", "avg_rating"]
        ].copy()
        result["predicted_rating"] = result["movie_id"].map(top_scores)
        result = result.sort_values("predicted_rating", ascending=False)
        result = result.drop(columns=["movie_id"]).reset_index(drop=True)
        result.index += 1
        return result

    # ------------------------------------------------------------------
    # Item-Based CF
    # ------------------------------------------------------------------

    def _compute_item_similarity(self) -> np.ndarray:
        """Cosine similarity matrix between all movie vectors."""
        item_matrix = self.user_item_matrix.T.values  # shape: (movies, users)
        return cosine_similarity(item_matrix)

    def item_based_recommendations(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """
        Recommend movies for *user_id* using Item-Based CF.

        For each movie the user rated highly, find the most similar
        unrated movies.

        Parameters
        ----------
        user_id : int
        n : int  Number of recommendations.

        Returns
        -------
        pd.DataFrame
        """
        if self.item_similarity is None:
            self.item_similarity = self._compute_item_similarity()

        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found.")

        user_ratings = self.user_item_matrix.loc[user_id]
        rated_mask = user_ratings > 0
        unrated_mask = user_ratings == 0

        movie_cols = list(self.user_item_matrix.columns)
        rated_indices = [movie_cols.index(m) for m in user_ratings[rated_mask].index
                         if m in movie_cols]
        unrated_indices = [movie_cols.index(m) for m in user_ratings[unrated_mask].index
                           if m in movie_cols]

        score_map: Dict[int, float] = {}
        for unrated_idx in unrated_indices:
            movie_id = movie_cols[unrated_idx]
            # Weighted sum: similarity * user_rating for all rated movies
            sims = np.array([self.item_similarity[unrated_idx][ri] for ri in rated_indices])
            user_r = user_ratings[rated_mask].values
            denom = np.abs(sims).sum()
            if denom < 1e-9:
                continue
            pred = np.dot(sims, user_r) / denom
            score_map[movie_id] = round(pred, 3)

        sorted_movies = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:n]
        top_ids = [m[0] for m in sorted_movies]
        top_scores = dict(sorted_movies)

        result = self.movies_df[self.movies_df["movie_id"].isin(top_ids)][
            ["movie_id", "title", "genres", "year", "avg_rating"]
        ].copy()
        result["predicted_rating"] = result["movie_id"].map(top_scores)
        result = result.sort_values("predicted_rating", ascending=False)
        result = result.drop(columns=["movie_id"]).reset_index(drop=True)
        result.index += 1
        return result

    # ------------------------------------------------------------------
    # SVD Matrix Factorisation
    # ------------------------------------------------------------------

    def _fit_svd(self, n_components: int = 20) -> None:
        """Fit TruncatedSVD and reconstruct the full ratings matrix."""
        matrix = self.user_item_matrix.values.astype(float)
        n_components = min(n_components, min(matrix.shape) - 1)
        self._svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        U = self._svd_model.fit_transform(matrix)
        Vt = self._svd_model.components_
        self._svd_matrix = np.dot(U, Vt)

    def svd_recommendations(
        self, user_id: int, n: int = 10, n_components: int = 20
    ) -> pd.DataFrame:
        """
        Recommend movies via SVD (matrix factorisation).

        Decomposes the user-item matrix, reconstructs it, then picks
        the top predicted-rating movies the user has not yet seen.

        Parameters
        ----------
        user_id : int
        n : int  Number of recommendations.
        n_components : int  Latent factors for SVD.

        Returns
        -------
        pd.DataFrame
        """
        if self._svd_matrix is None:
            self._fit_svd(n_components)

        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found.")

        user_idx = list(self.user_item_matrix.index).index(user_id)
        predicted_row = self._svd_matrix[user_idx]

        # Mask already-rated movies
        user_actual = self.user_item_matrix.loc[user_id].values
        movie_cols = list(self.user_item_matrix.columns)

        scored = [
            (movie_cols[i], predicted_row[i])
            for i in range(len(movie_cols))
            if user_actual[i] == 0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:n]
        top_ids = [t[0] for t in top]
        top_scores = {t[0]: round(t[1], 3) for t in top}

        result = self.movies_df[self.movies_df["movie_id"].isin(top_ids)][
            ["movie_id", "title", "genres", "year", "avg_rating"]
        ].copy()
        result["svd_predicted_rating"] = result["movie_id"].map(top_scores)
        result = result.sort_values("svd_predicted_rating", ascending=False)
        result = result.drop(columns=["movie_id"]).reset_index(drop=True)
        result.index += 1
        return result

    # ------------------------------------------------------------------
    # Rating Prediction
    # ------------------------------------------------------------------

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict the rating *user_id* would give *movie_id*.

        Uses the SVD-reconstructed matrix when available, otherwise
        returns the global mean rating.

        Returns
        -------
        float  Predicted rating in [1, 5].
        """
        if self._svd_matrix is None:
            self._fit_svd()

        if user_id not in self.user_item_matrix.index:
            return float(self.ratings_df["rating"].mean())
        if movie_id not in self.user_item_matrix.columns:
            return float(self.ratings_df["rating"].mean())

        user_idx = list(self.user_item_matrix.index).index(user_id)
        movie_idx = list(self.user_item_matrix.columns).index(movie_id)
        pred = self._svd_matrix[user_idx][movie_idx]
        return float(np.clip(pred, 1.0, 5.0))

    def get_similar_users(self, user_id: int, n: int = 5) -> List[Tuple[int, float]]:
        """Return the n most similar users to user_id (Pearson)."""
        if self.user_similarity is None:
            self.user_similarity = self._compute_user_similarity()

        if user_id not in self.user_item_matrix.index:
            return []

        user_idx = list(self.user_item_matrix.index).index(user_id)
        sims = list(enumerate(self.user_similarity[user_idx]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        result = []
        for idx, score in sims:
            if idx == user_idx:
                continue
            uid = list(self.user_item_matrix.index)[idx]
            result.append((uid, round(score, 4)))
            if len(result) >= n:
                break
        return result
