"""
Collaborative Filtering Models
  1. User-Based CF  – finds similar users, borrows their ratings.
  2. Item-Based CF  – finds similar items, recommends based on rated neighbours.
  3. Matrix Factorization (SVD) – latent factor model.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.stats import pearsonr


# ══════════════════════════════════════════════════════════════════════════════
# 1. USER-BASED COLLABORATIVE FILTERING
# ══════════════════════════════════════════════════════════════════════════════
class UserBasedCF:
    """
    User-Based Collaborative Filtering using Pearson Correlation similarity.
    """

    def __init__(self, k_neighbours: int = 5):
        self.k = k_neighbours
        self.user_movie_matrix = None
        self.user_similarity   = None

    def fit(self, user_movie_matrix: pd.DataFrame):
        """
        Parameters
        ----------
        user_movie_matrix : pivot table (users × movies), NaN = unrated
        """
        self.user_movie_matrix = user_movie_matrix

        # Fill NaN with row mean for similarity calc
        filled = user_movie_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)
        sim = cosine_similarity(filled)
        self.user_similarity = pd.DataFrame(
            sim, index=user_movie_matrix.index, columns=user_movie_matrix.index
        )
        print("✅ UserBasedCF model fitted.")
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict the rating user_id would give movie_id."""
        if movie_id not in self.user_movie_matrix.columns:
            return 0.0

        sim_scores = self.user_similarity[user_id].drop(user_id).sort_values(ascending=False)
        neighbours = sim_scores.head(self.k)

        ratings_col = self.user_movie_matrix[movie_id].dropna()
        common      = neighbours.index.intersection(ratings_col.index)
        if common.empty:
            return self.user_movie_matrix[movie_id].mean() or 0.0

        num   = (neighbours[common] * ratings_col[common]).sum()
        denom = neighbours[common].abs().sum()
        return float(num / denom) if denom else 0.0

    def recommend(self, user_id: int, top_n: int = 5) -> pd.DataFrame:
        """Return top-N unrated movie predictions for user_id."""
        rated   = self.user_movie_matrix.loc[user_id].dropna().index
        unrated = [m for m in self.user_movie_matrix.columns if m not in rated]

        preds = [(mid, self.predict_rating(user_id, mid)) for mid in unrated]
        preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
        return pd.DataFrame(preds, columns=["movie_id", "predicted_rating"])


# ══════════════════════════════════════════════════════════════════════════════
# 2. ITEM-BASED COLLABORATIVE FILTERING
# ══════════════════════════════════════════════════════════════════════════════
class ItemBasedCF:
    """
    Item-Based Collaborative Filtering using cosine similarity between movies.
    """

    def __init__(self, k_neighbours: int = 10):
        self.k = k_neighbours
        self.user_movie_matrix = None
        self.item_similarity   = None

    def fit(self, user_movie_matrix: pd.DataFrame):
        self.user_movie_matrix = user_movie_matrix

        # Transpose: movies × users, fill NaN with 0
        item_matrix = user_movie_matrix.T.fillna(0)
        sim = cosine_similarity(item_matrix)
        self.item_similarity = pd.DataFrame(
            sim, index=user_movie_matrix.columns, columns=user_movie_matrix.columns
        )
        print("✅ ItemBasedCF model fitted.")
        return self

    def recommend(self, user_id: int, top_n: int = 5) -> pd.DataFrame:
        """Recommend movies not yet rated by the user."""
        user_ratings = self.user_movie_matrix.loc[user_id].dropna()
        rated_ids    = user_ratings.index.tolist()
        all_ids      = self.user_movie_matrix.columns.tolist()
        candidate_ids = [m for m in all_ids if m not in rated_ids]

        scores = {}
        for cid in candidate_ids:
            scores[cid] = self._score(user_ratings, rated_ids, cid)

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return pd.DataFrame(top, columns=["movie_id", "predicted_rating"])

    def _score(self, user_ratings, rated_ids, movie_id: int) -> float:
        sim_scores = self.item_similarity[movie_id][rated_ids].sort_values(ascending=False).head(self.k)
        num   = (sim_scores * user_ratings[sim_scores.index]).sum()
        denom = sim_scores.abs().sum()
        return float(num / denom) if denom else 0.0

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict the rating user_id would give movie_id (used for RMSE/MAE eval)."""
        if user_id not in self.user_movie_matrix.index:
            return float(self.user_movie_matrix.stack().mean())
        user_ratings = self.user_movie_matrix.loc[user_id].dropna()
        rated_ids    = user_ratings.index.tolist()
        if not rated_ids or movie_id not in self.item_similarity.columns:
            return float(user_ratings.mean()) if len(user_ratings) else 0.0
        return self._score(user_ratings, rated_ids, movie_id)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MATRIX FACTORIZATION (SVD)
# ══════════════════════════════════════════════════════════════════════════════
class SVDRecommender:
    """
    Latent-factor model using Truncated SVD.
    Projects users and items into a shared low-dimensional space.
    """

    def __init__(self, n_factors: int = 20, n_iter: int = 10):
        self.n_factors = n_factors
        self.n_iter    = n_iter
        self.svd       = TruncatedSVD(n_components=n_factors, n_iter=n_iter, random_state=42)
        self.user_movie_matrix   = None
        self.user_factors        = None
        self.item_factors        = None
        self.predicted_matrix    = None
        self.mean_rating         = None

    def fit(self, user_movie_matrix: pd.DataFrame):
        self.user_movie_matrix = user_movie_matrix
        self.mean_rating = user_movie_matrix.stack().mean()
        filled = user_movie_matrix.fillna(self.mean_rating)

        # Decompose R ≈ U × Σ × Vᵀ
        self.user_factors = self.svd.fit_transform(filled)
        self.item_factors = self.svd.components_

        # Reconstruct predicted rating matrix
        pred = np.dot(self.user_factors, self.item_factors)
        self.predicted_matrix = pd.DataFrame(
            pred, index=user_movie_matrix.index, columns=user_movie_matrix.columns
        )
        print(f"✅ SVD model fitted | Variance explained: {self.svd.explained_variance_ratio_.sum():.1%}")
        return self

    def recommend(self, user_id: int, top_n: int = 5) -> pd.DataFrame:
        if user_id not in self.predicted_matrix.index:
            raise ValueError(f"user_id {user_id} not in training data.")

        rated     = self.user_movie_matrix.loc[user_id].dropna().index
        pred_row  = self.predicted_matrix.loc[user_id].drop(rated)
        top       = pred_row.sort_values(ascending=False).head(top_n)
        return pd.DataFrame({"movie_id": top.index, "predicted_rating": top.values})

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if user_id in self.predicted_matrix.index and movie_id in self.predicted_matrix.columns:
            return float(self.predicted_matrix.loc[user_id, movie_id])
        return self.mean_rating
