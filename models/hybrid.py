"""
Hybrid Recommender
Combines Content-Based + Collaborative Filtering scores with configurable weights.
"""

import numpy as np
import pandas as pd


class HybridRecommender:
    """
    Weighted hybrid of:
      • Content-Based Filtering (CBF)
      • Item-Based Collaborative Filtering (ItemCF)
      • SVD Matrix Factorization
    """

    def __init__(
        self,
        content_model,
        item_cf_model,
        svd_model,
        weights: dict = None,
    ):
        self.cbf      = content_model
        self.item_cf  = item_cf_model
        self.svd      = svd_model
        self.weights  = weights or {"content": 0.35, "item_cf": 0.35, "svd": 0.30}

    def _normalise(self, series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    def recommend(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
          movie_id | title | genres | year | rating | hybrid_score
        """
        # Content-Based scores
        try:
            cbf_recs = self.cbf.recommend_for_user(user_id, ratings_df, top_n=50)
            cbf_scores = pd.Series(cbf_recs["score"].values, index=cbf_recs["movie_id"])
        except Exception:
            cbf_scores = pd.Series(dtype=float)

        # Item-CF scores
        try:
            icf_recs = self.item_cf.recommend(user_id, top_n=50)
            icf_scores = pd.Series(icf_recs["predicted_rating"].values, index=icf_recs["movie_id"])
        except Exception:
            icf_scores = pd.Series(dtype=float)

        # SVD scores
        try:
            svd_recs = self.svd.recommend(user_id, top_n=50)
            svd_scores = pd.Series(svd_recs["predicted_rating"].values, index=svd_recs["movie_id"])
        except Exception:
            svd_scores = pd.Series(dtype=float)

        # Combine on common movies
        all_ids = set(cbf_scores.index) | set(icf_scores.index) | set(svd_scores.index)
        hybrid  = {}
        for mid in all_ids:
            c = self._normalise(cbf_scores).get(mid, 0.0)
            i = self._normalise(icf_scores).get(mid, 0.0)
            s = self._normalise(svd_scores).get(mid, 0.0)
            hybrid[mid] = (
                self.weights["content"]  * c +
                self.weights["item_cf"]  * i +
                self.weights["svd"]      * s
            )

        hybrid_series = pd.Series(hybrid).sort_values(ascending=False).head(top_n)
        mdf = movies_df.set_index("movie_id")
        result = mdf.loc[hybrid_series.index, ["title", "genres", "year", "rating"]].copy()
        result["hybrid_score"] = hybrid_series.values
        result.reset_index(inplace=True)
        return result
