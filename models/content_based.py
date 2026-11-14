"""
Content-Based Filtering using Cosine Similarity on Genres, Year, and IMDb Ratings.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self):
        self.similarity_matrix = None
        self.feature_matrix = None
        self.movies_df = None

    def fit(self, feature_matrix: pd.DataFrame, movies_df: pd.DataFrame):
        self.feature_matrix = feature_matrix
        self.movies_df = movies_df.set_index("movie_id")
        sim = cosine_similarity(feature_matrix.values)
        self.similarity_matrix = pd.DataFrame(
            sim, index=feature_matrix.index, columns=feature_matrix.index
        )
        return self

    def recommend_similar(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        if movie_id not in self.similarity_matrix.index:
            raise ValueError(f"movie_id {movie_id} not found in model.")
        scores = self.similarity_matrix[movie_id].drop(movie_id).sort_values(ascending=False)
        top_ids = scores.head(top_n).index
        result = self.movies_df.loc[top_ids, ["title", "genres", "year", "rating"]].copy()
        result["similarity_score"] = scores[top_ids].values
        return result.reset_index()

    def recommend_for_user(self, user_id: int, ratings_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        user_ratings = ratings_df[ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            raise ValueError(f"No ratings found for user_id {user_id}")
        rated_ids = set(user_ratings["movie_id"].tolist())
        all_ids = set(self.similarity_matrix.index.tolist())
        candidates = list(all_ids - rated_ids)
        if not candidates:
            return pd.DataFrame()

        scores = pd.Series(0.0, index=candidates)
        for _, row in user_ratings.iterrows():
            mid = row["movie_id"]
            if mid not in self.similarity_matrix.columns:
                continue
            sim_col = self.similarity_matrix.loc[candidates, mid]
            scores += row["rating"] * sim_col

        scores = scores.sort_values(ascending=False).head(top_n)
        result = self.movies_df.loc[scores.index, ["title", "genres", "year", "rating"]].copy()
        result["score"] = scores.values
        return result.reset_index()

    def predict_rating(self, user_id: int, movie_id: int, ratings_df: pd.DataFrame) -> float:
        try:
            user_ratings = ratings_df[ratings_df["user_id"] == user_id]
            if user_ratings.empty or movie_id not in self.similarity_matrix.columns:
                return float(ratings_df["rating"].mean())
            rated_ids = user_ratings["movie_id"].tolist()
            sim_scores = self.similarity_matrix.loc[movie_id, rated_ids]
            user_r_series = user_ratings.set_index("movie_id").loc[rated_ids, "rating"]
            num = (sim_scores * user_r_series).sum()
            denom = sim_scores.abs().sum()
            return float(num / denom) if denom > 0 else float(ratings_df["rating"].mean())
        except Exception:
            return float(ratings_df["rating"].mean())
