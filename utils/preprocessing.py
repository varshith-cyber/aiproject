"""
Data Preprocessing Utilities
Handles cleaning, feature extraction (genre one-hot, normalization),
and matrix formation for collaborative filtering.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split as sk_split


class DataPreprocessor:
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.scaler = MinMaxScaler()
        self.movies_df = None
        self.ratings_df = None
        self.users_df = None
        self.genre_matrix = None
        self.user_movie_matrix = None
        self.content_features = None

    def load_from_dataframes(self, movies_df, ratings_df, users_df=None):
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy()
        self.users_df = users_df.copy() if users_df is not None else None
        self._clean()
        return self

    def _clean(self):
        self.movies_df.drop_duplicates(subset=["movie_id"], inplace=True)
        self.ratings_df.drop_duplicates(subset=["user_id", "movie_id"], inplace=True)
        self.ratings_df["rating"] = self.ratings_df["rating"].clip(0.5, 5.0)

    def build_genre_matrix(self):
        genres_split = self.movies_df["genres"].str.split("|")
        encoded = self.mlb.fit_transform(genres_split)
        self.genre_matrix = pd.DataFrame(
            encoded,
            index=self.movies_df["movie_id"].values,
            columns=self.mlb.classes_,
        )
        return self.genre_matrix

    def build_content_features(self):
        gm = self.build_genre_matrix()
        meta = self.movies_df[["movie_id", "year", "rating"]].copy().set_index("movie_id")
        meta[["year_norm", "rating_norm"]] = self.scaler.fit_transform(meta[["year", "rating"]])
        combined = gm.join(meta[["year_norm", "rating_norm"]])
        self.content_features = combined
        return combined

    def build_user_movie_matrix(self):
        self.user_movie_matrix = self.ratings_df.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        )
        return self.user_movie_matrix

    def train_test_split(self, test_size=0.2, random_state=42):
        train, test = sk_split(self.ratings_df, test_size=test_size, random_state=random_state)
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def get_stats(self):
        n_users = self.ratings_df["user_id"].nunique()
        n_movies = len(self.movies_df)
        n_ratings = len(self.ratings_df)
        genres = sorted(self.mlb.classes_.tolist()) if hasattr(self.mlb, "classes_") else []
        return {
            "n_movies": n_movies,
            "n_users": n_users,
            "n_ratings": n_ratings,
            "avg_rating": round(float(self.ratings_df["rating"].mean()), 2),
            "sparsity": round(1 - n_ratings / (n_users * n_movies), 4) if (n_users * n_movies) > 0 else 0,
            "genres": genres,
        }
