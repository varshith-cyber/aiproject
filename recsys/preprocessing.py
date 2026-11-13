"""Data cleaning, feature preparation, train/test splitting."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    movies: pd.DataFrame      # one row per item, ordered by item index `i`
    ratings: pd.DataFrame     # userId, movieId, rating, u, i
    user_ids: np.ndarray      # u -> original userId
    item_ids: np.ndarray      # i -> original movieId

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def extract_year(title: str):
    m = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(m.group(1)) if m else np.nan


def clean_data(movies: pd.DataFrame, ratings: pd.DataFrame,
               min_user_ratings: int = 5, min_item_ratings: int = 5) -> Dataset:
    """Drop missing/duplicate rows, filter sparse users/items, re-index to 0..n-1."""
    ratings = ratings.dropna().drop_duplicates(["userId", "movieId"], keep="last")
    ratings = ratings[ratings.movieId.isin(movies.movieId)]

    while True:  # filtering users can make items sparse and vice versa
        before = len(ratings)
        ic = ratings.movieId.value_counts()
        ratings = ratings[ratings.movieId.map(ic) >= min_item_ratings]
        uc = ratings.userId.value_counts()
        ratings = ratings[ratings.userId.map(uc) >= min_user_ratings]
        if len(ratings) == before:
            break

    user_ids = np.sort(ratings.userId.unique())
    item_ids = np.sort(ratings.movieId.unique())
    umap = {u: k for k, u in enumerate(user_ids)}
    imap = {m: k for k, m in enumerate(item_ids)}
    ratings = ratings.assign(u=ratings.userId.map(umap), i=ratings.movieId.map(imap)).reset_index(drop=True)

    movies = movies[movies.movieId.isin(item_ids)].copy()
    movies["i"] = movies.movieId.map(imap)
    movies = movies.sort_values("i").reset_index(drop=True)
    movies["year"] = movies.title.map(extract_year)
    return Dataset(movies, ratings, user_ids, item_ids)


def train_test_split(ratings: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Per-user random split so every user appears in both sets."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, g in ratings.groupby("u"):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        test_idx.extend(idx[: max(1, int(round(len(idx) * test_size)))])
    mask = ratings.index.isin(test_idx)
    return ratings[~mask].reset_index(drop=True), ratings[mask].reset_index(drop=True)
