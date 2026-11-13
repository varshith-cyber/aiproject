"""Glue code shared by the CLI and the Streamlit app."""
from __future__ import annotations

from .data import load_data
from .models import (ContentBased, HybridRecommender, ItemBasedCF, MatrixFactorizationALS,
                     PopularityBaseline, UserBasedCF)
from .preprocessing import clean_data, train_test_split


def prepare(synthetic=False, min_ratings=5, test_size=0.2, seed=42):
    movies, ratings = load_data(synthetic)
    ds = clean_data(movies, ratings, min_ratings, min_ratings)
    train, test = train_test_split(ds.ratings, test_size, seed)
    return ds, train, test


def build_models(ds, alpha=0.7) -> dict:
    return {
        "popularity": PopularityBaseline(),
        "content": ContentBased(ds.movies),
        "user_cf_cosine": UserBasedCF("cosine"),
        "user_cf_pearson": UserBasedCF("pearson"),
        "item_cf_cosine": ItemBasedCF("cosine"),
        "item_cf_pearson": ItemBasedCF("pearson"),
        "svd": MatrixFactorizationALS(),
        "hybrid": HybridRecommender(MatrixFactorizationALS(), ContentBased(ds.movies), alpha),
    }
