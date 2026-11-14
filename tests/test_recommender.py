"""
Unit Tests — Movie Recommendation System
Run: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd

from data.sample_data import save_data, MOVIES
from utils.preprocessing import DataPreprocessor
from utils.evaluation import rmse, mae, precision_at_k, recall_at_k, ndcg_at_k
from models.content_based import ContentBasedRecommender
from models.collaborative_filtering import UserBasedCF, ItemBasedCF, SVDRecommender
from models.hybrid import HybridRecommender


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def data():
    movies_df, users_df, ratings_df = save_data(output_dir="data")
    prep = DataPreprocessor()
    prep.movies_df  = movies_df
    prep.ratings_df = ratings_df
    content_features  = prep.build_content_features()
    user_movie_matrix = prep.build_user_movie_matrix()
    return movies_df, users_df, ratings_df, content_features, user_movie_matrix


# ── Metric Tests ──────────────────────────────────────────────────────────────
class TestMetrics:
    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_mae_known(self):
        y_true = np.array([3.0, 4.0, 5.0])
        y_pred = np.array([2.0, 4.0, 6.0])
        assert mae(y_true, y_pred) == pytest.approx(2/3, rel=1e-4)

    def test_precision_at_k(self):
        assert precision_at_k([1, 2, 3, 4, 5], {1, 3, 5}, k=5) == pytest.approx(0.6)

    def test_recall_at_k(self):
        assert recall_at_k([1, 2, 3], {1, 3, 7, 9}, k=3) == pytest.approx(0.5)

    def test_ndcg_perfect(self):
        relevant = {1, 2, 3}
        assert ndcg_at_k([1, 2, 3], relevant, k=3) > 0.9

    def test_ndcg_zero(self):
        assert ndcg_at_k([4, 5, 6], {1, 2, 3}, k=3) == pytest.approx(0.0)


# ── Preprocessing Tests ───────────────────────────────────────────────────────
class TestPreprocessor:
    def test_genre_matrix_shape(self, data):
        movies_df, _, _, content_features, _ = data
        prep = DataPreprocessor()
        prep.movies_df = movies_df
        gm = prep.build_genre_matrix()
        assert gm.shape[0] == len(movies_df)
        assert gm.sum(axis=1).min() >= 1   # every movie has ≥1 genre

    def test_user_movie_matrix(self, data):
        movies_df, _, ratings_df, _, uvm = data
        assert uvm.shape[0] > 0
        assert uvm.shape[1] == len(movies_df)

    def test_content_features_no_nan(self, data):
        _, _, _, content_features, _ = data
        assert not content_features.isnull().values.any()


# ── Content-Based Tests ───────────────────────────────────────────────────────
class TestContentBased:
    def test_recommend_by_movie_returns_n(self, data):
        movies_df, _, _, content_features, _ = data
        cbf = ContentBasedRecommender().fit(content_features, movies_df)
        recs = cbf.recommend_by_movie(movie_id=7, top_n=5)
        assert len(recs) == 5

    def test_no_self_recommendation(self, data):
        movies_df, _, _, content_features, _ = data
        cbf = ContentBasedRecommender().fit(content_features, movies_df)
        recs = cbf.recommend_by_movie(movie_id=7, top_n=10)
        assert 7 not in recs["movie_id"].values

    def test_similarity_range(self, data):
        movies_df, _, _, content_features, _ = data
        cbf = ContentBasedRecommender().fit(content_features, movies_df)
        recs = cbf.recommend_by_movie(movie_id=1, top_n=5)
        assert recs["similarity"].between(0, 1).all()


# ── User-Based CF Tests ───────────────────────────────────────────────────────
class TestUserBasedCF:
    def test_recommend_shape(self, data):
        _, _, _, _, uvm = data
        ucf  = UserBasedCF(k_neighbours=5).fit(uvm)
        recs = ucf.recommend(user_id=1, top_n=5)
        assert len(recs) <= 5
        assert "movie_id" in recs.columns

    def test_no_already_rated(self, data):
        _, _, _, _, uvm = data
        ucf  = UserBasedCF(k_neighbours=5).fit(uvm)
        recs = ucf.recommend(user_id=1, top_n=10)
        rated_by_1 = set(uvm.loc[1].dropna().index)
        for mid in recs["movie_id"]:
            assert mid not in rated_by_1


# ── SVD Tests ────────────────────────────────────────────────────────────────
class TestSVD:
    def test_recommend_returns_rows(self, data):
        _, _, _, _, uvm = data
        svd  = SVDRecommender(n_factors=10).fit(uvm)
        recs = svd.recommend(user_id=1, top_n=5)
        assert len(recs) == 5

    def test_predicted_rating_in_range(self, data):
        _, _, _, _, uvm = data
        svd = SVDRecommender(n_factors=10).fit(uvm)
        pred = svd.predict_rating(1, 3)
        assert 0 <= pred <= 6   # allow slight overshoot from SVD


# ── Hybrid Tests ─────────────────────────────────────────────────────────────
class TestHybrid:
    def test_hybrid_combines_all(self, data):
        movies_df, _, ratings_df, content_features, uvm = data
        cbf = ContentBasedRecommender().fit(content_features, movies_df)
        icf = ItemBasedCF().fit(uvm)
        svd = SVDRecommender(n_factors=10).fit(uvm)
        hyb = HybridRecommender(cbf, icf, svd)
        recs = hyb.recommend(user_id=1, ratings_df=ratings_df, movies_df=movies_df, top_n=5)
        assert len(recs) > 0
        assert "hybrid_score" in recs.columns
