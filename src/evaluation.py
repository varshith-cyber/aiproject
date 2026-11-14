"""
Model Evaluation Module
========================
Provides offline evaluation metrics for recommendation systems:

* RMSE / MAE        – Rating-prediction accuracy
* Precision@K       – Fraction of top-K recs that are relevant
* Recall@K          – Fraction of relevant items in top-K
* NDCG@K            – Normalised Discounted Cumulative Gain
* Coverage          – Catalogue coverage
* Diversity         – Intra-list genre diversity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class ModelEvaluator:
    """
    Offline evaluator for movie recommendation models.

    Parameters
    ----------
    ratings_df : pd.DataFrame  Columns: user_id, movie_id, rating, timestamp
    movies_df  : pd.DataFrame  Columns: movie_id, title, genres, ...
    """

    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Train / Test split
    # ------------------------------------------------------------------

    def train_test_split(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chronological train-test split (preserves temporal order).

        Each user's ratings are sorted by timestamp; the most recent
        *test_size* fraction becomes the test set.

        Parameters
        ----------
        test_size : float  Fraction of each user's ratings for test.

        Returns
        -------
        (train_df, test_df)
        """
        df = self.ratings_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        train_parts, test_parts = [], []
        for uid, group in df.groupby("user_id"):
            split_pt = max(1, int(len(group) * (1 - test_size)))
            train_parts.append(group.iloc[:split_pt])
            test_parts.append(group.iloc[split_pt:])

        self.train_df = pd.concat(train_parts, ignore_index=True)
        self.test_df = pd.concat(test_parts, ignore_index=True)
        return self.train_df, self.test_df

    # ------------------------------------------------------------------
    # Rating-accuracy metrics
    # ------------------------------------------------------------------

    @staticmethod
    def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Root Mean Squared Error."""
        actual = np.array(actual, dtype=float)
        predicted = np.array(predicted, dtype=float)
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))

    @staticmethod
    def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Mean Absolute Error."""
        actual = np.array(actual, dtype=float)
        predicted = np.array(predicted, dtype=float)
        return float(np.mean(np.abs(actual - predicted)))

    # ------------------------------------------------------------------
    # Ranking metrics
    # ------------------------------------------------------------------

    @staticmethod
    def precision_at_k(
        recommended: List[int], relevant: List[int], k: int = 10
    ) -> float:
        """
        Precision@K — fraction of top-K recommendations that are relevant.

        Parameters
        ----------
        recommended : ordered list of movie_ids
        relevant    : list of actually-liked movie_ids
        k           : cut-off

        Returns
        -------
        float in [0, 1]
        """
        if not recommended or not relevant:
            return 0.0
        top_k = recommended[:k]
        relevant_set = set(relevant)
        hits = sum(1 for r in top_k if r in relevant_set)
        return hits / k

    @staticmethod
    def recall_at_k(
        recommended: List[int], relevant: List[int], k: int = 10
    ) -> float:
        """
        Recall@K — fraction of relevant items recovered in top-K.

        Returns
        -------
        float in [0, 1]
        """
        if not recommended or not relevant:
            return 0.0
        top_k = set(recommended[:k])
        relevant_set = set(relevant)
        hits = len(top_k & relevant_set)
        return hits / len(relevant_set)

    @staticmethod
    def ndcg_at_k(
        recommended: List[int], relevant: List[int], k: int = 10
    ) -> float:
        """
        Normalised Discounted Cumulative Gain @ K.

        Parameters
        ----------
        recommended : ordered list of movie_ids (most relevant first)
        relevant    : list of ground-truth relevant movie_ids

        Returns
        -------
        float in [0, 1]
        """
        if not recommended or not relevant:
            return 0.0
        relevant_set = set(relevant)
        top_k = recommended[:k]

        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, mid in enumerate(top_k)
            if mid in relevant_set
        )
        ideal_hits = min(len(relevant_set), k)
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
        return float(dcg / idcg) if idcg > 0 else 0.0

    # ------------------------------------------------------------------
    # Beyond-accuracy metrics
    # ------------------------------------------------------------------

    @staticmethod
    def coverage(all_recommendations: List[List[int]], total_items: int) -> float:
        """
        Catalogue coverage — fraction of items ever recommended.

        Parameters
        ----------
        all_recommendations : list of recommendation lists (one per user)
        total_items         : total number of items in catalogue

        Returns
        -------
        float percentage [0, 100]
        """
        recommended_items = set(mid for recs in all_recommendations for mid in recs)
        return round(100.0 * len(recommended_items) / total_items, 2)

    def diversity(
        self, recommendations: List[int], movies_df: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Intra-list genre diversity (1 - avg pairwise genre overlap).

        A score of 1 means every pair of recommended movies shares no
        genres; 0 means all movies share identical genres.

        Parameters
        ----------
        recommendations : list of movie_ids
        movies_df       : optional override; uses self.movies_df if None.

        Returns
        -------
        float in [0, 1]
        """
        df = movies_df if movies_df is not None else self.movies_df
        if len(recommendations) < 2:
            return 0.0

        genre_sets = []
        for mid in recommendations:
            row = df[df["movie_id"] == mid]
            if row.empty:
                genre_sets.append(set())
            else:
                genre_sets.append(set(row.iloc[0]["genres"].split("|")))

        overlaps = []
        for i in range(len(genre_sets)):
            for j in range(i + 1, len(genre_sets)):
                a, b = genre_sets[i], genre_sets[j]
                if not a or not b:
                    overlaps.append(0.0)
                else:
                    jaccard = len(a & b) / len(a | b)
                    overlaps.append(jaccard)

        return round(1.0 - float(np.mean(overlaps)), 4)

    # ------------------------------------------------------------------
    # Full evaluation runner
    # ------------------------------------------------------------------

    def evaluate_collaborative(self, recommender) -> Dict:
        """
        Run a full rating-accuracy evaluation on the collaborative recommender.

        Uses train/test split (chronological).  For each test (user, movie)
        pair calls `recommender.predict_rating(user_id, movie_id)`.

        Parameters
        ----------
        recommender : CollaborativeFilteringRecommender

        Returns
        -------
        dict  With keys: rmse, mae, n_test_ratings, coverage, ...
        """
        if self.test_df is None:
            self.train_test_split()

        actual, predicted = [], []
        for _, row in self.test_df.iterrows():
            try:
                pred = recommender.predict_rating(
                    int(row["user_id"]), int(row["movie_id"])
                )
            except Exception:
                pred = float(self.ratings_df["rating"].mean())
            actual.append(float(row["rating"]))
            predicted.append(float(pred))

        report = {
            "n_test_ratings": len(actual),
            "rmse": round(self.rmse(actual, predicted), 4),
            "mae": round(self.mae(actual, predicted), 4),
        }
        return report

    def evaluate_ranking(
        self,
        recommender,
        k: int = 10,
        relevance_threshold: float = 3.5,
        n_users: int = 20,
    ) -> Dict:
        """
        Evaluate ranking quality (Precision, Recall, NDCG) @ K.

        For *n_users* random users, the recommender is called to get top-K
        recommendations, and the test set for that user becomes the
        ground truth.

        Parameters
        ----------
        recommender : must expose `svd_recommendations(user_id, n)` returning
                      a DataFrame with column 'title'.
        k           : cut-off rank
        relevance_threshold : minimum rating to count as relevant
        n_users     : number of users to evaluate

        Returns
        -------
        dict  With keys: avg_precision_at_k, avg_recall_at_k, avg_ndcg_at_k
        """
        if self.test_df is None:
            self.train_test_split()

        all_prec, all_rec, all_ndcg = [], [], []
        all_recs_global: List[List[int]] = []

        users = self.test_df["user_id"].unique()
        np.random.shuffle(users)
        users = users[:n_users]

        for uid in users:
            # Ground truth: test movies rated >= threshold
            user_test = self.test_df[self.test_df["user_id"] == uid]
            relevant_ids = user_test[user_test["rating"] >= relevance_threshold][
                "movie_id"
            ].tolist()
            if not relevant_ids:
                continue

            try:
                recs_df = recommender.svd_recommendations(uid, n=k)
                rec_ids = []
                for title in recs_df["title"]:
                    mid = self.movies_df[self.movies_df["title"] == title]["movie_id"].values
                    if len(mid):
                        rec_ids.append(int(mid[0]))
            except Exception:
                continue

            if not rec_ids:
                continue

            all_recs_global.append(rec_ids)
            all_prec.append(self.precision_at_k(rec_ids, relevant_ids, k))
            all_rec.append(self.recall_at_k(rec_ids, relevant_ids, k))
            all_ndcg.append(self.ndcg_at_k(rec_ids, relevant_ids, k))

        cov = self.coverage(all_recs_global, len(self.movies_df))
        return {
            "k": k,
            "n_users_evaluated": len(all_prec),
            "avg_precision_at_k": round(float(np.mean(all_prec)), 4) if all_prec else 0.0,
            "avg_recall_at_k": round(float(np.mean(all_rec)), 4) if all_rec else 0.0,
            "avg_ndcg_at_k": round(float(np.mean(all_ndcg)), 4) if all_ndcg else 0.0,
            "catalog_coverage_pct": cov,
        }

    def print_report(self, rating_report: Dict, ranking_report: Dict) -> None:
        """Pretty-print combined evaluation results."""
        sep = "=" * 50
        print(f"\n{sep}")
        print("  MODEL EVALUATION REPORT")
        print(sep)
        print(f"  Test ratings:         {rating_report.get('n_test_ratings', 'N/A')}")
        print(f"  RMSE:                 {rating_report.get('rmse', 'N/A')}")
        print(f"  MAE:                  {rating_report.get('mae', 'N/A')}")
        print(f"\n  Ranking metrics @ K={ranking_report.get('k', '?')}:")
        print(f"  Users evaluated:      {ranking_report.get('n_users_evaluated', 'N/A')}")
        print(f"  Precision@K:          {ranking_report.get('avg_precision_at_k', 'N/A')}")
        print(f"  Recall@K:             {ranking_report.get('avg_recall_at_k', 'N/A')}")
        print(f"  NDCG@K:               {ranking_report.get('avg_ndcg_at_k', 'N/A')}")
        print(f"  Catalog coverage:     {ranking_report.get('catalog_coverage_pct', 'N/A')}%")
        print(sep + "\n")
