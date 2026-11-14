"""
Movie Recommendation System — Main Application
================================================
Ties together data loading, content-based filtering, collaborative
filtering, hybrid recommendations, and evaluation into a single,
easy-to-use class with a runnable demo.

Usage
-----
    cd src
    python app.py
"""

from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Optional

# Allow running from the src/ directory
sys.path.insert(0, os.path.dirname(__file__))

try:
    from content_based import ContentBasedRecommender
    from collaborative import CollaborativeFilteringRecommender
    from hybrid import HybridRecommender
    from evaluation import ModelEvaluator
except ImportError as e:
    print(f"[ERROR] Could not import modules: {e}")
    print("Make sure you are running from the 'src/' directory and that all "
          "dependencies are installed (pip install -r ../requirements.txt).")
    sys.exit(1)


class MovieRecommendationSystem:
    """
    End-to-end Movie Recommendation System.

    Parameters
    ----------
    data_dir : str
        Path to directory containing movies.csv, users.csv, ratings.csv.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        print("Loading data...")
        self.movies_df, self.users_df, self.ratings_df = self._load_data()
        print(f"  {len(self.movies_df)} movies | "
              f"{len(self.users_df)} users | "
              f"{len(self.ratings_df)} ratings")

        print("Building Content-Based recommender...")
        self.content_rec = ContentBasedRecommender(self.movies_df)

        print("Building Collaborative Filtering recommender...")
        self.collab_rec = CollaborativeFilteringRecommender(
            self.ratings_df, self.movies_df
        )

        print("Building Hybrid recommender...")
        self.hybrid_rec = HybridRecommender(
            self.content_rec, self.collab_rec,
            self.movies_df, self.ratings_df
        )

        print("Setting up evaluator...")
        self.evaluator = ModelEvaluator(self.ratings_df, self.movies_df)
        print("System ready!\n")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        """Load CSVs from data_dir.  Generates them if missing."""
        movies_path = os.path.join(self.data_dir, "movies.csv")
        users_path = os.path.join(self.data_dir, "users.csv")
        ratings_path = os.path.join(self.data_dir, "ratings.csv")

        if not all(os.path.exists(p) for p in [movies_path, users_path, ratings_path]):
            print("  Data files not found — generating synthetic data...")
            gen_script = os.path.join(self.data_dir, "generate_data.py")
            if os.path.exists(gen_script):
                import subprocess
                subprocess.run([sys.executable, gen_script], check=True)
            else:
                raise FileNotFoundError(
                    f"Data files not found in {self.data_dir} and "
                    "generate_data.py is missing."
                )

        movies_df = pd.read_csv(movies_path)
        users_df = pd.read_csv(users_path)
        ratings_df = pd.read_csv(ratings_path)
        return movies_df, users_df, ratings_df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend_for_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """
        Generate hybrid recommendations for an existing user.

        Parameters
        ----------
        user_id : int
        n : int  Number of recommendations.

        Returns
        -------
        pd.DataFrame
        """
        return self.hybrid_rec.recommend(user_id=user_id, n=n)

    def recommend_similar(self, movie_title: str, n: int = 10) -> pd.DataFrame:
        """
        Return movies similar to *movie_title* (content-based).

        Parameters
        ----------
        movie_title : str  Exact (case-insensitive) title.
        n : int

        Returns
        -------
        pd.DataFrame
        """
        return self.content_rec.get_recommendations(movie_title, n=n)

    def new_user_onboarding(
        self, preferred_genres: List[str], n: int = 10
    ) -> pd.DataFrame:
        """
        Cold-start recommendations for a new user with no rating history.

        Parameters
        ----------
        preferred_genres : list of str  e.g. ["Action", "Sci-Fi"]
        n : int

        Returns
        -------
        pd.DataFrame
        """
        return self.hybrid_rec.cold_start_recommendations(preferred_genres, n=n)

    def get_trending(self, n: int = 10) -> pd.DataFrame:
        """Return currently trending movies by rating count and quality."""
        return self.hybrid_rec.get_trending(n=n)

    def run_evaluation(self) -> None:
        """Run and print a full model evaluation report."""
        print("\nRunning model evaluation (this may take a moment)...")
        self.evaluator.train_test_split(test_size=0.2)
        rating_report = self.evaluator.evaluate_collaborative(self.collab_rec)
        ranking_report = self.evaluator.evaluate_ranking(self.collab_rec, k=10, n_users=15)
        self.evaluator.print_report(rating_report, ranking_report)

    # ------------------------------------------------------------------
    # Display helper
    # ------------------------------------------------------------------

    @staticmethod
    def display_recommendations(df: pd.DataFrame, title: str) -> None:
        """Pretty-print a recommendations DataFrame with a header."""
        width = 70
        print("\n" + "=" * width)
        print(f"  {title}")
        print("=" * width)
        if df.empty:
            print("  (no recommendations found)")
        else:
            pd.set_option("display.max_colwidth", 35)
            pd.set_option("display.width", width)
            print(df.to_string())
        print("=" * width + "\n")

    # ------------------------------------------------------------------
    # Demo
    # ------------------------------------------------------------------

    def run_demo(self) -> None:
        """Full end-to-end demonstration of all system features."""
        sep = "\n" + "#" * 70

        print(sep)
        print("#  MOVIE RECOMMENDATION SYSTEM — DEMO")
        print("#" * 70)

        # ── 1. Content-Based ──────────────────────────────────────────
        seed_movie = self.movies_df["title"].iloc[0]
        print(f"\n[1] CONTENT-BASED: Movies similar to '{seed_movie}'")
        try:
            cb_recs = self.recommend_similar(seed_movie, n=5)
            self.display_recommendations(cb_recs, f"Similar to '{seed_movie}'")
        except Exception as e:
            print(f"  Content-based error: {e}")

        # ── 2. Explain a recommendation ───────────────────────────────
        if len(self.movies_df) >= 2:
            m1, m2 = self.movies_df["title"].iloc[0], self.movies_df["title"].iloc[1]
            explanation = self.content_rec.explain_recommendation(m1, m2)
            print(f"[2] EXPLANATION: {explanation}\n")

        # ── 3. User-Based CF ──────────────────────────────────────────
        sample_user = int(self.ratings_df["user_id"].iloc[0])
        print(f"[3] USER-BASED CF for User {sample_user}")
        try:
            ub_recs = self.collab_rec.user_based_recommendations(sample_user, n=5)
            self.display_recommendations(ub_recs, f"User-Based CF — User {sample_user}")
        except Exception as e:
            print(f"  User-based CF error: {e}")

        # ── 4. SVD-Based CF ───────────────────────────────────────────
        print(f"[4] SVD (Matrix Factorisation) for User {sample_user}")
        try:
            svd_recs = self.collab_rec.svd_recommendations(sample_user, n=5)
            self.display_recommendations(svd_recs, f"SVD Recommendations — User {sample_user}")
        except Exception as e:
            print(f"  SVD error: {e}")

        # ── 5. Hybrid ─────────────────────────────────────────────────
        print(f"[5] HYBRID RECOMMENDATIONS for User {sample_user}")
        try:
            hybrid_recs = self.recommend_for_user(sample_user, n=5)
            self.display_recommendations(
                hybrid_recs, f"Hybrid Recommendations — User {sample_user}"
            )
        except Exception as e:
            print(f"  Hybrid error: {e}")

        # ── 6. Cold-Start ─────────────────────────────────────────────
        print("[6] COLD-START (new user likes Action and Sci-Fi)")
        try:
            cs_recs = self.new_user_onboarding(["Action", "Sci-Fi"], n=5)
            self.display_recommendations(cs_recs, "Cold-Start Recommendations")
        except Exception as e:
            print(f"  Cold-start error: {e}")

        # ── 7. Trending ───────────────────────────────────────────────
        print("[7] TRENDING MOVIES")
        try:
            trending = self.get_trending(n=5)
            self.display_recommendations(trending, "Trending Movies")
        except Exception as e:
            print(f"  Trending error: {e}")

        # ── 8. Evaluation ─────────────────────────────────────────────
        print("[8] MODEL EVALUATION")
        try:
            self.run_evaluation()
        except Exception as e:
            print(f"  Evaluation error: {e}")

        print("\nDemo complete!")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Resolve the data directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    data_dir = os.path.normpath(data_dir)

    system = MovieRecommendationSystem(data_dir=data_dir)
    system.run_demo()
