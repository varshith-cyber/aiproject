"""
Main Pipeline — Movie Recommendation System
Run this file to train all models and get recommendations.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data.sample_data import save_data
from utils.preprocessing import DataPreprocessor
from models.content_based import ContentBasedRecommender
from models.collaborative_filtering import UserBasedCF, ItemBasedCF, SVDRecommender
from models.hybrid import HybridRecommender
from utils.evaluation import ModelEvaluator

BANNER = """
╔══════════════════════════════════════════════════════╗
║       🎬  Movie Recommendation System  🎬            ║
║   Content-Based | Collaborative | Hybrid | SVD       ║
╚══════════════════════════════════════════════════════╝
"""


def print_section(title: str):
    print(f"\n{'─'*54}")
    print(f"  {title}")
    print(f"{'─'*54}")


def print_recommendations(df: pd.DataFrame, score_col: str = None):
    score_col = score_col or ("hybrid_score" if "hybrid_score" in df.columns else "predicted_rating")
    for i, row in df.iterrows():
        score = f"  [{row.get(score_col, 0):.3f}]" if score_col in row else ""
        year  = int(row.get("year", 0)) if "year" in row else ""
        print(f"  {i+1}. {row.get('title','?')} ({year}){score}")
        if "genres" in row:
            print(f"     Genre: {row['genres']}  |  IMDb: {row.get('rating','')}")


def main():
    print(BANNER)

    # ── 1. Generate / load data ──────────────────────────────────────────────
    print_section("1. Loading Data")
    movies_df, users_df, ratings_df = save_data(output_dir="data")

    prep = DataPreprocessor()
    prep.movies_df  = movies_df
    prep.ratings_df = ratings_df

    stats = prep.get_stats()
    print(f"  Movies   : {stats['n_movies']}")
    print(f"  Users    : {stats['n_users']}")
    print(f"  Ratings  : {stats['n_ratings']}")
    print(f"  Avg ⭐   : {stats['avg_rating']}")
    print(f"  Sparsity : {stats['sparsity']:.1%}")

    # ── 2. Feature Engineering ───────────────────────────────────────────────
    print_section("2. Feature Engineering")
    content_features   = prep.build_content_features()
    user_movie_matrix  = prep.build_user_movie_matrix()
    train_df, test_df  = prep.train_test_split()
    print(f"  Content feature shape : {content_features.shape}")
    print(f"  User-Movie matrix     : {user_movie_matrix.shape}")
    print(f"  Train/Test split      : {len(train_df)} / {len(test_df)}")

    # ── 3. Train Models ───────────────────────────────────────────────────────
    print_section("3. Training Models")

    cbf = ContentBasedRecommender().fit(content_features, movies_df)

    train_matrix = train_df.pivot_table(
        index="user_id", columns="movie_id", values="rating"
    )
    ucf = UserBasedCF(k_neighbours=5).fit(train_matrix)
    icf = ItemBasedCF(k_neighbours=10).fit(train_matrix)
    svd = SVDRecommender(n_factors=15).fit(train_matrix)
    hyb = HybridRecommender(cbf, icf, svd, weights={"content": 0.35, "item_cf": 0.35, "svd": 0.30})

    # ── 4. Sample Recommendations ─────────────────────────────────────────────
    USER_ID   = 1
    MOVIE_ID  = 7   # Inception

    print_section(f"4a. Content-Based: Movies similar to 'Inception'")
    recs_cbf = cbf.recommend_by_movie(MOVIE_ID, top_n=5)
    print_recommendations(recs_cbf, "similarity")

    print_section(f"4b. User-Based CF: Recommendations for User {USER_ID}")
    recs_ucf = ucf.recommend(USER_ID, top_n=5)
    recs_ucf = recs_ucf.merge(movies_df, on="movie_id", how="left")
    print_recommendations(recs_ucf, "predicted_rating")

    print_section(f"4c. Item-Based CF: Recommendations for User {USER_ID}")
    recs_icf = icf.recommend(USER_ID, top_n=5)
    recs_icf = recs_icf.merge(movies_df, on="movie_id", how="left")
    print_recommendations(recs_icf, "predicted_rating")

    print_section(f"4d. SVD: Recommendations for User {USER_ID}")
    recs_svd = svd.recommend(USER_ID, top_n=5)
    recs_svd = recs_svd.merge(movies_df, on="movie_id", how="left")
    print_recommendations(recs_svd, "predicted_rating")

    print_section(f"4e. Hybrid: Recommendations for User {USER_ID}")
    recs_hyb = hyb.recommend(USER_ID, ratings_df, movies_df, top_n=5)
    print_recommendations(recs_hyb, "hybrid_score")

    # ── 5. Evaluation ────────────────────────────────────────────────────────
    print_section("5. Model Evaluation")
    evaluator = ModelEvaluator(test_df)

    for name, model in [("UserCF", ucf), ("ItemCF", icf), ("SVD", svd)]:
        pred_fn = lambda uid, mid, m=model: m.predict_rating(uid, mid)
        metrics = evaluator.evaluate_rating_prediction(pred_fn)
        print(f"  [{name}]  RMSE={metrics['RMSE']}  MAE={metrics['MAE']}")

    # Ranking metrics for SVD
    def svd_rec_fn(uid):
        try:
            return svd.recommend(uid, top_n=5)["movie_id"].tolist()
        except Exception:
            return []

    ranking = evaluator.evaluate_ranking(svd_rec_fn, ratings_df, k=5)
    print(f"\n  [SVD Ranking@5]")
    for k, v in ranking.items():
        print(f"    {k} = {v}")

    print_section("✅ Done!")
    print("  Run the interactive dashboard:")
    print("    python app.py\n")


if __name__ == "__main__":
    main()
