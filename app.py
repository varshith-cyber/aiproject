"""
Flask Web App — Interactive Movie Recommendation Dashboard
Run: python app.py  →  visit http://localhost:5000
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify
import pandas as pd

from data.sample_data import save_data, MOVIES, USER_PROFILES
from utils.preprocessing import DataPreprocessor
from models.content_based import ContentBasedRecommender
from models.collaborative_filtering import UserBasedCF, ItemBasedCF, SVDRecommender
from models.hybrid import HybridRecommender

app = Flask(__name__, template_folder="static")

# ── Bootstrap ─────────────────────────────────────────────────────────────────
print("🎬  Initialising recommendation engine …")
movies_df, users_df, ratings_df = save_data(output_dir="data")

prep = DataPreprocessor()
prep.movies_df  = movies_df
prep.ratings_df = ratings_df
content_features  = prep.build_content_features()
user_movie_matrix = prep.build_user_movie_matrix()

cbf = ContentBasedRecommender().fit(content_features, movies_df)
icf = ItemBasedCF(k_neighbours=10).fit(user_movie_matrix)
svd = SVDRecommender(n_factors=15).fit(user_movie_matrix)
hyb = HybridRecommender(cbf, icf, svd)
print("✅  Engine ready.")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
        movies=MOVIES,
        users=USER_PROFILES,
    )


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data      = request.json
    algo      = data.get("algo", "hybrid")
    user_id   = int(data.get("user_id", 1))
    movie_id  = int(data.get("movie_id", 7))
    top_n     = int(data.get("top_n", 5))

    try:
        if algo == "content":
            recs = cbf.recommend_by_movie(movie_id, top_n=top_n)
            recs = recs.rename(columns={"similarity": "score"})

        elif algo == "usercf":
            ucf = UserBasedCF(k_neighbours=5).fit(user_movie_matrix)
            raw = ucf.recommend(user_id, top_n=top_n)
            recs = raw.merge(movies_df, on="movie_id", how="left")
            recs = recs.rename(columns={"predicted_rating": "score"})

        elif algo == "itemcf":
            raw  = icf.recommend(user_id, top_n=top_n)
            recs = raw.merge(movies_df, on="movie_id", how="left")
            recs = recs.rename(columns={"predicted_rating": "score"})

        elif algo == "svd":
            raw  = svd.recommend(user_id, top_n=top_n)
            recs = raw.merge(movies_df, on="movie_id", how="left")
            recs = recs.rename(columns={"predicted_rating": "score"})

        else:   # hybrid
            recs = hyb.recommend(user_id, ratings_df, movies_df, top_n=top_n)
            recs = recs.rename(columns={"hybrid_score": "score"})

        return jsonify({
            "status": "ok",
            "algo": algo,
            "recommendations": recs[["movie_id","title","genres","year","rating","score"]].to_dict("records"),
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/stats")
def stats():
    return jsonify(prep.get_stats())


@app.route("/api/movies")
def movies():
    return jsonify(MOVIES)


@app.route("/api/users")
def users():
    return jsonify(USER_PROFILES)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
