"""
CineMatch — Movie Recommendation System
========================================
Main Streamlit application entry point.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
import os

# Allow imports from project root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="CineMatch 🎬",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global dark cinema theme */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0a;
    color: #ffffff;
}
[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
/* Headers */
h1, h2, h3, h4 {
    color: #FFD700 !important;
}
/* Metric cards */
[data-testid="stMetricValue"] {
    color: #FFD700 !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: #aaaaaa !important;
}
/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #E50914, #b0070f);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    padding: 0.5rem 1.5rem;
    transition: transform 0.1s;
}
.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 4px 15px rgba(229,9,20,0.4);
}
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #111111;
    border-radius: 8px;
}
.stTabs [data-baseweb="tab"] {
    color: #aaaaaa;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #FFD700 !important;
    border-bottom: 2px solid #E50914;
}
/* Divider */
hr { border-color: #333; }
/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 8px; }
/* Input widgets */
.stSelectbox label, .stSlider label, .stNumberInput label,
.stMultiSelect label, .stRadio label {
    color: #cccccc !important;
}
/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a0a0a 50%, #0a0a1a 100%);
    border: 1px solid #E50914;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin-bottom: 24px;
}
/* Movie card */
.movie-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    cursor: pointer;
}
.movie-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(229,9,20,0.3);
}
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ────────────────────────────────────────────────────
DATA_DIR = os.path.join(ROOT, "data")


@st.cache_data(show_spinner="📂 Loading movie data…")
def load_data():
    movies_path = os.path.join(DATA_DIR, "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    users_path = os.path.join(DATA_DIR, "users.csv")

    # Auto-generate if not found
    if not os.path.exists(movies_path):
        st.warning("Dataset not found — generating now…")
        from data.generate_data import main as gen
        gen()

    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    users_df = pd.read_csv(users_path) if os.path.exists(users_path) else pd.DataFrame()

    # Normalize column names
    if "avg_rating" not in movies_df.columns and "rating" in movies_df.columns:
        movies_df["avg_rating"] = movies_df["rating"]
    if "description" not in movies_df.columns and "overview" in movies_df.columns:
        movies_df["description"] = movies_df["overview"]

    return movies_df, ratings_df, users_df


@st.cache_resource(show_spinner="🧠 Training recommendation models…")
def load_models(movies_csv_hash: str):
    """Load and fit all recommendation models. Cached by data hash."""
    movies_df, ratings_df, _ = load_data()

    sys.path.insert(0, os.path.join(ROOT, "src"))
    from content_based import ContentBasedRecommender
    from collaborative import CollaborativeFilteringRecommender
    from hybrid import HybridRecommender

    content_rec = ContentBasedRecommender(movies_df)
    collab_rec = CollaborativeFilteringRecommender(ratings_df, movies_df)
    hybrid_rec = HybridRecommender(content_rec, collab_rec, movies_df, ratings_df)
    return content_rec, collab_rec, hybrid_rec


# ── Helpers ──────────────────────────────────────────────────────────────────
from utils.helpers import (
    get_genre_emoji, get_genre_color, get_star_rating,
    format_runtime, format_number, get_trending_movies
)


def genre_badge(genre: str) -> str:
    return (
        f'<span style="background:{get_genre_color(genre)};color:#fff;'
        f'padding:2px 9px;border-radius:12px;font-size:11px;margin:1px;display:inline-block">'
        f'{get_genre_emoji(genre)} {genre}</span>'
    )


def render_movie_card(row: pd.Series, show_score: bool = False,
                       score_label: str = "", score: float = 0.0) -> str:
    genres = [g.strip() for g in str(row.get("genres", "")).split("|") if g.strip()]
    badges = "".join(genre_badge(g) for g in genres[:2])
    rating = row.get("avg_rating", row.get("rating", 0))
    votes = row.get("votes", row.get("vote_count", 0))
    score_html = (
        f'<div style="color:#00B4DB;font-size:12px;margin-top:4px"><b>{score_label}:</b> {score:.3f}</div>'
        if show_score else ""
    )
    return f"""
    <div class="movie-card" style="min-height:230px">
        <div style="font-size:36px;margin-bottom:4px">🎬</div>
        <div style="color:#fff;font-weight:700;font-size:13px;line-height:1.3;
                    height:40px;overflow:hidden;margin-bottom:4px">{row.get('title','')}</div>
        <div style="color:#888;font-size:11px">{int(row.get('year',0)) if row.get('year') else ''}</div>
        <div style="margin:5px 0">{badges}</div>
        <div style="color:#FFD700;font-size:12px">{get_star_rating(float(rating) if rating else 0)}</div>
        <div style="color:#FFD700;font-size:12px"><b>{rating}/10</b></div>
        <div style="color:#888;font-size:10px">👁 {format_number(votes)} votes</div>
        {score_html}
    </div>
    """


# ── Page: Home ───────────────────────────────────────────────────────────────
def page_home(movies_df, ratings_df, users_df, hybrid_rec):
    # Hero
    st.markdown("""
    <div class="hero-banner">
        <div style="font-size:4rem">🎬</div>
        <h1 style="font-size:3rem;margin:8px 0;color:#FFD700!important">CineMatch</h1>
        <p style="color:#aaa;font-size:1.2rem">Your AI-Powered Movie Recommendation Engine</p>
        <p style="color:#666;font-size:0.9rem">
            Content-Based · Collaborative Filtering · SVD · Hybrid Models
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🎬 Movies", f"{len(movies_df):,}")
    c2.metric("👤 Users", f"{ratings_df['user_id'].nunique():,}")
    c3.metric("⭐ Ratings", f"{len(ratings_df):,}")
    c4.metric("📊 Avg Rating", f"{movies_df['avg_rating'].mean():.1f}/10")
    all_genres = set(g for gs in movies_df["genres"].dropna() for g in gs.split("|"))
    c5.metric("🎭 Genres", len(all_genres))

    st.divider()

    # Trending Movies
    st.markdown("### 🔥 Trending Right Now")
    try:
        trending = get_trending_movies(movies_df, ratings_df, n=10)
        cols = st.columns(5)
        for i, (_, row) in enumerate(trending.head(10).iterrows()):
            with cols[i % 5]:
                st.markdown(render_movie_card(row), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load trending: {e}")

    st.divider()

    # Top Rated by Genre
    st.markdown("### 🏆 Top Rated by Genre")
    all_genres_list = sorted(all_genres)
    tabs = st.tabs([f"{get_genre_emoji(g)} {g}" for g in all_genres_list[:10]])
    for tab, genre in zip(tabs, all_genres_list[:10]):
        with tab:
            genre_movies = movies_df[
                movies_df["genres"].str.contains(genre, na=False)
            ].nlargest(5, "avg_rating")
            if genre_movies.empty:
                st.info("No movies in this genre.")
                continue
            cols = st.columns(5)
            for i, (_, row) in enumerate(genre_movies.iterrows()):
                with cols[i % 5]:
                    st.markdown(render_movie_card(row), unsafe_allow_html=True)

    st.divider()

    # Quick stats charts
    st.markdown("### 📊 Quick Stats")
    col_a, col_b = st.columns(2)
    with col_a:
        genre_counts = {}
        for gs in movies_df["genres"].dropna():
            for g in gs.split("|"):
                genre_counts[g.strip()] = genre_counts.get(g.strip(), 0) + 1
        gdf = pd.DataFrame({"Genre": list(genre_counts.keys()),
                             "Count": list(genre_counts.values())}).sort_values("Count", ascending=False).head(12)
        fig = px.bar(gdf, x="Genre", y="Count", color="Count",
                     color_continuous_scale="Reds", title="Movies by Genre")
        fig.update_layout(plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a",
                          font_color="#fff", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        yr = movies_df["year"].value_counts().sort_index().reset_index()
        yr.columns = ["Year", "Movies"]
        fig2 = px.area(yr, x="Year", y="Movies", title="Movies per Year",
                       color_discrete_sequence=["#E50914"])
        fig2.update_traces(fill="tozeroy", fillcolor="rgba(229,9,20,0.25)")
        fig2.update_layout(plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a",
                            font_color="#fff", height=350)
        st.plotly_chart(fig2, use_container_width=True)


# ── Page: About ──────────────────────────────────────────────────────────────
def page_about():
    st.markdown("## ℹ️ About CineMatch")
    st.markdown("""
    **CineMatch** is a full-featured Movie Recommendation System built with Python and Streamlit.

    ### 🧠 Recommendation Algorithms
    | Algorithm | Description | Key Library |
    |---|---|---|
    | **Content-Based** | TF-IDF on genres/director/cast/overview → Cosine Similarity | scikit-learn |
    | **SVD (Matrix Factorization)** | Latent factor model decomposing user-item matrix | scikit-learn TruncatedSVD |
    | **User-Based CF** | Finds similar users (Pearson correlation) → aggregates ratings | scipy, numpy |
    | **Item-Based CF** | Finds similar items (Cosine similarity) → weighted predictions | scikit-learn |
    | **Hybrid** | Weighted blend of content (40%) + collaborative (60%) signals | custom |

    ### 📐 Evaluation Metrics
    | Metric | Formula | What it measures |
    |---|---|---|
    | **RMSE** | √(Σ(actual−pred)²/n) | Rating prediction accuracy |
    | **MAE** | Σ|actual−pred|/n | Rating prediction accuracy |
    | **Precision@K** | Hits / K | Fraction of top-K that are relevant |
    | **Recall@K** | Hits / Relevant | Fraction of relevant items in top-K |
    | **NDCG@K** | DCG / IDCG | Rank-weighted relevance quality |
    | **Coverage** | Unique recs / Total items | Catalog diversity |

    ### 📦 Dataset
    - **450 movies** across 20 genres (1990–2024)
    - **200 users** with preference profiles
    - **4,000+ ratings** (0.5–5.0 scale, biased toward true movie quality)

    ### 🛠️ Tech Stack
    - **Frontend**: Streamlit, Plotly
    - **ML**: scikit-learn, scipy, numpy, pandas
    - **Models**: TF-IDF, TruncatedSVD, Pearson/Cosine CF
    - **Data**: Synthetic dataset (fully reproducible with seed=42)

    ### 📁 Project Structure
    ```
    movie-recommender/
    ├── streamlit_app.py        ← Main Streamlit entry point
    ├── src/
    │   ├── content_based.py    ← Content-Based Recommender
    │   ├── collaborative.py    ← CF: User-Based, Item-Based, SVD
    │   ├── hybrid.py           ← Hybrid Recommender
    │   └── evaluation.py       ← RMSE, MAE, Precision/Recall/NDCG
    ├── data/
    │   ├── generate_data.py    ← Dataset generator (450 movies)
    │   ├── movies.csv
    │   ├── ratings.csv
    │   └── users.csv
    ├── pages/
    │   ├── search_recommend.py ← Search & recommend page
    │   ├── browse_genre.py     ← Genre browser page
    │   └── analytics.py        ← Analytics & metrics page
    ├── utils/
    │   └── helpers.py          ← Formatting, filtering, search
    └── requirements.txt
    ```
    """)


# ── Sidebar navigation ────────────────────────────────────────────────────────
def main():
    # Load data
    movies_df, ratings_df, users_df = load_data()

    # Store in session state for access in sub-pages
    st.session_state["movies_df"] = movies_df
    st.session_state["ratings_df"] = ratings_df
    st.session_state["users_df"] = users_df

    # Create a stable hash for model caching
    data_hash = str(len(movies_df)) + "_" + str(len(ratings_df))

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0">
            <div style="font-size:2.5rem">🎬</div>
            <div style="color:#FFD700;font-weight:800;font-size:1.3rem">CineMatch</div>
            <div style="color:#666;font-size:0.75rem">AI Movie Recommender</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        nav = st.radio(
            "Navigate",
            ["🏠 Home", "🔍 Search & Recommend", "🎭 Browse by Genre",
             "📊 Analytics & Metrics", "ℹ️ About"],
            label_visibility="collapsed"
        )

        st.divider()
        st.markdown("**📊 Dataset Stats**")
        st.caption(f"🎬 {len(movies_df):,} movies")
        st.caption(f"👤 {ratings_df['user_id'].nunique():,} users")
        st.caption(f"⭐ {len(ratings_df):,} ratings")

        st.divider()
        st.markdown("**🧠 Algorithms**")
        for algo, desc in [
            ("Content-Based", "TF-IDF + Cosine"),
            ("Collaborative (SVD)", "Matrix Factorization"),
            ("User/Item-Based CF", "Pearson / Cosine"),
            ("Hybrid", "Weighted Ensemble"),
        ]:
            st.caption(f"• **{algo}**: {desc}")

        st.divider()
        st.caption("Built with ❤️ using Streamlit, scikit-learn & Plotly")

    # Load models (cached after first run)
    if nav in ["🔍 Search & Recommend", "🏠 Home"]:
        with st.spinner("Loading models…"):
            content_rec, collab_rec, hybrid_rec = load_models(data_hash)
    else:
        try:
            content_rec, collab_rec, hybrid_rec = load_models(data_hash)
        except Exception:
            content_rec, collab_rec, hybrid_rec = None, None, None

    # Route pages
    if nav == "🏠 Home":
        page_home(movies_df, ratings_df, users_df, hybrid_rec)

    elif nav == "🔍 Search & Recommend":
        from pages.search_recommend import show_search_recommend_page
        show_search_recommend_page(movies_df, content_rec, collab_rec, hybrid_rec)

    elif nav == "🎭 Browse by Genre":
        from pages.browse_genre import show_browse_genre_page
        show_browse_genre_page(movies_df)

    elif nav == "📊 Analytics & Metrics":
        from pages.analytics import show_analytics_page
        show_analytics_page(movies_df, ratings_df, collab_rec)

    elif nav == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
