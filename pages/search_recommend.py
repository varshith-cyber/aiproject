"""
Search & Recommend page for CineMatch Streamlit dashboard.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import (
    get_genre_emoji, get_genre_color, get_star_rating,
    format_runtime, format_number, get_user_genre_profile
)


def _movie_card_html(row: pd.Series, score_label: str = "", score_val: float = 0.0) -> str:
    genres = str(row.get("genres", "")).split("|")
    badges = " ".join(
        f'<span style="background:{get_genre_color(g)};color:#fff;padding:2px 8px;border-radius:12px;font-size:11px;margin:1px">{get_genre_emoji(g)} {g}</span>'
        for g in genres if g
    )
    rating = row.get("avg_rating", 0)
    stars = get_star_rating(float(rating) if rating else 0)
    score_html = f'<div style="color:#FFD700;font-size:13px;margin-top:4px"><b>{score_label}:</b> {score_val:.3f}</div>' if score_label else ""
    return f"""
    <div style="background:#1a1a2e;border:1px solid #333;border-radius:12px;padding:14px;margin:4px;min-height:220px">
        <div style="font-size:28px;text-align:center">🎬</div>
        <div style="color:#fff;font-weight:700;font-size:14px;text-align:center;margin:6px 0;line-height:1.3">{row.get('title','')}</div>
        <div style="color:#aaa;font-size:12px;text-align:center">{int(row.get('year',0)) if row.get('year') else ''}</div>
        <div style="text-align:center;margin:6px 0">{badges}</div>
        <div style="color:#FFD700;font-size:13px;text-align:center">{stars} {rating}/10</div>
        {score_html}
        <div style="color:#888;font-size:11px;text-align:center;margin-top:4px">🎬 {row.get('director','')}</div>
    </div>
    """


def show_search_recommend_page(movies_df, content_rec, collab_rec, hybrid_rec):
    st.markdown("## 🔍 Search & Recommend")
    tab1, tab2, tab3 = st.tabs(["🎬 Movie-Based", "👤 User-Based", "🆕 New User (Cold Start)"])

    # ── Tab 1: Movie-Based ──────────────────────────────────────────────────
    with tab1:
        col_a, col_b, col_c = st.columns([3, 1, 1])
        with col_a:
            all_titles = sorted(movies_df["title"].tolist())
            selected_title = st.selectbox("🎬 Select a Movie", all_titles,
                                          help="Pick a seed movie to find similar ones")
        with col_b:
            n_recs = st.slider("# Results", 5, 20, 10, key="cb_n")
        with col_c:
            algo = st.radio("Algorithm", ["Content-Based", "Hybrid"], key="cb_algo")

        if st.button("🚀 Get Recommendations", key="btn_movie"):
            with st.spinner("Finding similar movies…"):
                try:
                    if algo == "Content-Based":
                        recs = content_rec.get_recommendations(selected_title, n=n_recs)
                        score_col = "similarity_score"
                        score_label = "Similarity"
                    else:
                        # Hybrid: use user 1 as proxy + movie seed
                        recs = hybrid_rec.recommend(1, movie_title=selected_title, n=n_recs)
                        score_col = "hybrid_score" if "hybrid_score" in recs.columns else None
                        score_label = "Hybrid Score"

                    if recs.empty:
                        st.warning("No recommendations found.")
                    else:
                        # Movie info panel
                        seed = movies_df[movies_df["title"] == selected_title]
                        if not seed.empty:
                            s = seed.iloc[0]
                            with st.expander(f"📋 About: {selected_title}", expanded=True):
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("⭐ Rating", f"{s.get('avg_rating','?')}/10")
                                c2.metric("📅 Year", str(s.get('year','')))
                                c3.metric("⏱ Runtime", format_runtime(s.get('runtime_min',0)))
                                c4.metric("🌍 Language", str(s.get('language','')))
                                st.markdown(f"**Director:** {s.get('director','')}")
                                st.markdown(f"**Cast:** {str(s.get('cast','')).replace('|', ', ')}")
                                st.markdown(f"**Overview:** {s.get('description','')}")

                        st.markdown(f"### 🎯 Top {len(recs)} Recommendations")
                        # Score bar chart
                        if score_col and score_col in recs.columns:
                            fig = px.bar(
                                recs.head(10), x=score_col, y="title",
                                orientation="h", color=score_col,
                                color_continuous_scale="Reds",
                                title=f"{score_label} Score",
                                labels={"title": "Movie", score_col: score_label}
                            )
                            fig.update_layout(
                                plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a",
                                font_color="#ffffff", height=360,
                                yaxis={"autorange": "reversed"}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Cards grid
                        cols_per_row = 4
                        for row_start in range(0, min(len(recs), n_recs), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for ci, ri in enumerate(range(row_start, min(row_start + cols_per_row, len(recs)))):
                                r = recs.iloc[ri]
                                sv = float(r[score_col]) if score_col and score_col in recs.columns else 0.0
                                with cols[ci]:
                                    st.markdown(_movie_card_html(r, score_label, sv),
                                                unsafe_allow_html=True)

                        # Explanation
                        with st.expander("💡 How does this work?"):
                            if algo == "Content-Based":
                                st.markdown("""
**Content-Based Filtering** analyzes the features of **{}** (genres, director, cast, description)
and uses **TF-IDF Vectorization** + **Cosine Similarity** to find movies with the most
similar feature profiles. Higher similarity score = more similar content.
                                """.format(selected_title))
                            else:
                                st.markdown("""
**Hybrid Recommender** blends:
- 🎭 **Content signal** (40%) – TF-IDF cosine similarity to the seed movie
- 👥 **Collaborative signal** (60%) – SVD-predicted ratings from similar users

Both scores are min-max normalized then combined with a weighted sum.
                                """)
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Tab 2: User-Based ───────────────────────────────────────────────────
    with tab2:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            user_id = st.number_input("👤 User ID", min_value=1, max_value=200,
                                      value=1, step=1, key="user_id_input")
            n_user_recs = st.slider("# Recommendations", 5, 20, 10, key="user_n")
            user_algo = st.radio("Algorithm",
                                 ["SVD (Matrix Factorization)", "User-Based CF",
                                  "Item-Based CF", "Hybrid"],
                                 key="user_algo")
        with col_b:
            # Genre profile radar chart
            profile = get_user_genre_profile(int(user_id),
                                             st.session_state.ratings_df,
                                             movies_df)
            if profile:
                genres_p = list(profile.keys())
                scores_p = list(profile.values())
                fig_radar = go.Figure(go.Scatterpolar(
                    r=scores_p + [scores_p[0]],
                    theta=genres_p + [genres_p[0]],
                    fill="toself",
                    line_color="#E50914",
                    fillcolor="rgba(229,9,20,0.3)"
                ))
                fig_radar.update_layout(
                    polar=dict(bgcolor="#1a1a1a",
                               radialaxis=dict(visible=True, range=[0, 5])),
                    paper_bgcolor="#0a0a0a", font_color="#fff",
                    title=f"Genre Preferences — User {user_id}",
                    height=320
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        if st.button("🎯 Get My Recommendations", key="btn_user"):
            with st.spinner("Personalizing recommendations…"):
                try:
                    if user_algo == "SVD (Matrix Factorization)":
                        recs = collab_rec.svd_recommendations(int(user_id), n=n_user_recs)
                        score_col = "svd_predicted_rating"
                    elif user_algo == "User-Based CF":
                        recs = collab_rec.user_based_recommendations(int(user_id), n=n_user_recs)
                        score_col = "predicted_rating"
                    elif user_algo == "Item-Based CF":
                        recs = collab_rec.item_based_recommendations(int(user_id), n=n_user_recs)
                        score_col = "predicted_rating"
                    else:
                        recs = hybrid_rec.recommend(int(user_id), n=n_user_recs)
                        score_col = "hybrid_score"

                    # Watch history
                    user_hist = st.session_state.ratings_df[
                        st.session_state.ratings_df["user_id"] == int(user_id)
                    ].sort_values("rating", ascending=False).head(8)
                    hist_merged = user_hist.merge(movies_df[["movie_id", "title", "genres", "year"]],
                                                   on="movie_id", how="left")

                    with st.expander(f"📚 Watch History — User {user_id} ({len(user_hist)} recent ratings)"):
                        if not hist_merged.empty:
                            st.dataframe(
                                hist_merged[["title", "genres", "year", "rating"]].rename(
                                    columns={"rating": "Your Rating"}
                                ),
                                use_container_width=True, hide_index=True
                            )

                    if not recs.empty:
                        st.markdown(f"### 🌟 Personalized Picks for User {user_id}")
                        # Predicted rating chart
                        if score_col in recs.columns:
                            fig_bar = px.bar(
                                recs.head(10), x=score_col, y="title",
                                orientation="h", color=score_col,
                                color_continuous_scale="Reds",
                                title="Predicted Ratings",
                                labels={"title": "Movie", score_col: "Predicted Rating"}
                            )
                            fig_bar.update_layout(
                                plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a",
                                font_color="#fff", height=360,
                                yaxis={"autorange": "reversed"}
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                        cols_per_row = 4
                        for row_start in range(0, min(len(recs), n_user_recs), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for ci, ri in enumerate(range(row_start, min(row_start + cols_per_row, len(recs)))):
                                r = recs.iloc[ri]
                                sv = float(r[score_col]) if score_col in recs.columns else 0.0
                                with cols[ci]:
                                    st.markdown(_movie_card_html(r, "Predicted", sv),
                                                unsafe_allow_html=True)
                    else:
                        st.warning("No recommendations could be generated for this user.")
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")

    # ── Tab 3: Cold Start ───────────────────────────────────────────────────
    with tab3:
        st.markdown("### 🆕 New User? Tell us what you like!")
        all_genres = sorted(set(
            g for genres_str in movies_df["genres"].dropna()
            for g in genres_str.split("|") if g.strip()
        ))
        pref_genres = st.multiselect(
            "Select your favorite genres",
            all_genres,
            default=["Action", "Drama"],
            key="cold_genres"
        )
        n_cold = st.slider("# Recommendations", 5, 20, 10, key="cold_n")
        if st.button("🎬 Show Recommendations", key="btn_cold"):
            if not pref_genres:
                st.warning("Please select at least one genre.")
            else:
                with st.spinner("Finding movies you'll love…"):
                    try:
                        recs = hybrid_rec.cold_start_recommendations(pref_genres, n=n_cold)
                        if recs.empty:
                            st.warning("No results found. Try different genres.")
                        else:
                            st.markdown(f"### 🎯 Top Picks for {', '.join(pref_genres)} fans")
                            cols_per_row = 4
                            for row_start in range(0, len(recs), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for ci, ri in enumerate(range(row_start, min(row_start + cols_per_row, len(recs)))):
                                    r = recs.iloc[ri]
                                    with cols[ci]:
                                        st.markdown(_movie_card_html(r), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
