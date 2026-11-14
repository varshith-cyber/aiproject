"""
Browse by Genre page for CineMatch Streamlit dashboard.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.helpers import (
    get_genre_emoji, get_genre_color, get_star_rating,
    format_runtime, format_number, filter_movies
)


def _movie_tile(row: pd.Series):
    genres = str(row.get("genres", "")).split("|")
    badges = " ".join(
        f'<span style="background:{get_genre_color(g)};color:#fff;padding:1px 7px;border-radius:10px;font-size:11px">{g}</span>'
        for g in genres if g
    )
    rating = row.get("avg_rating", 0)
    votes = row.get("votes", 0)
    return f"""
    <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:14px;margin:5px;height:230px;overflow:hidden">
        <div style="font-size:30px;text-align:center">🎬</div>
        <div style="color:#fff;font-weight:700;font-size:13px;text-align:center;margin:5px 0;line-height:1.3;height:36px;overflow:hidden">{row.get('title','')}</div>
        <div style="color:#888;font-size:11px;text-align:center">{int(row.get('year',0)) if row.get('year') else ''} · {format_runtime(row.get('runtime_min',0))}</div>
        <div style="text-align:center;margin:5px 0;height:24px;overflow:hidden">{badges}</div>
        <div style="color:#FFD700;text-align:center;font-size:12px">{get_star_rating(float(rating))} {rating}/10</div>
        <div style="color:#888;font-size:11px;text-align:center">👁 {format_number(votes)} votes</div>
        <div style="color:#aaa;font-size:11px;text-align:center;margin-top:3px">🎬 {str(row.get('director',''))[:25]}</div>
    </div>
    """


def show_browse_genre_page(movies_df: pd.DataFrame):
    st.markdown("## 🎭 Browse by Genre")

    # Sidebar-like filters inside the page
    all_genres = sorted(set(
        g for genres_str in movies_df["genres"].dropna()
        for g in genres_str.split("|") if g.strip()
    ))

    col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns([2, 1.5, 1.5, 1.5, 1.5])
    with col_f1:
        selected_genres = st.multiselect(
            "🎭 Genres", all_genres,
            default=[], placeholder="All genres",
            key="browse_genres"
        )
    with col_f2:
        year_range = st.slider("📅 Year Range", 1990, 2024, (1990, 2024), key="browse_year")
    with col_f3:
        min_rating = st.slider("⭐ Min Rating", 0.0, 10.0, 0.0, 0.5, key="browse_rating")
    with col_f4:
        languages = ["All"] + sorted(movies_df["language"].dropna().unique().tolist())
        language = st.selectbox("🌍 Language", languages, key="browse_lang")
    with col_f5:
        sort_by = st.selectbox("📊 Sort By",
                               ["avg_rating", "year", "title", "votes"],
                               format_func=lambda x: {
                                   "avg_rating": "⭐ Rating", "year": "📅 Year",
                                   "title": "🔤 Title", "votes": "👁 Votes"
                               }[x],
                               key="browse_sort")

    # Apply filters
    filtered = filter_movies(
        movies_df,
        genres=selected_genres if selected_genres else None,
        min_year=year_range[0], max_year=year_range[1],
        min_rating=min_rating if min_rating > 0 else None,
        language=language
    )
    ascending = sort_by == "title"
    filtered = filtered.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    # Stats bar
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎬 Movies Found", len(filtered))
    c2.metric("⭐ Avg Rating",
              f"{filtered['avg_rating'].mean():.1f}" if not filtered.empty else "—")
    c3.metric("📅 Year Span",
              f"{int(filtered['year'].min())}–{int(filtered['year'].max())}" if not filtered.empty else "—")
    top_dir = filtered["director"].value_counts().idxmax() if not filtered.empty else "—"
    c4.metric("🎬 Top Director", str(top_dir)[:22])

    st.divider()

    if filtered.empty:
        st.warning("No movies match your filters. Try relaxing some criteria.")
        return

    # Pagination
    page_size = 20
    total_pages = max(1, (len(filtered) - 1) // page_size + 1)
    page_num = st.number_input(
        f"Page (1–{total_pages})", min_value=1, max_value=total_pages, value=1,
        key="browse_page"
    )
    start_idx = (page_num - 1) * page_size
    page_df = filtered.iloc[start_idx: start_idx + page_size]

    st.markdown(f"**Showing {start_idx + 1}–{min(start_idx + page_size, len(filtered))} of {len(filtered)} movies**")

    # Movie grid
    cols_per_row = 5
    for row_start in range(0, len(page_df), cols_per_row):
        cols = st.columns(cols_per_row)
        for ci, ri in enumerate(range(row_start, min(row_start + cols_per_row, len(page_df)))):
            with cols[ci]:
                st.markdown(_movie_tile(page_df.iloc[ri]), unsafe_allow_html=True)

    # Genre analytics panel
    if selected_genres:
        st.divider()
        st.markdown("### 📊 Genre Insights")
        col_a, col_b = st.columns(2)
        with col_a:
            dir_counts = filtered["director"].value_counts().head(10).reset_index()
            dir_counts.columns = ["director", "count"]
            fig = px.bar(dir_counts, x="count", y="director", orientation="h",
                         color="count", color_continuous_scale="Reds",
                         title="Top Directors in Selection")
            fig.update_layout(plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a",
                               font_color="#fff", height=350,
                               yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            yr_counts = filtered["year"].value_counts().sort_index().reset_index()
            yr_counts.columns = ["year", "count"]
            fig2 = px.area(yr_counts, x="year", y="count",
                           title="Movies per Year", color_discrete_sequence=["#E50914"])
            fig2.update_layout(plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a",
                                font_color="#fff", height=350)
            st.plotly_chart(fig2, use_container_width=True)
