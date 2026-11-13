"""Optional web UI:  streamlit run app.py"""
import streamlit as st

from recsys.models import ContentBased
from recsys.pipeline import build_models, prepare

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")

synthetic = st.sidebar.checkbox("Use synthetic data (offline)", value=False)


@st.cache_resource(show_spinner="Loading data and training models...")
def load(synthetic):
    ds, _, _ = prepare(synthetic)
    models = build_models(ds)
    for m in models.values():
        m.fit(ds.ratings, ds.n_users, ds.n_items)
    return ds, models


ds, models = load(synthetic)
tab1, tab2 = st.tabs(["Recommend for a user", "Similar movies"])

with tab1:
    model_name = st.selectbox("Model", list(models), index=list(models).index("hybrid"))
    user_id = st.selectbox("User ID", ds.user_ids)
    n = st.slider("How many?", 5, 25, 10)
    u = int(list(ds.user_ids).index(user_id))
    items, preds = models[model_name].recommend(u, n)
    st.dataframe(ds.movies.loc[items, ["title", "genres"]].assign(predicted_rating=preds.round(2)),
                 use_container_width=True)

with tab2:
    title = st.selectbox("Pick a movie", ds.movies.title)
    idx = int(ds.movies.index[ds.movies.title == title][0])
    items, sims = models["content"].similar_items(idx, 10)
    st.dataframe(ds.movies.loc[items, ["title", "genres"]].assign(similarity=sims.round(3)),
                 use_container_width=True)
