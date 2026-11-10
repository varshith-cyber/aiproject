import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Movie Dashboard", layout="wide")
st.title("Movie Box Office Analytics Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("/Users/sasidharmamidi/Documents/movie_boxoffice_dataset/Movie Dataset-Table 1.csv")
    return df

df = load_data()

st.sidebar.header("Controls")

numeric_cols = [
    'Production Budget ($M)', 'WW Box Office Gross ($M)', 'Sequel Number',
    'Streaming Days After Release', 'Marketing Budget Est. ($M)',
    'Total Cost Est. ($M)', 'Net Profit/Loss ($M)', 'ROI (%)',
    'Opening Weekend Ratio (%)', 'Runtime (min)', 'IMDb Score',
    'Rotten Tomatoes (%)', 'Opening Weekend ($M)'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_clean = df.dropna(subset=['Net Profit/Loss ($M)', 'Production Budget ($M)', 'ROI (%)'])

budget_filter = st.sidebar.selectbox(
    "Select Budget Tier",
    ["All"] + list(df_clean["Budget Tier"].dropna().unique())
)
if budget_filter != "All":
    df_clean = df_clean[df_clean["Budget Tier"] == budget_filter]

# ── Key Metrics ──────────────────────────────────────────────────────────────
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Movies", len(df_clean))
col2.metric("Avg ROI (%)", round(df_clean["ROI (%)"].mean(), 2))
col3.metric("Avg Profit ($M)", round(df_clean["Net Profit/Loss ($M)"].mean(), 2))

# ── Interactive Scatter ───────────────────────────────────────────────────────
st.subheader("Interactive Scatter Plot")
x_axis = st.selectbox("X-axis", numeric_cols, index=0)
y_axis = st.selectbox("Y-axis", numeric_cols, index=6)
color_by = st.selectbox("Color By", ["Budget Tier", "Genre", "Season"])

fig_scatter = px.scatter(
    df_clean,
    x=x_axis, y=y_axis,
    color=color_by,
    hover_data=["Movie_Title", "IMDb Score", "ROI (%)"],
    title=f"{y_axis} vs {x_axis}"
)
st.plotly_chart(fig_scatter, width="stretch", key="k_scatter_main")


st.subheader("Profit Distribution")
fig_hist = px.histogram(df_clean, x="Net Profit/Loss ($M)", nbins=40)
st.plotly_chart(fig_hist, width="stretch", key="k_hist_profit")


st.subheader("ROI Trend Over Years")
yearly = df_clean.groupby("Release Year")["ROI (%)"].mean().reset_index()
fig_yearly = px.line(yearly, x="Release Year", y="ROI (%)", markers=True)
st.plotly_chart(fig_yearly, width="stretch", key="k_line_yearly")


st.subheader("Top 10 Profitable Movies")
top_movies = df_clean.nlargest(10, "Net Profit/Loss ($M)")[
    ["Movie_Title", "Net Profit/Loss ($M)", "ROI (%)"]
]
st.dataframe(top_movies)


st.subheader("Profit Prediction Model")

feature_cols = [
    'Production Budget ($M)', 'Runtime (min)', 'IMDb Score',
    'Rotten Tomatoes (%)', 'Opening Weekend ($M)',
    'Sequel Number', 'Marketing Budget Est. ($M)'
]
target = 'Net Profit/Loss ($M)'

df_model = df_clean[feature_cols + [target]].dropna()
X = df_model[feature_cols]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.write(f"**Model R² Score:** {r2:.3f}")
st.write(f"**Mean Absolute Error:** {mae:.2f} million $")


st.subheader("Predict Movie Profit")
col1, col2 = st.columns(2)
with col1:
    budget   = st.number_input("Production Budget ($M)", 10, 500, 150)
    runtime  = st.number_input("Runtime (min)", 80, 200, 120)
    imdb     = st.slider("IMDb Score", 1.0, 10.0, 7.0)
with col2:
    rt       = st.slider("Rotten Tomatoes (%)", 0, 100, 80)
    opening  = st.number_input("Opening Weekend ($M)", 1, 300, 50)
    sequel   = st.number_input("Sequel Number", 0, 5, 0)
    marketing = st.number_input("Marketing Budget ($M)", 5, 200, 50)

if st.button("Predict Profit"):
    input_data = pd.DataFrame({
        'Production Budget ($M)':    [budget],
        'Runtime (min)':             [runtime],
        'IMDb Score':                [imdb],
        'Rotten Tomatoes (%)':       [rt],
        'Opening Weekend ($M)':      [opening],
        'Sequel Number':             [sequel],
        'Marketing Budget Est. ($M)':[marketing]
    })
    prediction = model.predict(scaler.transform(input_data))[0]
    st.success(f"Predicted Net Profit: ${prediction:.2f} million")


st.subheader("Correlation Heatmap")
corr_cols = [
    'Production Budget ($M)', 'WW Box Office Gross ($M)',
    'Net Profit/Loss ($M)', 'ROI (%)', 'IMDb Score',
    'Rotten Tomatoes (%)', 'Runtime (min)', 'Opening Weekend ($M)'
]
corr = df_clean[corr_cols].corr()
fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
st.plotly_chart(fig_corr, width="stretch", key="k_heatmap_corr")


st.subheader("Profit by Genre")
genre_profit = df_clean.groupby("Genre")["Net Profit/Loss ($M)"].mean().reset_index()
fig_genre = px.bar(
    genre_profit.sort_values("Net Profit/Loss ($M)", ascending=False),
    x="Genre", y="Net Profit/Loss ($M)",
    title="Average Profit by Genre"
)
st.plotly_chart(fig_genre, width="stretch", key="k_bar_genre")


st.subheader("Movies Released Per Year")
year_count = df_clean["Release Year"].value_counts().sort_index().reset_index()
year_count.columns = ["Year", "Count"]
fig_year = px.bar(year_count, x="Year", y="Count", title="Number of Movies Released Each Year")
st.plotly_chart(fig_year, width="stretch", key="k_bar_year")


st.subheader("IMDb Score vs ROI")
fig_imdb = px.scatter(
    df_clean, x="IMDb Score", y="ROI (%)", color="Genre",
    hover_data=["Movie_Title"],
    title="Do Better Movies Make More Profit?"
)
st.plotly_chart(fig_imdb, width="stretch", key="k_scatter_imdb")


st.subheader("Profit by Release Season")
fig_season = px.box(
    df_clean, x="Season", y="Net Profit/Loss ($M)",
    title="Seasonal Profit Trends"
)
st.plotly_chart(fig_season, width="stretch", key="k_box_season")


st.subheader("Budget vs Box Office Gross")
fig_budget_gross = px.scatter(
    df_clean,
    x="Production Budget ($M)", y="WW Box Office Gross ($M)",
    color="Genre",
    hover_data=["Movie_Title", "ROI (%)", "Net Profit/Loss ($M)"],
    title="Does a Higher Budget Lead to Higher Gross?"
)
st.plotly_chart(fig_budget_gross, width="stretch", key="k_scatter_budget_gross")


st.subheader("ROI by Genre")
fig_roi_genre = px.box(
    df_clean, x="Genre", y="ROI (%)",
    title="ROI Distribution Across Genres"
)
st.plotly_chart(fig_roi_genre, width="stretch", key="k_box_roi_genre")


st.subheader("Top 10 Highest Grossing Movies")
top_gross = df_clean.nlargest(10, "WW Box Office Gross ($M)")[
    ["Movie_Title", "WW Box Office Gross ($M)", "Production Budget ($M)", "Net Profit/Loss ($M)"]
]
fig_top_gross = px.bar(
    top_gross.sort_values("WW Box Office Gross ($M)"),
    x="WW Box Office Gross ($M)", y="Movie_Title",
    orientation="h",
    title="Top 10 Highest Grossing Movies"
)
st.plotly_chart(fig_top_gross, width="stretch", key="k_bar_top_gross")

st.subheader("Marketing Budget vs Net Profit")
fig_marketing = px.scatter(
    df_clean,
    x="Marketing Budget Est. ($M)", y="Net Profit/Loss ($M)",
    color="Budget Tier",
    hover_data=["Movie_Title"],
    title="Does More Marketing Spend Lead to Higher Profit?"
)
st.plotly_chart(fig_marketing, width="stretch", key="k_scatter_marketing")


st.subheader("Opening Weekend vs Total Gross")
fig_opening = px.scatter(
    df_clean,
    x="Opening Weekend ($M)", y="WW Box Office Gross ($M)",
    color="Genre",
    hover_data=["Movie_Title", "Net Profit/Loss ($M)"],
    title="Opening Weekend Performance vs Total Worldwide Gross"
)
st.plotly_chart(fig_opening, width="stretch", key="k_scatter_opening")

st.subheader("Profit by MPAA Rating")
fig_mpaa = px.box(
    df_clean, x="MPAA Rating", y="Net Profit/Loss ($M)",
    title="Profit Distribution by MPAA Rating"
)
st.plotly_chart(fig_mpaa, width="stretch", key="k_box_mpaa")


st.subheader("Sequel vs Original — Profit Comparison")
df_clean = df_clean.copy()
df_clean["Is Sequel"] = df_clean["Sequel Number"].apply(lambda x: "Sequel" if x > 1 else "Original")
sequel_profit = df_clean.groupby("Is Sequel")["Net Profit/Loss ($M)"].mean().reset_index()
fig_sequel = px.bar(
    sequel_profit, x="Is Sequel", y="Net Profit/Loss ($M)",
    title="Average Profit: Sequels vs Originals"
)
st.plotly_chart(fig_sequel, width="stretch", key="k_bar_sequel")

st.subheader("Runtime vs IMDb Score")
fig_runtime = px.scatter(
    df_clean, x="Runtime (min)", y="IMDb Score",
    color="Genre", hover_data=["Movie_Title"],
    title="Does Runtime Affect Audience Rating?"
)
st.plotly_chart(fig_runtime, width="stretch", key="k_scatter_runtime")

st.subheader("Streaming Days vs ROI")
fig_streaming = px.scatter(
    df_clean,
    x="Streaming Days After Release", y="ROI (%)",
    color="Budget Tier", hover_data=["Movie_Title"],
    title="Does Earlier Streaming Release Affect ROI?"
)
st.plotly_chart(fig_streaming, width="stretch", key="k_scatter_streaming")


st.subheader("Model: Actual vs Predicted Profit")
y_pred_all = model.predict(scaler.transform(df_model[feature_cols]))
fig_actual_pred = px.scatter(
    x=df_model[target], y=y_pred_all,
    labels={"x": "Actual Net Profit ($M)", "y": "Predicted Net Profit ($M)"},
    title=f"Actual vs Predicted Net Profit (R² = {r2:.3f})"
)
fig_actual_pred.add_shape(
    type="line",
    x0=df_model[target].min(), y0=df_model[target].min(),
    x1=df_model[target].max(), y1=df_model[target].max(),
    line=dict(dash="dash")
)
st.plotly_chart(fig_actual_pred, width="stretch", key="k_scatter_actual_pred")


st.subheader("Feature Importance (Model Coefficients)")
coeff_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
fig_coeff = px.bar(
    coeff_df, x="Coefficient", y="Feature",
    orientation="h",
    title="Feature Coefficients — Impact on Net Profit ($M per 1 std dev)"
)
st.plotly_chart(fig_coeff, width="stretch", key="k_bar_coeff")


st.caption("Built with Streamlit")