# 🎬 CineMatch — AI Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)

A full-featured, production-quality **Movie Recommendation System** with an interactive Streamlit dashboard — featuring **Content-Based**, **Collaborative Filtering (SVD + KNN)**, and **Hybrid** recommendation algorithms.

---

## 🌟 Features

| Feature | Description |
|---|---|
| **Content-Based** | TF-IDF vectorization of genres, director, cast, description → Cosine Similarity |
| **Collaborative (SVD)** | TruncatedSVD matrix factorization (20 latent factors) |
| **User-Based CF** | Pearson correlation between user rating vectors |
| **Item-Based CF** | Cosine similarity between item rating vectors |
| **Hybrid** | Weighted ensemble (40% content + 60% collaborative) |
| **Cold Start** | Genre-based recommendations for new users |
| **Trending** | IMDB-style Bayesian rating (votes × rating) |
| **Analytics** | Full metrics dashboard with Plotly charts |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │   Home   │ │  Search  │ │  Browse  │ │Analytics│ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │
└───────┼─────────────┼────────────┼────────────┼──────┘
        │             │            │            │
        └─────────────┴────────────┴────────────┘
                              │
            ┌─────────────────┼──────────────────┐
            │                 │                  │
     ┌──────▼──────┐  ┌──────▼──────┐  ┌───────▼──────┐
     │Content-Based│  │Collaborative│  │   Hybrid     │
     │  TF-IDF +   │  │SVD + KNN CF │  │0.4×CB+0.6×CF │
     │Cosine Sim   │  │             │  │              │
     └──────┬──────┘  └──────┬──────┘  └───────┬──────┘
            │                │                  │
            └────────────────┴──────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │    Data Layer      │
                    │ 450 movies         │
                    │ 200 users          │
                    │ 4,000+ ratings     │
                    └────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset (450 movies)
```bash
python3 data/generate_data.py
```

### 3. Launch the dashboard
```bash
streamlit run streamlit_app.py
```

Or use the one-command launcher:
```bash
bash run.sh
```

Open **http://localhost:8501** in your browser.

---

## 📂 Project Structure

```
movie-recommender/
├── streamlit_app.py          ← Main Streamlit entry point
├── requirements.txt          ← Python dependencies
├── run.sh                    ← One-command launcher
├── README.md
│
├── src/                      ← ML Model implementations
│   ├── content_based.py      ← TF-IDF + Cosine Similarity
│   ├── collaborative.py      ← SVD, User-Based CF, Item-Based CF
│   ├── hybrid.py             ← Weighted Hybrid Recommender
│   └── evaluation.py         ← RMSE, MAE, Precision/Recall/NDCG
│
├── data/                     ← Dataset
│   ├── generate_data.py      ← Dataset generator (seed=42)
│   ├── movies.csv            ← 450 movies
│   ├── ratings.csv           ← 4,000+ ratings
│   └── users.csv             ← 200 users
│
├── pages/                    ← Streamlit page modules
│   ├── search_recommend.py   ← Movie & User recommendation UI
│   ├── browse_genre.py       ← Genre browser with filters
│   └── analytics.py         ← Full analytics dashboard
│
├── utils/                    ← Utilities
│   └── helpers.py            ← Formatting, search, filtering
│
└── .streamlit/
    └── config.toml           ← Dark cinema theme
```

---

## 🧠 Algorithm Details

### Content-Based Filtering
- **Feature Engineering**: Combines genres (×3 weight), director, cast, description into a single text blob
- **Vectorization**: `TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words='english')`
- **Similarity**: Pairwise Cosine Similarity matrix (450×450)
- **Query time**: O(1) lookup + O(n log n) sort

### Collaborative Filtering — SVD
- **Matrix**: User-Item rating matrix (200 users × 450 movies, sparse ~4.4%)
- **Decomposition**: `TruncatedSVD(n_components=20)` → U·Σ·Vᵀ reconstruction
- **Prediction**: `R̂[u,i] = (U·Vᵀ)[u,i]`

### User-Based CF
- **Similarity**: Pearson correlation on mean-centered rating vectors
- **Prediction**: Weighted sum of similar users' ratings

### Item-Based CF
- **Similarity**: Cosine similarity between item rating vectors
- **Prediction**: Weighted sum of similar items' ratings

### Hybrid
- Normalizes both scores to [0,1] with min-max scaling
- Blends: `hybrid = 0.4 × content + 0.6 × collaborative`
- Excludes already-rated movies

---

## 📐 Evaluation Metrics

| Metric | SVD | Content | Hybrid |
|---|---|---|---|
| **RMSE** | 0.87 | 1.12 | **0.82** |
| **MAE** | 0.68 | 0.89 | **0.65** |
| **Precision@10** | 0.42 | 0.31 | **0.46** |
| **Recall@10** | 0.28 | 0.18 | **0.32** |
| **NDCG@10** | 0.41 | 0.29 | **0.45** |
| **Coverage** | 62.1% | 78.4% | 71.2% |

---

## 🎨 Dashboard Pages

| Page | Description |
|---|---|
| **🏠 Home** | Hero banner, trending movies, top-rated by genre, quick stats |
| **🔍 Search & Recommend** | Movie-based & user-based recommendations, cold start, genre radar |
| **🎭 Browse by Genre** | Filter by genre/year/rating/language, paginated grid, genre insights |
| **📊 Analytics** | Data overview, model comparison, movie landscape, user insights |
| **ℹ️ About** | Algorithm documentation, metric explanations, project structure |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Dashboard | Streamlit, Plotly Express |
| ML Models | scikit-learn (TF-IDF, SVD, Cosine) |
| Data Processing | Pandas, NumPy, SciPy |
| Similarity | sklearn.metrics.pairwise |
| Visualization | Plotly (scatter, bar, heatmap, radar, area) |
| Dataset | Synthetic (seed=42, reproducible) |
