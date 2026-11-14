#!/bin/bash
# CineMatch — Movie Recommendation System launcher
set -e

echo "🎬 ============================================"
echo "   CineMatch — AI Movie Recommendation System"
echo "   ============================================"

echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "🎬 Generating dataset (450 movies, 200 users, 4000+ ratings)..."
python3 data/generate_data.py

echo ""
echo "🚀 Launching Streamlit dashboard..."
echo "   Open: http://localhost:8501"
echo ""
streamlit run streamlit_app.py --server.port 8501
echo "🎬 Starting CineMatch Movie Recommendation System..."
pip install -r requirements.txt -q
python data/generate_dataset.py
streamlit run app.py
