# Movie Recommendation System

A complete recommender built on the MovieLens dataset (auto-downloaded, ~100k ratings), implementing
content-based filtering, collaborative filtering, matrix factorization and a hybrid model, with proper evaluation.

## Quick start
```bash
pip install -r requirements.txt
python main.py evaluate                       # compare all models
python main.py recommend --user 1 --n 10      # personalised top-10 (hybrid by default)
python main.py recommend --user 1 --model item_cf_pearson
python main.py similar --title "Toy Story"    # similar movies
streamlit run app.py                          # optional web UI
```
No internet? Add `--synthetic` (e.g. `python main.py --synthetic evaluate`). The code also falls back to it automatically.

## Project structure
```
main.py                 CLI (evaluate / recommend / similar)
app.py                  Streamlit UI
recsys/data.py          MovieLens download + synthetic data generator
recsys/preprocessing.py cleaning, sparse-user/item filtering, re-indexing, per-user train/test split
recsys/models.py        all recommenders
recsys/evaluation.py    RMSE, MAE, Precision@K, Recall@K, NDCG@K, coverage
recsys/pipeline.py      shared glue code
```

## Concepts -> where they live
| Concept | Implementation |
|---|---|
| Data preprocessing | `clean_data`: drop NaN/duplicates, iterative min-ratings filter, re-index; `extract_year` |
| Feature extraction | `build_item_features`: TF-IDF over genre tokens + release decade |
| Content-based filtering | `ContentBased`: movie-movie cosine similarity and user taste profile |
| Collaborative filtering | `UserBasedCF`, `ItemBasedCF` (k nearest neighbours, similarity shrinkage) |
| Similarity measures | `cosine` (raw ratings) and `pearson` (mean-centred cosine) options |
| Machine learning model | `MatrixFactorizationALS`: biases + latent factors via Alternating Least Squares |
| Hybrid | `HybridRecommender`: `alpha * CF + (1 - alpha) * content` |
| Baseline | `PopularityBaseline` |
| Evaluation | RMSE/MAE (rating accuracy); Precision/Recall/NDCG@K (ranking, relevant = test rating >= 4) |

## How the models work
- **Content-based**: each movie becomes a TF-IDF vector; a user profile is the sum of their mean-centred ratings times
  those vectors. Movies closest (cosine) to the profile rank first. Handles new movies (no ratings needed).
- **User-based CF**: predict `mean_u + sum(sim(u,v) * (r_vi - mean_v)) / sum(sim)` over the k most similar users.
- **Item-based CF**: same idea using similar *items* the user already rated.
- **ALS matrix factorization**: learns hidden "taste" factors for users and movies; usually best RMSE.
- **Hybrid**: blends the SVD and content scores, which helps with sparse data / cold-start items.

## Notes
- Evaluation uses a per-user 80/20 split; recommendations in `recommend` use all ratings.
- A truly new user (no ratings) can't be personalised: fall back to the popularity model.
- Tune `k`, `shrink`, `n_factors`, `reg`, `alpha` for better scores on real MovieLens data.
- On the synthetic demo data the popularity baseline wins at ranking because synthetic ratings are popularity-skewed;
  on MovieLens, personalised models normally beat it on RMSE, and SVD/hybrid are typically strongest.
