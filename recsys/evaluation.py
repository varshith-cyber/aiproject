"""Evaluation: rating-prediction error (RMSE, MAE) and top-N ranking quality (precision/recall/NDCG@K)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rating_metrics(model, test: pd.DataFrame) -> dict:
    y = test.rating.values
    p = model.pred[test.u.values, test.i.values]
    return {"RMSE": float(np.sqrt(np.mean((y - p) ** 2))), "MAE": float(np.mean(np.abs(y - p)))}


def ranking_metrics(model, test: pd.DataFrame, k: int = 10, threshold: float = 4.0) -> dict:
    """A test movie is 'relevant' if the user rated it >= threshold.
    Candidates = every movie the user did not rate in the training set."""
    relevant = test[test.rating >= threshold].groupby("u").i.apply(set)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    prec, rec, ndcg, seen_items = [], [], [], set()
    for u, items in relevant.items():
        s = model.rank_scores[u].astype(float).copy()
        s[model.M[u] > 0] = -np.inf
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        hits = np.array([i in items for i in top], dtype=float)
        prec.append(hits.mean())
        rec.append(hits.sum() / len(items))
        ndcg.append((hits * discounts).sum() / discounts[: min(len(items), k)].sum())
        seen_items.update(top.tolist())
    return {f"Precision@{k}": float(np.mean(prec)), f"Recall@{k}": float(np.mean(rec)),
            f"NDCG@{k}": float(np.mean(ndcg)), "Coverage": len(seen_items) / model.n_items}


def evaluate_all(models: dict, train, test, n_users, n_items, k=10, threshold=4.0) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        print(f"  fitting {name} ...")
        model.fit(train, n_users, n_items)
        rows.append({"model": name, **rating_metrics(model, test), **ranking_metrics(model, test, k, threshold)})
    return pd.DataFrame(rows).set_index("model").round(4)
