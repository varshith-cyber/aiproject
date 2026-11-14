"""
Evaluation metrics: RMSE, MAE, Precision@K, Recall@K, NDCG@K, F1@K
"""
import numpy as np
import pandas as pd
from math import log2


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def precision_at_k(recommended, relevant, k):
    if not recommended or k == 0:
        return 0.0
    return sum(1 for x in recommended[:k] if x in relevant) / k


def recall_at_k(recommended, relevant, k):
    if not relevant or k == 0:
        return 0.0
    return sum(1 for x in recommended[:k] if x in relevant) / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    def dcg(items):
        return sum(1.0 / log2(i + 2) for i, x in enumerate(items[:k]) if x in relevant)

    ideal = min(len(relevant), k)
    if ideal == 0:
        return 0.0
    idcg = sum(1.0 / log2(i + 2) for i in range(ideal))
    return float(dcg(recommended) / idcg) if idcg else 0.0


def f1_at_k(recommended, relevant, k):
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


class ModelEvaluator:
    def __init__(self, test_df):
        self.test_df = test_df

    def evaluate_rating(self, predict_fn):
        y_true, y_pred = [], []
        for _, row in self.test_df.iterrows():
            pred = predict_fn(int(row["user_id"]), int(row["movie_id"]))
            y_true.append(float(row["rating"]))
            y_pred.append(float(pred))
        return {
            "RMSE": round(rmse(y_true, y_pred), 4),
            "MAE": round(mae(y_true, y_pred), 4),
            "N": len(y_true),
        }

    def evaluate_ranking(self, recommend_fn, ratings_df, k=10, threshold=3.5):
        relevant_map = (
            ratings_df[ratings_df["rating"] >= threshold]
            .groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )
        p_list, r_list, n_list, f_list = [], [], [], []
        for uid, relevant in relevant_map.items():
            recs = recommend_fn(uid)
            if not recs:
                continue
            p_list.append(precision_at_k(recs, relevant, k))
            r_list.append(recall_at_k(recs, relevant, k))
            n_list.append(ndcg_at_k(recs, relevant, k))
            f_list.append(f1_at_k(recs, relevant, k))
        return {
            f"Precision@{k}": round(np.mean(p_list), 4) if p_list else 0.0,
            f"Recall@{k}": round(np.mean(r_list), 4) if r_list else 0.0,
            f"NDCG@{k}": round(np.mean(n_list), 4) if n_list else 0.0,
            f"F1@{k}": round(np.mean(f_list), 4) if f_list else 0.0,
        }
