"""Recommendation models.

All models share one interface:
    model.fit(train_df, n_users, n_items)
    model.pred          -> (n_users, n_items) predicted ratings   (used for RMSE / MAE)
    model.rank_scores   -> (n_users, n_items) ranking scores      (used for top-N lists)
    model.recommend(u, n)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MIN_RATING, MAX_RATING = 0.5, 5.0


# ----------------------------------------------------------------------------- helpers
def cosine_sim(X: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T


def keep_top_k(S: np.ndarray, k: int | None) -> np.ndarray:
    """Keep only the k largest similarities in each row (the k nearest neighbours)."""
    if k is None or k >= S.shape[1]:
        return S
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    rows = np.arange(S.shape[0])[:, None]
    out = np.zeros_like(S)
    out[rows, idx] = S[rows, idx]
    return out


def minmax_rows(A: np.ndarray) -> np.ndarray:
    lo, hi = A.min(1, keepdims=True), A.max(1, keepdims=True)
    return (A - lo) / np.maximum(hi - lo, 1e-9)


# ----------------------------------------------------------------------------- base
class Recommender:
    name = "base"

    def _prepare(self, train: pd.DataFrame, n_users: int, n_items: int):
        self.n_users, self.n_items = n_users, n_items
        self.R = np.zeros((n_users, n_items))
        self.M = np.zeros((n_users, n_items))
        self.R[train.u.values, train.i.values] = train.rating.values
        self.M[train.u.values, train.i.values] = 1.0
        self.global_mean = float(train.rating.mean())
        cnt = self.M.sum(1)
        self.user_mean = np.where(cnt > 0, self.R.sum(1) / np.maximum(cnt, 1), self.global_mean)
        self.Rc = (self.R - self.user_mean[:, None]) * self.M  # mean-centred ratings
        self.rank_scores = None

    def fit(self, train, n_users, n_items):
        self._prepare(train, n_users, n_items)
        self._fit()
        self.pred = np.clip(self.pred, MIN_RATING, MAX_RATING)
        if self.rank_scores is None:
            self.rank_scores = self.pred
        return self

    def _fit(self):
        raise NotImplementedError

    def _biases(self, reg: float = 10.0, iters: int = 10):
        """Baseline bias terms: rating ~ mu + b_user + b_item (regularised)."""
        mu = self.global_mean
        bu, bi = np.zeros(self.n_users), np.zeros(self.n_items)
        for _ in range(iters):
            bi = ((self.R - mu - bu[:, None]) * self.M).sum(0) / (self.M.sum(0) + reg)
            bu = ((self.R - mu - bi[None, :]) * self.M).sum(1) / (self.M.sum(1) + reg)
        return bu, bi

    def recommend(self, u: int, n: int = 10, exclude_seen: bool = True):
        s = self.rank_scores[u].astype(float).copy()
        if exclude_seen:
            s[self.M[u] > 0] = -np.inf
        n = min(n, self.n_items)
        top = np.argpartition(-s, n - 1)[:n]
        top = top[np.argsort(-s[top])]
        return top, self.pred[u, top]


# ----------------------------------------------------------------------------- baseline
class PopularityBaseline(Recommender):
    """Non-personalised: recommends the most-rated movies; rating = global mean + biases."""
    name = "popularity"

    def _fit(self):
        bu, bi = self._biases()
        self.pred = self.global_mean + bu[:, None] + bi[None, :]
        self.rank_scores = np.tile(self.M.sum(0), (self.n_users, 1))


# ----------------------------------------------------------------------------- content-based
def _tokens(s: str):
    return [t for t in s.split("|") if t]


def build_item_features(movies: pd.DataFrame) -> np.ndarray:
    """TF-IDF over genre tokens + release-decade token (feature extraction)."""
    genres = movies.genres.str.replace("(no genres listed)", "", regex=False)
    decade = movies.year.map(lambda y: f"decade{int(y) // 10 * 10}" if pd.notna(y) else "")
    text = genres + "|" + decade
    vec = TfidfVectorizer(tokenizer=_tokens, token_pattern=None, lowercase=False)
    return vec.fit_transform(text).toarray()


class ContentBased(Recommender):
    """Content-based filtering: movie-movie cosine similarity on TF-IDF features.

    * rating prediction: user mean + similarity-weighted average of the user's own residuals
    * ranking: cosine between the user's taste profile (sum of centred ratings x features) and each movie
    """
    name = "content"

    def __init__(self, movies: pd.DataFrame):
        self.features = build_item_features(movies)
        self.item_sim = cosine_sim(self.features)
        np.fill_diagonal(self.item_sim, 0.0)

    def _fit(self):
        S = self.item_sim
        num, den = self.Rc @ S.T, self.M @ S.T
        self.pred = self.user_mean[:, None] + np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
        profile = self.Rc @ self.features                     # (users x features)
        self.rank_scores = cosine_sim_cross(profile, self.features)

    def similar_items(self, item_idx: int, n: int = 10):
        s = self.item_sim[item_idx]
        top = np.argsort(-s)[:n]
        return top, s[top]


def cosine_sim_cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    an = np.linalg.norm(A, axis=1, keepdims=True); an[an == 0] = 1
    bn = np.linalg.norm(B, axis=1, keepdims=True); bn[bn == 0] = 1
    return (A / an) @ (B / bn).T


# ----------------------------------------------------------------------------- collaborative
class _NeighbourCF(Recommender):
    def __init__(self, similarity: str = "cosine", k: int = 50, shrink: float = 10.0):
        assert similarity in ("cosine", "pearson")
        self.similarity, self.k, self.shrink = similarity, k, shrink
        self.name = f"{self.kind}_cf_{similarity}"

    def _similarity(self, X, co):
        S = cosine_sim(X)
        S *= co / (co + self.shrink)          # shrinkage: distrust sims built on few co-ratings
        np.fill_diagonal(S, 0.0)
        return keep_top_k(np.clip(S, 0.0, None), self.k)


class UserBasedCF(_NeighbourCF):
    """Predict from the k most similar users who rated the item.
    'pearson' = cosine on mean-centred ratings (= Pearson correlation on co-rated items)."""
    kind = "user"

    def _fit(self):
        X = self.R if self.similarity == "cosine" else self.Rc          # users x items
        S = self._similarity(X, self.M @ self.M.T)                      # users x users
        num, den = S @ self.Rc, S @ self.M
        self.pred = self.user_mean[:, None] + np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)


class ItemBasedCF(_NeighbourCF):
    """Predict from the k most similar items the user already rated.
    'pearson' here = adjusted cosine (user-mean-centred), the standard item-based variant."""
    kind = "item"

    def __init__(self, similarity="cosine", k=100, shrink=10.0):
        super().__init__(similarity, k, shrink)

    def _fit(self):
        X = (self.R if self.similarity == "cosine" else self.Rc).T      # items x users
        S = self._similarity(X, self.M.T @ self.M)                      # items x items
        num, den = self.Rc @ S.T, self.M @ S.T
        self.pred = self.user_mean[:, None] + np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
        self.item_sim = S


# ----------------------------------------------------------------------------- matrix factorisation
class MatrixFactorizationALS(Recommender):
    """Latent-factor model: rating ~ mu + b_u + b_i + p_u . q_i, fitted by Alternating Least Squares."""
    name = "svd_als"

    def __init__(self, n_factors: int = 20, reg: float = 0.1, n_iters: int = 12, seed: int = 42):
        self.k, self.reg, self.n_iters, self.seed = n_factors, reg, n_iters, seed

    def _fit(self):
        rng = np.random.default_rng(self.seed)
        bu, bi = self._biases()
        base = self.global_mean + bu[:, None] + bi[None, :]
        E = (self.R - base) * self.M                                    # residuals on observed cells
        P = rng.normal(0, 0.1, (self.n_users, self.k))
        Q = rng.normal(0, 0.1, (self.n_items, self.k))
        user_idx = [np.flatnonzero(self.M[u]) for u in range(self.n_users)]
        item_idx = [np.flatnonzero(self.M[:, i]) for i in range(self.n_items)]
        eye = np.eye(self.k)

        def solve(target, other, idx_lists, resid):
            for t, idx in enumerate(idx_lists):
                if len(idx) == 0:
                    continue
                O = other[idx]
                A = O.T @ O + self.reg * len(idx) * eye
                target[t] = np.linalg.solve(A, O.T @ resid(t, idx))

        for _ in range(self.n_iters):
            solve(P, Q, user_idx, lambda u, idx: E[u, idx])
            solve(Q, P, item_idx, lambda i, idx: E[idx, i])
        self.pred = base + P @ Q.T
        self.P, self.Q = P, Q


# ----------------------------------------------------------------------------- hybrid
class HybridRecommender(Recommender):
    """Weighted blend: alpha * collaborative + (1 - alpha) * content-based."""
    name = "hybrid"

    def __init__(self, cf: Recommender, content: ContentBased, alpha: float = 0.7):
        self.cf, self.content, self.alpha = cf, content, alpha

    def fit(self, train, n_users, n_items):
        self._prepare(train, n_users, n_items)
        self.cf.fit(train, n_users, n_items)
        self.content.fit(train, n_users, n_items)
        a = self.alpha
        self.pred = np.clip(a * self.cf.pred + (1 - a) * self.content.pred, MIN_RATING, MAX_RATING)
        self.rank_scores = a * minmax_rows(self.cf.rank_scores) + (1 - a) * minmax_rows(self.content.rank_scores)
        return self
