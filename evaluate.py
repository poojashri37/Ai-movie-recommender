"""
evaluate.py  (Person 4)
evaluate_all() -> DataFrame comparing Content-Based, Collaborative and Hybrid
on RMSE, MAE, Precision@K and Recall@K.

Requirements from teammates (agree on these on day one):
  P3 (collaborative.py):
      fit(train_ratings_df)          - rebuild the model on the training split ONLY
                                       (otherwise test ratings leak into the model)
      predict_rating(user_id, title) - predicted rating (needed for RMSE/MAE)
  P2 (content_based.py): get_similar_movies(title, n) as agreed.
"""
import numpy as np
import pandas as pd

import collaborative
from content_based import get_similar_movies
from hybrid import get_hybrid_recs
from preprocess import load_data

K = 10
RELEVANT_THRESHOLD = 4.0     # rating >= 4 counts as "liked"
ALPHA = 0.5
SEED = 42
MAX_USERS = 200              # users sampled for ranking metrics
MAX_TEST_ROWS = 500          # rows sampled for RMSE / MAE


# ---------------------------------------------------------------- split
def train_test_split_ratings(ratings, test_frac=0.2, seed=SEED):
    """Per-user random split; users with <5 ratings stay fully in train."""
    rng = np.random.RandomState(seed)
    test_idx = []
    for _, grp in ratings.groupby("userId"):
        if len(grp) >= 5:
            test_idx.extend(rng.choice(grp.index, int(len(grp) * test_frac), replace=False))
    test = ratings.loc[test_idx]
    return ratings.drop(test.index), test


# ---------------------------------------------------------------- metrics
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def precision_recall_at_k(recommended, relevant, k=K):
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / k, (hits / len(relevant) if relevant else 0.0)


# ---------------------------------------------------------------- rating predictors
def _cb_predict(user_hist, target_title, user_mean):
    """Content-based rating: similarity-weighted average of the user's own ratings."""
    try:
        sims = dict(get_similar_movies(target_title, n=50))
    except Exception:
        return user_mean
    num = den = 0.0
    for t, r in user_hist.items():
        if t in sims:
            num += sims[t] * r
            den += abs(sims[t])
    return num / den if den > 0 else user_mean


def _cf_predict(user_id, title, user_mean):
    fn = getattr(collaborative, "predict_rating", None)
    if fn is None:
        return np.nan
    try:
        return float(fn(user_id, title))
    except Exception:
        return user_mean


# ---------------------------------------------------------------- recommenders for ranking
def _seed_titles(user_hist, m=3):
    return [t for t, _ in sorted(user_hist.items(), key=lambda x: -x[1])[:m]]


def _rec_content(user_id, user_hist):
    scores = {}
    for seed in _seed_titles(user_hist):
        try:
            for t, s in get_similar_movies(seed, n=50):
                scores[t] = scores.get(t, 0) + s
        except Exception:
            pass
    return [t for t, _ in sorted(scores.items(), key=lambda x: -x[1])]


def _rec_cf(user_id, user_hist):
    try:
        return [t for t, _ in collaborative.get_user_recs(user_id, n=50)]
    except Exception:
        return []


def _rec_hybrid(user_id, user_hist):
    seeds = _seed_titles(user_hist, 1)
    seed = seeds[0] if seeds else None
    return [t for t, _ in get_hybrid_recs(user_id, seed, n=50, alpha=ALPHA)]


# ---------------------------------------------------------------- main entry point
def evaluate_all(k=K):
    """Run every model on the same split and return a results DataFrame."""
    movies, ratings = load_data()
    ratings = ratings.merge(movies[["movieId", "title"]], on="movieId")
    train, test = train_test_split_ratings(ratings)

    # Rebuild collaborative model on train only (see module docstring).
    if hasattr(collaborative, "fit"):
        collaborative.fit(train[["userId", "movieId", "rating"]])
    else:
        print("WARNING: collaborative.fit() not found - CF results may be leaked/inflated.")

    hist = {u: dict(zip(g["title"], g["rating"])) for u, g in train.groupby("userId")}
    means = train.groupby("userId")["rating"].mean().to_dict()
    global_mean = train["rating"].mean()

    # ---- RMSE / MAE
    sample = test.sample(min(MAX_TEST_ROWS, len(test)), random_state=SEED)
    y, p_cb, p_cf, p_hy = [], [], [], []
    for row in sample.itertuples():
        uh = hist.get(row.userId, {})
        um = means.get(row.userId, global_mean)
        cb = _cb_predict(uh, row.title, um)
        cf = _cf_predict(row.userId, row.title, um)
        hy = ALPHA * cf + (1 - ALPHA) * cb if not np.isnan(cf) else np.nan
        y.append(row.rating); p_cb.append(cb); p_cf.append(cf); p_hy.append(hy)

    def _err(pred):
        pred = np.array(pred, dtype=float)
        ok = ~np.isnan(pred)
        if not ok.any():
            return np.nan, np.nan
        yt = np.array(y)[ok]
        return rmse(yt, pred[ok]), mae(yt, pred[ok])

    # ---- Precision@K / Recall@K
    relevant = (test[test["rating"] >= RELEVANT_THRESHOLD]
                .groupby("userId")["title"].apply(set).to_dict())
    users = [u for u in relevant if u in hist]
    rng = np.random.RandomState(SEED)
    users = list(rng.choice(users, min(MAX_USERS, len(users)), replace=False))

    models = {
        "Content-Based": (_rec_content, _err(p_cb)),
        "Collaborative": (_rec_cf, _err(p_cf)),
        "Hybrid":        (_rec_hybrid, _err(p_hy)),
    }

    rows = []
    for name, (rec_fn, (r, m)) in models.items():
        precs, recs = [], []
        for u in users:
            seen = set(hist[u])
            rec_list = [t for t in rec_fn(u, hist[u]) if t not in seen]
            pr, rc = precision_recall_at_k(rec_list, relevant[u], k)
            precs.append(pr); recs.append(rc)
        rows.append({
            "Model": name,
            "RMSE": round(r, 4) if not np.isnan(r) else np.nan,
            "MAE": round(m, 4) if not np.isnan(m) else np.nan,
            f"Precision@{k}": round(float(np.mean(precs)), 4),
            f"Recall@{k}": round(float(np.mean(recs)), 4),
        })

    return pd.DataFrame(rows).set_index("Model")


if __name__ == "__main__":
    results = evaluate_all()
    print(results)
    results.to_csv("results.csv")
