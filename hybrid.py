"""
hybrid.py  (Person 4)
Combines Person 2's content-based scores and Person 3's collaborative scores.

Depends on:
    content_based.get_similar_movies(title, n)  -> [(title, score), ...]
    collaborative.get_user_recs(user_id, n)     -> [(title, score), ...]
    preprocess.load_data()                      -> movies, ratings DataFrames
"""
from functools import lru_cache

from content_based import get_similar_movies
from collaborative import get_user_recs
from preprocess import load_data

POOL = 50  # candidates requested from each model before blending


def _normalize(pairs):
    """Min-max scale a list of (title, score) to 0-1 and return {title: score}."""
    if not pairs:
        return {}
    scores = [s for _, s in pairs]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {t: 1.0 for t, _ in pairs}
    return {t: (s - lo) / (hi - lo) for t, s in pairs}


@lru_cache(maxsize=1)
def _popular_movies():
    """Last-resort fallback: best-rated movies with enough ratings."""
    movies, ratings = load_data()
    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    stats = stats[stats["count"] >= 50].sort_values("mean", ascending=False)
    merged = stats.join(movies.set_index("movieId")["title"], how="inner")
    return [(row.title, row.mean / 5.0) for row in merged.head(100).itertuples()]


def get_hybrid_recs(user_id=None, title=None, n=10, alpha=0.5):
    """
    Weighted hybrid recommendations.

    score = alpha * CF_score + (1 - alpha) * CB_score   (both min-max normalised)

    alpha = 1 -> pure collaborative, alpha = 0 -> pure content-based.
    Cold-start handling:
      * unknown user / no user_id -> content-based only
      * no title / unknown title  -> collaborative only
      * neither works             -> popular movies
    Returns a list of (title, score), best first.
    """
    cb, cf = {}, {}

    if title:
        try:
            cb = _normalize(get_similar_movies(title, n=POOL))
        except Exception:
            cb = {}

    if user_id is not None:
        try:
            cf = _normalize(get_user_recs(user_id, n=POOL))
        except Exception:
            cf = {}

    if not cb and not cf:
        return [(t, round(s, 4)) for t, s in _popular_movies()[:n]]

    # If one source is missing, use the other at full weight.
    a = 1.0 if not cb else 0.0 if not cf else alpha

    combined = {}
    for t in set(cb) | set(cf):
        if title and t == title:        # don't recommend the seed movie itself
            continue
        combined[t] = a * cf.get(t, 0.0) + (1 - a) * cb.get(t, 0.0)

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:n]
    return [(t, round(s, 4)) for t, s in ranked]


if __name__ == "__main__":
    print(get_hybrid_recs(user_id=1, title="Toy Story (1995)", n=5))
