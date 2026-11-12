"""Person 4: hybrid model (content-based + collaborative).

Combines the scores from content_based.py and collaborative.py.
"""
from collaborative import get_user_recs
from content_based import get_similar_movies


def get_hybrid_recs(user_id, title, n=10, alpha=0.5):
    """Return a list of (movie_title, hybrid_score).

    alpha = weight of the content-based score (1 - alpha for collaborative).
    """
    content = dict(get_similar_movies(title, n=n * 2))
    collab = dict(get_user_recs(user_id, n=n * 2))

    # TODO: normalise both score sets to the same 0-1 scale before combining.
    combined = {}
    for movie in set(content) | set(collab):
        combined[movie] = alpha * content.get(movie, 0) + (1 - alpha) * collab.get(movie, 0)

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return ranked[:n]
