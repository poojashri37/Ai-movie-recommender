"""Person 2: content-based filtering.

TODO: build TF-IDF features from genres/tags, compute cosine similarity,
and return the most similar movies. Keep the function signature unchanged.
"""


def get_similar_movies(title, n=10):
    """Return a list of (movie_title, similarity_score) similar to `title`."""
    # DUMMY OUTPUT so the app can run before the real model exists.
    # Replace with the real TF-IDF + cosine similarity logic.
    return [(f"Similar Movie {i}", round(1 - i * 0.05, 2)) for i in range(1, n + 1)]
