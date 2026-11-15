"""Person 2: Content-Based Filtering.

Idea: recommend movies whose CONTENT (genres, plus tags / overview if the
project has them) is similar to a movie the user already likes.

Pipeline:
    1. Build one piece of text (a "soup") for every movie from its features.
    2. Turn every soup into a TF-IDF vector (a list of numbers).
    3. Compare vectors with cosine similarity.
    4. Return the N most similar movies.

Person 1's load_data() already cleans movies.csv and ratings.csv, so we reuse it
instead of repeating any cleaning.

Public function (required by the team):
    get_similar_movies(title, n=10) -> [(movie_title, similarity_score), ...]
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import DATA_DIR, load_data  # Person 1's code (we only import it)


# ---------------------------------------------------------------------------
# STEP 1 + 2: Feature extraction and text preprocessing
# ---------------------------------------------------------------------------
def _clean_text(text):
    """Lowercase the text and turn punctuation into spaces.

    "Sci-Fi|Action" -> "scifi action"
    Hyphens are removed (not turned into spaces) so that "Sci-Fi" stays ONE word
    and matches a "sci-fi" tag too.
    """
    text = str(text).lower().replace("-", "")
    return re.sub(r"[^a-z0-9 ]", " ", text)


def _load_tags():
    """Read data/tags.csv IF it exists. Returns a Series: movieId -> 'tag tag tag'.

    Returns None when there is no usable tags file, so the code keeps working.
    """
    path = os.path.join(DATA_DIR, "tags.csv")
    if not os.path.exists(path):
        return None
    tags = pd.read_csv(path)
    if not {"movieId", "tag"}.issubset(tags.columns):
        return None
    tags = tags.dropna(subset=["tag"])
    # One movie has many tag rows (from different users): join them into one string
    return tags.groupby("movieId")["tag"].apply(lambda t: " ".join(t.astype(str)))


def _build_soup(movies):
    """Combine every available text feature into one string per movie.

    We CHECK which columns / files exist instead of assuming them.
    Returns (soup Series, list of feature names that were used).
    """
    features_used = ["genres"]
    soup = movies["genres"].fillna("").apply(_clean_text)

    tag_series = _load_tags()
    if tag_series is not None:
        tag_text = movies["movieId"].map(tag_series).fillna("").apply(_clean_text)
        soup = soup + " " + tag_text
        features_used.append("tags")

    if "overview" in movies.columns:  # not in MovieLens, but handled if added later
        soup = soup + " " + movies["overview"].fillna("").apply(_clean_text)
        features_used.append("overview")

    return soup.str.strip(), features_used


# ---------------------------------------------------------------------------
# Title searching helpers (so users can type "Toy Story" or "The Matrix")
# ---------------------------------------------------------------------------
def _make_search_title(text):
    """Lowercase a title and fix MovieLens' moved article: 'Matrix, The' -> 'the matrix'."""
    text = str(text).lower().strip()
    match = re.match(r"^(.*), (the|a|an)$", text)
    if match:
        text = match.group(2) + " " + match.group(1)
    return text


def _find_movie_index(title, model):
    """Return the row number of the movie matching `title`, or None.

    Tries, in order: exact full title "Toy Story (1995)", exact title without
    the year, then "contains". If several movies match, the most-rated wins.
    """
    query = str(title).lower().strip()
    if not query:
        return None
    movies, popularity = model["movies"], model["popularity"]

    def best(hits):
        return int(hits[np.argmax(popularity[hits])]) if len(hits) else None

    # 1) exact match including the year
    hits = np.flatnonzero((movies["title"].str.lower() == query).to_numpy())
    if len(hits):
        return best(hits)

    # 2) exact match ignoring the year, 3) partial match
    query = _make_search_title(re.sub(r"\s*\(\d{4}\)\s*$", "", query))
    search = model["search_titles"]
    hits = np.flatnonzero((search == query).to_numpy())
    if len(hits):
        return best(hits)
    hits = np.flatnonzero(search.str.contains(query, regex=False).to_numpy())
    return best(hits)


# ---------------------------------------------------------------------------
# STEP 3: TF-IDF model (built once, then reused)
# ---------------------------------------------------------------------------
_model = {}  # simple cache, so we don't rebuild TF-IDF on every call


def _get_model():
    """Build the TF-IDF model the first time it is needed."""
    if _model:
        return _model

    movies, ratings = load_data()  # already cleaned by Person 1
    movies = movies.reset_index(drop=True)

    soup, features_used = _build_soup(movies)

    # TF-IDF: each soup -> a vector. Rare, distinctive words get higher weight.
    # stop_words removes filler words ("the", "and"); sublinear_tf stops a word
    # repeated many times (common in tags) from dominating.
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(soup)

    # How many ratings each movie has. Used ONLY to break ties between movies
    # with identical scores (many movies share the exact same genres).
    popularity = (
        movies["movieId"].map(ratings["movieId"].value_counts()).fillna(0).to_numpy()
    )

    _model.update(
        movies=movies,
        tfidf_matrix=tfidf_matrix,
        popularity=popularity,
        search_titles=movies["clean_title"].apply(_make_search_title),
        features_used=features_used,
    )
    return _model


# ---------------------------------------------------------------------------
# STEP 4 + 5: Cosine similarity and Top-N recommendations
# ---------------------------------------------------------------------------
def get_similar_movies(title, n=10):
    """Return the n movies most similar to `title`.

    Output: list of (movie_title, similarity_score) tuples, best first.
    Score is between 0 and 1 (1 = identical content).
    Returns [] if the title is not found.
    """
    model = _get_model()
    idx = _find_movie_index(title, model)
    if idx is None:
        return []

    # Similarity of ONE movie against all movies (a single row, so memory stays small)
    scores = cosine_similarity(model["tfidf_matrix"][idx], model["tfidf_matrix"]).ravel()
    scores = np.round(scores, 6)
    scores[idx] = -1  # never recommend the movie itself

    # Sort: highest score first; if tied, the movie with more ratings first.
    # (np.lexsort uses the LAST key as the main sort key.)
    order = np.lexsort((-model["popularity"], -scores))

    results = []
    for i in order[:n]:
        if scores[i] <= 0:  # no shared content at all -> not a real recommendation
            break
        results.append((model["movies"].loc[i, "title"], round(float(scores[i]), 4)))
    return results


# ---------------------------------------------------------------------------
# Testing with real movie titles: run `python content_based.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Features used:", _get_model()["features_used"])

    test_titles = [
        "Toy Story (1995)",
        "Jumanji",                 # no year typed
        "The Matrix",              # dataset stores it as "Matrix, The (1999)"
        "Heat (1995)",
        "Pulp Fiction",
        "Titanic",
        "Star Wars: Episode IV - A New Hope (1977)",
        "This Movie Does Not Exist",  # should give an empty list
    ]
    for t in test_titles:
        print("\nBecause you liked:", t)
        recs = get_similar_movies(t, n=5)
        if not recs:
            print("   (title not found)")
        for name, score in recs:
            print(f"   {score:.4f}  {name}")
