"""
app.py  (Person 5 - Streamlit front end)

Run from the project root:
    streamlit run app.py

This file only imports and calls the team's functions:
    preprocess.load_data()                          -> movies, ratings DataFrames
    content_based.get_similar_movies(title, n)      -> [(title, score), ...]
    hybrid.get_hybrid_recs(user_id, title, n, alpha) -> [(title, score), ...]
    hybrid.get_user_recs(user_id, n)                -> [(title, score), ...]
Posters from TMDB are optional (paste a free API key in the sidebar).
"""
import os
import re

import pandas as pd
import requests
import streamlit as st

from content_based import get_similar_movies
from hybrid import get_hybrid_recs, get_user_recs
from preprocess import load_data

st.set_page_config(page_title="AI Movie Recommender", page_icon="🎬", layout="wide")


# ----------------------------------------------------------------- helpers
@st.cache_data(show_spinner="Loading data...")
def get_data():
    movies, ratings = load_data()
    return movies, ratings


def to_pairs(result):
    """Turn whatever a model returns into a list of (title, score)."""
    if result is None:
        return []
    if isinstance(result, pd.DataFrame):
        cols = list(result.columns)
        title_col = "title" if "title" in cols else cols[0]
        score_col = next((c for c in ("score", "similarity_score", "predicted_score")
                          if c in cols), None)
        scores = result[score_col] if score_col else [0.0] * len(result)
        return [(t, float(s)) for t, s in zip(result[title_col], scores)]
    pairs = []
    for item in result:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            pairs.append((item[0], float(item[1])))
        else:
            pairs.append((str(item), 0.0))
    return pairs


@st.cache_resource(show_spinner="Loading models (the first start can take a minute)...")
def warm_up(first_user):
    """Builds the collaborative model once so later clicks are fast."""
    get_user_recs(first_user, n=1)
    return True


@st.cache_data(show_spinner="Finding recommendations...")
def hybrid_recs(user_id, title, n, alpha):
    return to_pairs(get_hybrid_recs(user_id, title, n=n, alpha=alpha))


@st.cache_data(show_spinner="Finding similar movies...")
def similar_recs(title, n):
    return to_pairs(get_similar_movies(title, n=n))


@st.cache_data(show_spinner="Finding picks for this user...")
def user_recs(user_id, n):
    return to_pairs(get_user_recs(user_id, n=n))


def clean_title(title):
    """'Usual Suspects, The (1995)' -> ('The Usual Suspects', '1995')."""
    m = re.match(r"^(.*)\s\((\d{4})\)\s*$", title)
    name, year = (m.group(1), m.group(2)) if m else (title, None)
    m2 = re.match(r"^(.*), (The|A|An)$", name)
    if m2:
        name = f"{m2.group(2)} {m2.group(1)}"
    return name, year


@st.cache_data(show_spinner=False)
def poster_url(title, api_key):
    if not api_key:
        return None
    name, year = clean_title(title)
    params = {"api_key": api_key, "query": name}
    if year:
        params["year"] = year
    try:
        r = requests.get("https://api.themoviedb.org/3/search/movie",
                         params=params, timeout=5)
        results = r.json().get("results", [])
        if results and results[0].get("poster_path"):
            return "https://image.tmdb.org/t/p/w342" + results[0]["poster_path"]
    except Exception:
        pass
    return None


def show_results(pairs, api_key):
    if not pairs:
        st.info("No recommendations found.")
        return
    if api_key:
        per_row = 5
        for i in range(0, len(pairs), per_row):
            cols = st.columns(per_row)
            for col, (title, score) in zip(cols, pairs[i:i + per_row]):
                url = poster_url(title, api_key)
                if url:
                    col.image(url)
                col.caption(f"**{title}**  \nscore {score:.2f}")
    else:
        df = pd.DataFrame(pairs, columns=["Title", "Score"])
        df["Score"] = df["Score"].round(3)
        df.index = range(1, len(df) + 1)
        st.dataframe(df)


def safe(fn, *args):
    """Show a friendly error instead of crashing the whole page."""
    try:
        return fn(*args)
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        return []


# ----------------------------------------------------------------- page
movies, ratings = get_data()
titles = sorted(movies["title"].dropna().unique())
user_ids = sorted(ratings["userId"].unique())

st.title("🎬 AI-Based Movie Recommender")
st.write("Search a movie and get recommendations from content-based, "
         "collaborative and hybrid models.")

with st.sidebar:
    st.header("Your choices")
    movie = st.selectbox("Search a movie (type to filter)", titles)
    new_user = st.checkbox("I'm a new user (no ratings yet)")
    if new_user:
        user_id = None
    else:
        user_id = int(st.number_input("User ID", min_value=int(min(user_ids)),
                                      max_value=int(max(user_ids)),
                                      value=int(min(user_ids)), step=1))
    n = st.slider("How many recommendations?", 5, 20, 10)
    alpha = st.slider("Hybrid weight on collaborative (alpha)", 0.0, 1.0, 0.8, 0.05,
                      help="1 = only collaborative filtering, 0 = only content-based. "
                           "0.8 was best in our tuning.")
    api_key = st.text_input("TMDB API key (optional, for posters)",
                            value=os.environ.get("TMDB_API_KEY", ""), type="password")

warm_up(int(user_ids[0]))

genres = movies.loc[movies["title"] == movie, "genres"]
if len(genres):
    st.subheader(movie)
    st.caption(str(genres.iloc[0]).replace("|", " • "))

tab1, tab2, tab3 = st.tabs(["⭐ Hybrid (best)", "🎞 Similar movies", "👤 For this user"])

with tab1:
    st.write("Combines content-based and collaborative scores.")
    show_results(safe(hybrid_recs, user_id, movie, n, alpha), api_key)

with tab2:
    st.write(f"Movies similar to **{movie}** (content-based).")
    show_results(safe(similar_recs, movie, n), api_key)

with tab3:
    if user_id is None:
        st.info("Untick 'I'm a new user' and choose a User ID to see personal picks.")
    else:
        st.write(f"Top picks for user {user_id} (collaborative filtering).")
        show_results(safe(user_recs, user_id, n), api_key)