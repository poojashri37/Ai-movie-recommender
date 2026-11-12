"""Person 5: Streamlit web app. Run with:  streamlit run app.py"""
import streamlit as st

from collaborative import get_user_recs
from content_based import get_similar_movies
from hybrid import get_hybrid_recs

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 AI-Based Movie Recommendation System")

mode = st.radio(
    "Recommendation type",
    ["Similar to a movie", "For a user", "Hybrid"],
    horizontal=True,
)
n = st.slider("Number of recommendations", 5, 20, 10)

title, user_id = "", 1
if mode in ("Similar to a movie", "Hybrid"):
    title = st.text_input("Movie title", "Toy Story")
if mode in ("For a user", "Hybrid"):
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)

if st.button("Recommend"):
    if mode == "Similar to a movie":
        results = get_similar_movies(title, n)
    elif mode == "For a user":
        results = get_user_recs(int(user_id), n)
    else:
        results = get_hybrid_recs(int(user_id), title, n)

    st.subheader("Top recommendations")
    for rank, (movie, score) in enumerate(results, start=1):
        st.write(f"**{rank}. {movie}**  (score: {score:.2f})")
