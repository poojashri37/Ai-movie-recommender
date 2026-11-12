"""Person 1: data loading and cleaning.

Expects MovieLens "ml-latest-small" (100K) CSV files in the data/ folder:
    data/movies.csv   (movieId, title, genres)
    data/ratings.csv  (userId, movieId, rating, timestamp)
Download: https://grouplens.org/datasets/movielens/latest/
"""
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_data():
    """Return cleaned (movies, ratings) DataFrames."""
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

    # Movies: remove duplicates, fix empty genres, split out the release year
    movies = movies.drop_duplicates(subset="movieId").reset_index(drop=True)
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")
    movies["clean_title"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)\s*$", "", regex=True
    )

    # Ratings: drop missing/duplicate rows and ratings for unknown movies
    ratings = ratings.dropna().drop_duplicates(subset=["userId", "movieId"])
    ratings = ratings[ratings["movieId"].isin(movies["movieId"])].reset_index(drop=True)

    return movies, ratings


def save_clean_data(out_dir=DATA_DIR):
    """Optional: save cleaned copies to share with the team."""
    movies, ratings = load_data()
    movies.to_csv(os.path.join(out_dir, "movies_clean.csv"), index=False)
    ratings.to_csv(os.path.join(out_dir, "ratings_clean.csv"), index=False)


if __name__ == "__main__":
    movies, ratings = load_data()
    print("Movies:", movies.shape)
    print("Ratings:", ratings.shape)
    print(movies.head())
    print(ratings.head())
