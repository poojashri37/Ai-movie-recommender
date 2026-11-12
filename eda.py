"""Person 1: exploratory data analysis (EDA).

Run:  python eda.py
Saves charts into the eda_charts/ folder and prints summary numbers
you can copy into the report's dataset description section.
"""
import os

import matplotlib

matplotlib.use("Agg")  # save charts to files, no window needed
import matplotlib.pyplot as plt

from preprocess import load_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_charts")
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig_name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fig_name), dpi=150)
    plt.close()


def main():
    movies, ratings = load_data()

    # ---- Summary numbers for the report ----
    n_users = ratings["userId"].nunique()
    n_movies = ratings["movieId"].nunique()
    n_ratings = len(ratings)
    sparsity = 1 - n_ratings / (n_users * n_movies)
    print("Users:", n_users)
    print("Movies rated:", n_movies, "(total movies:", len(movies), ")")
    print("Ratings:", n_ratings)
    print(f"Rating range: {ratings['rating'].min()} to {ratings['rating'].max()}")
    print(f"Average rating: {ratings['rating'].mean():.2f}")
    print(f"Sparsity of user-item matrix: {sparsity:.2%}")

    # ---- 1. Rating distribution ----
    plt.figure(figsize=(7, 4))
    ratings["rating"].value_counts().sort_index().plot(kind="bar", color="#4C72B0")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of ratings")
    save("1_rating_distribution.png")

    # ---- 2. Ratings per user ----
    plt.figure(figsize=(7, 4))
    ratings.groupby("userId").size().plot(kind="hist", bins=40, color="#55A868")
    plt.title("Ratings per User")
    plt.xlabel("Number of ratings given")
    plt.ylabel("Number of users")
    save("2_ratings_per_user.png")

    # ---- 3. Ratings per movie ----
    plt.figure(figsize=(7, 4))
    ratings.groupby("movieId").size().plot(kind="hist", bins=40, color="#C44E52")
    plt.title("Ratings per Movie")
    plt.xlabel("Number of ratings received")
    plt.ylabel("Number of movies")
    save("3_ratings_per_movie.png")

    # ---- 4. Genre popularity (number of movies per genre) ----
    genres = movies["genres"].str.split("|").explode()
    genres = genres[genres != ""]
    plt.figure(figsize=(8, 5))
    genres.value_counts().sort_values().plot(kind="barh", color="#8172B2")
    plt.title("Number of Movies per Genre")
    plt.xlabel("Number of movies")
    save("4_genre_popularity.png")

    # ---- 5. Top 10 most rated movies ----
    counts = ratings.groupby("movieId").size().sort_values(ascending=False).head(10)
    top = movies.set_index("movieId").loc[counts.index, "title"]
    plt.figure(figsize=(8, 5))
    plt.barh(top.values[::-1], counts.values[::-1], color="#CCB974")
    plt.title("Top 10 Most Rated Movies")
    plt.xlabel("Number of ratings")
    save("5_top_rated_count.png")

    # ---- 6. Movies by release decade ----
    years = movies["year"].dropna().astype(int)
    decades = (years // 10 * 10).value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    decades.plot(kind="bar", color="#64B5CD")
    plt.title("Movies by Release Decade")
    plt.xlabel("Decade")
    plt.ylabel("Number of movies")
    save("6_movies_by_decade.png")

    # ---- 7. Average rating per genre ----
    merged = ratings.merge(movies[["movieId", "genres"]], on="movieId")
    merged["genre"] = merged["genres"].str.split("|")
    merged = merged.explode("genre")
    merged = merged[merged["genre"] != ""]
    avg = merged.groupby("genre")["rating"].mean().sort_values()
    plt.figure(figsize=(8, 5))
    avg.plot(kind="barh", color="#4C72B0")
    plt.title("Average Rating per Genre")
    plt.xlabel("Average rating")
    save("7_avg_rating_per_genre.png")

    print(f"\nCharts saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
