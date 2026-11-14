import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


class CollaborativeRecommender:
    def __init__(self, ratings_path="data/ratings.csv", movies_path="data/movies.csv", n_components=20):
        """
        Initializes the Collaborative Filtering recommender using user-item ratings.
        """
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.n_components = n_components
        
        self.ratings = None
        self.movies = None
        self.user_item_matrix = None
        self.item_similarity_df = None
        self.predicted_ratings_df = None

    def load_data(self):
        """Loads ratings and optional movies metadata."""
        self.ratings = pd.read_csv(self.ratings_path)
        try:
            self.movies = pd.read_csv(self.movies_path)
        except FileNotFoundError:
            self.movies = None

    def fit(self):
        """
        Builds the user-item matrix, computes item-item cosine similarity,
        and trains a TruncatedSVD model for rating prediction.
        """
        if self.ratings is None:
            self.load_data()

        # Create User-Item Rating Matrix
        self.user_item_matrix = self.ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating"
        ).fillna(0)

        # 1. Item-Based Collaborative Filtering (Cosine Similarity)
        item_matrix = self.user_item_matrix.T
        item_sim = cosine_similarity(item_matrix)
        self.item_similarity_df = pd.DataFrame(
            item_sim,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )

        # 2. Matrix Factorization (SVD) for predicting missing ratings
        n_comp = min(self.n_components, self.user_item_matrix.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        user_factors = svd.fit_transform(self.user_item_matrix)
        item_factors = svd.components_

        predicted_matrix = np.dot(user_factors, item_factors)
        self.predicted_ratings_df = pd.DataFrame(
            predicted_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns
        )

    def predict_rating(self, user_id, movie_id):
        """
        Predicts the rating a user would give to a specific movie (useful for evaluate.py / hybrid.py).
        """
        if self.predicted_ratings_df is None:
            self.fit()

        if user_id not in self.predicted_ratings_df.index or movie_id not in self.predicted_ratings_df.columns:
            # Fallback to global average rating if user or movie is unseen
            return float(self.ratings["rating"].mean())

        pred = self.predicted_ratings_df.loc[user_id, movie_id]
        # Clip predictions to valid rating range (e.g., 0.5 to 5.0)
        return float(np.clip(pred, 0.5, 5.0))

    def recommend_for_user(self, user_id, top_n=10):
        """
        Recommends top_n unseen movies for a given userId based on predicted collaborative scores.
        Returns a DataFrame with movieId, predicted_score (and title if movies.csv exists).
        """
        if self.predicted_ratings_df is None:
            self.fit()

        if user_id not in self.user_item_matrix.index:
            # Cold-start fallback: return top popular highest-rated movies
            popular = (
                self.ratings.groupby("movieId")["rating"]
                .agg(["count", "mean"])
                .query("count >= 10")
                .sort_values(by="mean", ascending=False)
                .head(top_n)
                .reset_index()
            )
            popular.rename(columns={"mean": "predicted_score"}, inplace=True)
            recs = popular[["movieId", "predicted_score"]]
        else:
            # Get movies the user has already rated
            user_ratings = self.user_item_matrix.loc[user_id]
            watched_movies = user_ratings[user_ratings > 0].index

            # Predict ratings for all movies and filter out already watched ones
            user_preds = self.predicted_ratings_df.loc[user_id].drop(index=watched_movies)
            top_movies = user_preds.sort_values(ascending=False).head(top_n)

            recs = pd.DataFrame({
                "movieId": top_movies.index,
                "predicted_score": top_movies.values
            })

        if self.movies is not None and "title" in self.movies.columns:
            recs = recs.merge(self.movies[["movieId", "title"]], on="movieId", how="left")

        return recs.reset_index(drop=True)

    def get_similar_movies(self, movie_id, top_n=10):
        """
        Finds top_n movies most similar to a given movieId based on user rating patterns.
        """
        if self.item_similarity_df is None:
            self.fit()

        if movie_id not in self.item_similarity_df.index:
            return pd.DataFrame(columns=["movieId", "similarity_score"])

        sim_scores = self.item_similarity_df[movie_id].drop(index=movie_id)
        top_similar = sim_scores.sort_values(ascending=False).head(top_n)

        recs = pd.DataFrame({
            "movieId": top_similar.index,
            "similarity_score": top_similar.values
        })

        if self.movies is not None and "title" in self.movies.columns:
            recs = recs.merge(self.movies[["movieId", "title"]], on="movieId", how="left")

        return recs.reset_index(drop=True)


if __name__ == "__main__":
    # Quick test run
    cf = CollaborativeRecommender(ratings_path="data/ratings.csv", movies_path="data/movies.csv")
    cf.fit()

    sample_user = cf.user_item_matrix.index[0]
    print(f"Top 5 Collaborative Recommendations for User {sample_user}:")
    print(cf.recommend_for_user(user_id=sample_user, top_n=5))

    sample_movie = cf.user_item_matrix.columns[0]
    print(f"\nTop 5 Similar Movies to Movie ID {sample_movie}:")
    print(cf.get_similar_movies(movie_id=sample_movie, top_n=5))
