# AI-Based Movie Recommendation System

A movie recommendation system that suggests movies using content-based filtering, collaborative filtering, and a hybrid of both. Built with Python and presented through a Streamlit web app.

## Features

- **Content-based filtering**: recommends movies similar to a given movie using genres, tags, and text features (TF-IDF + cosine similarity)
- **Collaborative filtering**: recommends movies based on similar users' ratings (user/item-based CF and SVD)
- **Hybrid model**: combines both approaches for better recommendations
- **Evaluation**: RMSE, MAE, Precision@K, Recall@K
- **Web app**: search a movie or pick a user ID and get the top-N recommendations

## Dataset

[MovieLens](https://grouplens.org/datasets/movielens/) (100K / 1M version). Place the files in the `data/` folder.

## Project Structure

```
movie-recommender/
├── data/                  # Dataset files
├── preprocess.py          # Data loading and cleaning
├── content_based.py       # Content-based filtering
├── collaborative.py       # Collaborative filtering
├── hybrid.py              # Hybrid model
├── evaluate.py            # Model evaluation
├── app.py                 # Streamlit web app
├── requirements.txt       # Dependencies
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
```

## Usage

Run the web app:

```bash
streamlit run app.py
```

It opens in your browser at `http://localhost:8501`.

## Team and Responsibilities

| Member | Module | Responsibility |
|--------|--------|----------------|
| Member 1 | `preprocess.py`, `data/` | Dataset, cleaning, EDA |
| Member 2 | `content_based.py` | Content-based filtering |
| Member 3 | `collaborative.py` | Collaborative filtering |
| Member 4 | `hybrid.py`, `evaluate.py` | Hybrid model and evaluation |
| Member 5 | `app.py`, `requirements.txt` | Streamlit app and integration |

## Function Interfaces

| Function | Returns |
|----------|---------|
| `load_data()` | `movies, ratings` DataFrames |
| `get_similar_movies(title, n=10)` | List of (title, score) |
| `get_user_recs(user_id, n=10)` | List of (title, score) |
| `get_hybrid_recs(user_id, title, n=10)` | List of (title, score) |
| `evaluate_all()` | Results table comparing all models |

## Git Workflow

1. Work on your own branch, e.g. `p2-content-based`
2. Only edit your own file
3. Push and open a pull request to `main`
4. Member 5 merges and tests

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Surprise / SciPy, Streamlit

## Future Work

- Movie posters via the TMDB API
- Deep learning based recommendations
- Real-time user feedback

## License

For academic use.
