"""Person 4: evaluation of all models.

TODO: train/test split, then compute RMSE, MAE, Precision@K, Recall@K
for content-based, collaborative, and hybrid models.
"""
import pandas as pd


def evaluate_all():
    """Return a results table comparing all models."""
    # DUMMY OUTPUT. Replace with real metrics.
    return pd.DataFrame(
        {
            "Model": ["Content-Based", "Collaborative", "Hybrid"],
            "RMSE": [None, None, None],
            "MAE": [None, None, None],
            "Precision@10": [None, None, None],
            "Recall@10": [None, None, None],
        }
    )


if __name__ == "__main__":
    print(evaluate_all())
