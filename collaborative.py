"""Person 3: collaborative filtering.

TODO: build the user-item matrix, implement user/item-based CF and SVD,
and handle cold-start users. Keep the function signature unchanged.
"""


def get_user_recs(user_id, n=10):
    """Return a list of (movie_title, predicted_rating) for `user_id`."""
    # DUMMY OUTPUT so the app can run before the real model exists.
    # Replace with the real collaborative filtering logic.
    return [(f"Recommended Movie {i}", round(5 - i * 0.2, 2)) for i in range(1, n + 1)]
