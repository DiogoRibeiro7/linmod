# linmod/evaluation/crossval.py
"""
Cross-validation utilities for linear and regularized models.
"""
import numpy as np
from sklearn.model_selection import KFold
from typing import Callable


def cross_val_score(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    param_grid: dict,
    k: int = 5,
    scoring: Callable[[np.ndarray, np.ndarray], float] = None
) -> dict:
    """
    Generic k-fold cross-validation for any model constructor.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    model_fn : Callable
        Function that returns a model instance (e.g., lambda: RidgeLinearModel(lambda_=0.1)).
    param_grid : dict
        Hyperparameter name → list of values.
    k : int
        Number of folds.
    scoring : Callable
        Function(y_true, y_pred) -> float. If None, use mean squared error.

    Returns
    -------
    dict
        Dictionary mapping parameter combinations to average score.
    """
    if scoring is None:
        def scoring(y_true, y_pred): return -np.mean((y_true - y_pred) ** 2)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = {}

    from itertools import product
    keys, values = zip(*param_grid.items())
    for combo in product(*values):
        params = dict(zip(keys, combo))
        scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = model_fn(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(scoring(y_test, y_pred))

        results[tuple(combo)] = np.mean(scores)

    return results
