# linmod/transforms/build.py

import numpy as np
from typing import Any

def suggest_transformed_model(X_design: np.ndarray, residuals: np.ndarray, fitted_values: np.ndarray,
                               diagnostics: dict[str, Any], alpha: float = 0.05) -> dict[str, Any]:
    """
    Build a transformed dataset based on diagnostic test results.

    Parameters
    ----------
    X_design : np.ndarray
        Original design matrix (with intercept).
    residuals : np.ndarray
        Model residuals.
    fitted_values : np.ndarray
        Fitted values from the original model.
    diagnostics : dict
        Output from model.diagnostics().
    alpha : float
        Significance level.

    Returns
    -------
    dict with:
        - y_trans : transformed response
        - X_trans : transformed design matrix
        - features : feature names
        - description : list of transformations applied
    """
    X = X_design[:, 1:]  # exclude intercept
    y = fitted_values + residuals
    n, p = X.shape

    transforms = {
        "description": [],
        "X_trans": None,
        "y_trans": y.copy(),
        "features": [f"x{i+1}" for i in range(p)]
    }

    # -------- Response: Box-Cox ----------
    boxcox = diagnostics.get("box_cox", {})
    lmb = boxcox.get("lambda", 1.0)
    if abs(lmb - 1.0) > 0.1:
        if lmb == 0:
            y_t = np.log(y + 1e-8)
            transforms["description"].append("log(y)")
        else:
            y_t = (y ** lmb - 1) / lmb
            transforms["description"].append(f"Box–Cox transform on y (λ={lmb:.2f})")
        transforms["y_trans"] = y_t

    # -------- Predictors ----------
    X_parts = [np.ones((n, 1)), X]  # intercept + original X
    feature_names = transforms["features"]

    # Polynomial terms
    if diagnostics["reset"]["p-value"] < alpha or diagnostics["harvey_collier"]["p-value"] < alpha:
        for i in range(p):
            X_parts.append(X[:, i] ** 2)
            feature_names.append(f"x{i+1}^2")
        transforms["description"].append("Added squared terms (x^2)")

    # Interactions
    if diagnostics["white_nonlinearity"]["p-value"] < alpha:
        for i in range(p):
            for j in range(i + 1, p):
                X_parts.append(X[:, i] * X[:, j])
                feature_names.append(f"x{i+1}*x{j+1}")
        transforms["description"].append("Added interaction terms (x_i * x_j)")

    # Log, sqrt, inverse (Glejser / Park)
    if diagnostics["park"]["p-value"] < alpha or diagnostics["glejser"]["p-value"] < alpha:
        for i in range(p):
            xi = X[:, i]
            X_parts.extend([
                np.log(np.abs(xi) + 1e-8),
                np.sqrt(np.abs(xi)),
                1 / (np.abs(xi) + 1e-8)
            ])
            feature_names.extend([f"log(x{i+1})", f"sqrt(x{i+1})", f"1/x{i+1}"])
        transforms["description"].append("Log, sqrt, and inverse transforms of predictors")

    transforms["X_trans"] = np.column_stack(X_parts)
    transforms["features"] = feature_names

    return transforms
