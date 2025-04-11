# linmod/stats/wls.py
"""
Weighted Least Squares Regression
"""
import numpy as np
from linmod.base import BaseLinearModel

class WeightedLinearModel(BaseLinearModel):
    def __init__(self, weights: np.ndarray):
        super().__init__()
        self.weights = weights
        self.coefficients = None
        self.intercept = None
        self.fitted_values = None
        self.residuals = None
        self.X_design_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.weights.ndim == 1:
            W = np.diag(self.weights)
        else:
            W = self.weights

        X_design = np.hstack([np.ones((X.shape[0], 1)), X])
        beta = np.linalg.pinv(X_design.T @ W @ X_design) @ (X_design.T @ W @ y)

        self.coefficients = beta[1:]
        self.intercept = beta[0]
        self.fitted_values = X_design @ beta
        self.residuals = y - self.fitted_values
        self.X_design_ = X_design
