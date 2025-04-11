# linmod/regularization/ridge.py
"""
Ridge Regression using closed-form solution.
"""
import numpy as np
from linmod.base import BaseLinearModel

class RidgeLinearModel(BaseLinearModel):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, p = X.shape
        X_design = np.hstack([np.ones((n, 1)), X])
        I = np.eye(p + 1)
        I[0, 0] = 0  # Do not penalize intercept

        beta = np.linalg.pinv(X_design.T @ X_design + self.lambda_ * I) @ (X_design.T @ y)

        self.intercept = beta[0]
        self.coefficients = beta[1:]
        self.fitted_values = X_design @ beta
        self.residuals = y - self.fitted_values
        self.X_design_ = X_design

