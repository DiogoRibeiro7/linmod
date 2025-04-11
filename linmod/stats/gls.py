# linmod/stats/gls.py
"""
Generalized Least Squares Regression
"""
import numpy as np
from linmod.base import BaseLinearModel

class GeneralizedLinearModel(BaseLinearModel):
    def __init__(self, sigma: np.ndarray):
        super().__init__()
        self.sigma = sigma

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_design = np.hstack([np.ones((X.shape[0], 1)), X])
        sigma_inv = np.linalg.inv(self.sigma)
        beta = np.linalg.pinv(X_design.T @ sigma_inv @ X_design) @ (X_design.T @ sigma_inv @ y)

        self.intercept = beta[0]
        self.coefficients = beta[1:]
        self.fitted_values = X_design @ beta
        self.residuals = y - self.fitted_values
        self.X_design_ = X_design
