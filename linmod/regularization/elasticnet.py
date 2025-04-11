# linmod/regularization/elasticnet.py
"""
Elastic Net Regression using coordinate descent.
"""
import numpy as np
from linmod.base import BaseLinearModel

class ElasticNetLinearModel(BaseLinearModel):
    def __init__(self, lambda_: float = 1.0, alpha: float = 0.5, max_iter: int = 1000, tol: float = 1e-4):
        super().__init__()
        self.lambda_ = lambda_
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.X_mean = None
        self.X_std = None
        self.y_mean = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, p = X.shape
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        self.y_mean = y.mean()

        X_std = (X - self.X_mean) / self.X_std
        y_centered = y - self.y_mean

        beta = np.zeros(p)
        for _ in range(self.max_iter):
            beta_old = beta.copy()
            for j in range(p):
                r_j = y_centered - X_std @ beta + X_std[:, j] * beta[j]
                rho = X_std[:, j].T @ r_j
                z = (X_std[:, j] ** 2).sum() + self.lambda_ * (1 - self.alpha)
                if rho < -self.lambda_ * self.alpha / 2:
                    beta[j] = (rho + self.lambda_ * self.alpha / 2) / z
                elif rho > self.lambda_ * self.alpha / 2:
                    beta[j] = (rho - self.lambda_ * self.alpha / 2) / z
                else:
                    beta[j] = 0
            if np.linalg.norm(beta - beta_old, ord=1) < self.tol:
                break

        self.coefficients = beta / self.X_std
        self.intercept = self.y_mean - self.X_mean @ self.coefficients
        self.fitted_values = X @ self.coefficients + self.intercept
        self.residuals = y - self.fitted_values
