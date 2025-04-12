# linmod/base.py
import numpy as np

class BaseLinearModel:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.fitted_values = None
        self.residuals = None
        self.X_design_ = None
        
    def _fit_ols_via_pinv(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Internal method to fit OLS using pseudoinverse. Returns:
        - beta coefficients
        - fitted values
        - residuals
        """
        n = X.shape[0]
        X_design = np.hstack([np.ones((n, 1)), X])
        beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
        fitted = X_design @ beta
        residuals = y - fitted

        self.X_design_ = X_design
        self.intercept = beta[0]
        self.coefficients = beta[1:]
        self.fitted_values = fitted
        self.residuals = residuals

        return beta, fitted, residuals

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not fit yet.")
        return self.intercept + X @ self.coefficients
