# linmod/base.py
import numpy as np

class BaseLinearModel:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.fitted_values = None
        self.residuals = None
        self.X_design_ = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not fit yet.")
        return self.intercept + X @ self.coefficients
