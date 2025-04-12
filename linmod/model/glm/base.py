# linmod/glm/base.py

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from linmod.model.glm.links import LinkFunction  


class BaseGLM(ABC):
    """
    Abstract base class for Generalized Linear Models (GLMs).
    Uses Iteratively Reweighted Least Squares (IRLS) for fitting.
    """

    def __init__(self, link: LinkFunction, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.link = link
        self.max_iter = max_iter
        self.tol = tol

        self.coefficients: Optional[np.ndarray] = None
        self.fitted_values: Optional[np.ndarray] = None
        self.linear_predictor: Optional[np.ndarray] = None
        self.n_iter: int = 0
        self.converged: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the GLM model using IRLS.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (with intercept).
        y : np.ndarray
            Response variable.
        """
        n_samples, n_features = X.shape
        beta = np.zeros(n_features)
        eta = X @ beta
        mu = self.link.inverse(eta)

        for iteration in range(self.max_iter):
            var = self._variance(mu)
            deriv = self.link.derivative(mu)

            z = eta + (y - mu) / deriv
            W = deriv**2 / var

            WX = X * np.sqrt(W[:, np.newaxis])
            wz = z * np.sqrt(W)

            beta_new, *_ = np.linalg.lstsq(WX, wz, rcond=None)
            diff = np.linalg.norm(beta_new - beta)

            beta = beta_new
            eta = X @ beta
            mu = self.link.inverse(eta)

            self.n_iter = iteration + 1

            if diff < self.tol:
                self.converged = True
                break

        self.coefficients = beta
        self.fitted_values = mu
        self.linear_predictor = eta

    @abstractmethod
    def _variance(self, mu: np.ndarray) -> np.ndarray:
        """
        Return the variance function V(mu) for the distribution family.
        """
        pass

    def predict(self, X: np.ndarray, type: str = "response") -> np.ndarray:
        """
        Predict values from the fitted model.

        Parameters
        ----------
        X : np.ndarray
            New design matrix.
        type : str
            "response" for μ̂, "link" for η̂.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        if self.coefficients is None:
            raise ValueError("Model is not fit.")

        eta = X @ self.coefficients
        if type == "link":
            return eta
        elif type == "response":
            return self.link.inverse(eta)
        else:
            raise ValueError("type must be 'link' or 'response'")
