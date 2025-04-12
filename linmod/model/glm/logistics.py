import numpy as np
from typing import Any
from linmod.model.glm.base import BaseGLM
from linmod.model.glm.links import LogitLink


class LogisticRegressionGLM(BaseGLM):
    """
    Generalized Linear Model for binary outcomes using the logit link (Logistic Regression).
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-6) -> None:
        super().__init__(link=LogitLink(), max_iter=max_iter, tol=tol)

    def _variance(self, mu: np.ndarray) -> np.ndarray:
        """
        Variance function for the binomial distribution: V(μ) = μ(1 - μ)
        """
        return mu * (1 - mu)

    def deviance(self, y: np.ndarray | None = None) -> float:
        """
        Compute binomial deviance for logistic regression.

        Parameters
        ----------
        y : np.ndarray, optional
            True binary response values. If None, uses self.y.

        Returns
        -------
        float
            Deviance.
        """
        if self.fitted_values is None or self.y is None:
            raise ValueError("Model must be fit before computing deviance.")

        y_true = self.y if y is None else y
        mu = np.clip(self.fitted_values, 1e-8, 1 - 1e-8)

        return 2 * np.sum(
            y_true * np.log(y_true / mu + 1e-8) +
            (1 - y_true) * np.log((1 - y_true) / (1 - mu + 1e-8))
        )

    def summary(self) -> dict[str, Any]:
        """
        Return model summary for logistic regression.

        Returns
        -------
        dict
            Summary with coefficients, deviance, etc.
        """
        if self.beta is None or self.fitted_values is None or self.X is None:
            raise ValueError("Model must be fit before requesting summary.")

        dev = self.deviance()
        df_resid = self.X.shape[0] - self.X.shape[1]

        return {
            "coefficients": self.beta,
            "fitted_values": self.fitted_values,
            "deviance": dev,
            "residual_df": df_resid,
            "deviance_per_df": dev / df_resid,
            "link_function": self.link.__class__.__name__,
            "iterations": self.iterations
        }

    def print_summary(self) -> None:
        """
        Print formatted logistic regression summary.
        """
        summary = self.summary()

        print("\nLogistic GLM Summary")
        print("=" * 25)
        print(f"Link function     : {summary['link_function']}")
        print(f"Iterations        : {summary['iterations']}")
        print(f"Deviance          : {summary['deviance']:.4f}")
        print(f"Residual DF       : {summary['residual_df']}")
        print(f"Deviance / DF     : {summary['deviance_per_df']:.4f}\n")

        print("Coefficients:")
        for i, coef in enumerate(summary["coefficients"]):
            name = "Intercept" if i == 0 else f"x{i}"
            print(f"  {name:<10}: {coef:.4f}")
