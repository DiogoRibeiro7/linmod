# linmod/glm/poisson.py

import numpy as np
from linmod.glm.base import BaseGLM
from linmod.glm.links import LogLink


class PoissonRegressionGLM(BaseGLM):
    """
    Poisson GLM with log link for count data.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-6) -> None:
        super().__init__(link=LogLink(), max_iter=max_iter, tol=tol)

    def _variance(self, mu: np.ndarray) -> np.ndarray:
        return mu

    def deviance(self, y: np.ndarray, mu: np.ndarray | None = None) -> float:
        """
        Compute the total deviance.

        Parameters
        ----------
        y : np.ndarray
            Observed response values.
        mu : np.ndarray, optional
            Predicted means. If None, uses fitted values.

        Returns
        -------
        float
            Deviance value.
        """
        if mu is None:
            if self.fitted_values is None:
                raise ValueError("Model must be fit before computing deviance.")
            mu = self.fitted_values

        eps = 1e-8
        y = np.clip(y, eps, None)
        mu = np.clip(mu, eps, None)

        return 2 * np.sum(y * np.log(y / mu) - (y - mu))

    def deviance_residuals(self, y: np.ndarray, mu: np.ndarray | None = None) -> np.ndarray:
        """
        Compute deviance residuals.

        Parameters
        ----------
        y : np.ndarray
            Observed response values.
        mu : np.ndarray, optional
            Predicted means. If None, uses fitted values.

        Returns
        -------
        np.ndarray
            Deviance residuals.
        """
        if mu is None:
            if self.fitted_values is None:
                raise ValueError("Model must be fit before computing residuals.")
            mu = self.fitted_values

        eps = 1e-8
        y = np.clip(y, eps, None)
        mu = np.clip(mu, eps, None)

        d_i = 2 * (y * np.log(y / mu) - (y - mu))
        return np.sign(y - mu) * np.sqrt(d_i)

    def summary(self) -> dict[str, Any]:
        """
        Return model summary including deviance and coefficients.

        Returns
        -------
        dict[str, Any]
            Dictionary with coefficients, fitted values, deviance, etc.
        """
        if self.coefficients is None or self.fitted_values is None or self.X_design_ is None:
            raise ValueError("Model must be fit before requesting summary.")

        dev = self.deviance(self.y_)
        df_resid = self.X_design_.shape[0] - self.X_design_.shape[1]

        return {
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "fitted_values": self.fitted_values,
            "deviance": dev,
            "degrees_of_freedom": df_resid,
            "deviance_per_df": dev / df_resid,
            "link_function": self.link.__class__.__name__,
            "iterations": self.iterations
        }
        
    def print_summary(self) -> None:
        """
        Print formatted model summary to stdout.
        """
        summary = self.summary()
        print("\nPoisson GLM Summary")
        print("=" * 25)
        print(f"Link function     : {summary['link_function']}")
        print(f"Iterations        : {summary['iterations']}")
        print(f"Deviance          : {summary['deviance']:.4f}")
        print(f"Residual DF       : {summary['degrees_of_freedom']}")
        print(f"Deviance / DF     : {summary['deviance_per_df']:.4f}\n")
        print("Coefficients:")
        for i, coef in enumerate(summary["coefficients"]):
            name = "Intercept" if i == 0 else f"x{i}"
            print(f"  {name:<10}: {coef:.4f}")


