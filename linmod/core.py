import numpy as np
from scipy import stats
from typing import Any


class LinearModel:
    def __init__(self) -> None:
        self.coefficients: np.ndarray | None = None
        self.fitted_values: np.ndarray | None = None
        self.residuals: np.ndarray | None = None
        self.df_residual: int | None = None
        self.residual_std_error: float | None = None
        self.r_squared: float | None = None
        self.adj_r_squared: float | None = None
        self.std_errors: np.ndarray | None = None
        self.t_values: np.ndarray | None = None
        self.p_values: np.ndarray | None = None
        self.f_statistic: float | None = None
        self.f_p_value: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit linear model using normal equation and compute statistics.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        y : np.ndarray
            Target vector of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        X_design = np.hstack([np.ones((n_samples, 1)), X])
        p = X_design.shape[1]

        # Normal equation
        XtX_inv = np.linalg.pinv(X_design.T @ X_design)
        beta = XtX_inv @ X_design.T @ y
        y_pred = X_design @ beta
        residuals = y - y_pred

        rss = np.sum(residuals ** 2)
        tss = np.sum((y - y.mean()) ** 2)
        df_residual = n_samples - p
        df_model = p - 1

        mse = rss / df_residual
        se_beta = np.sqrt(np.diag(mse * XtX_inv))

        t_stats = beta / se_beta
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_residual))

        # F-statistic
        ms_model = (tss - rss) / df_model
        f_stat = ms_model / mse
        f_p_val = 1 - stats.f.cdf(f_stat, df_model, df_residual)

        self.coefficients = beta
        self.fitted_values = y_pred
        self.residuals = residuals
        self.df_residual = df_residual
        self.residual_std_error = np.sqrt(mse)
        self.r_squared = 1 - rss / tss
        self.adj_r_squared = 1 - (1 - self.r_squared) * (n_samples - 1) / df_residual
        self.std_errors = se_beta
        self.t_values = t_stats
        self.p_values = p_vals
        self.f_statistic = f_stat
        self.f_p_value = f_p_val

    def summary(self) -> dict[str, Any]:
        """
        Returns a summary of the model similar to R's `summary(lm(...))`.
        """
        if self.coefficients is None:
            raise ValueError("Model is not fit yet.")

        return {
            "coefficients": self.coefficients,
            "std_errors": self.std_errors,
            "t_values": self.t_values,
            "p_values": self.p_values,
            "fitted.values": self.fitted_values,
            "residuals": self.residuals,
            "df.residual": self.df_residual,
            "residual std error": self.residual_std_error,
            "r.squared": self.r_squared,
            "adj.r.squared": self.adj_r_squared,
            "f.statistic": self.f_statistic,
            "f.p.value": self.f_p_value
        }
