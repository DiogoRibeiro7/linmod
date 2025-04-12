# linmod/inference.py
import numpy as np
from scipy import stats
from typing import Any

class LinearInferenceMixin:
    def __init__(self):
        self._inference_alpha = 0.05
        self._inference_robust = "HC1"

    def set_inference_options(self, *, alpha: float = 0.05, robust: str = "HC1") -> None:
        """
        Set default inference options for all subsequent calls to .fit().

        Parameters
        ----------
        alpha : float
            Confidence level (e.g. 0.05 for 95% intervals).
        robust : str
            Type of robust SE: 'HC0', 'HC1', 'HC2', 'HC3', 'HC4'.
        """
        self._inference_alpha = alpha
        self._inference_robust = robust

    def compute_inference(
        self,
        beta: np.ndarray,
        X_design: np.ndarray,
        y: np.ndarray,
        residuals: np.ndarray,
        alpha: float = 0.05,
        robust: str = "HC1"
    ) -> dict[str, Any]:
        n, p = X_design.shape
        df_residual = n - p
        rss = np.sum(residuals ** 2)
        tss = np.sum((y - y.mean()) ** 2)
        mse = rss / df_residual
        XtX_inv = np.linalg.pinv(X_design.T @ X_design)

        se_beta = np.sqrt(np.diag(mse * XtX_inv))
        t_stats = beta / se_beta
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_residual))
        t_crit = stats.t.ppf(1 - alpha / 2, df=df_residual)
        ci_lower = beta - t_crit * se_beta
        ci_upper = beta + t_crit * se_beta
        conf_ints = np.column_stack([ci_lower, ci_upper])

        df_model = p - 1
        f_stat = ((tss - rss) / df_model) / mse
        f_p_val = 1 - stats.f.cdf(f_stat, df_model, df_residual)

        robust_se = self._robust_standard_errors(X_design, residuals, robust)

        ssr = tss - rss
        msr = ssr / df_model

        anova_table = {
            "df": np.array([df_model, df_residual]),
            "SS": np.array([ssr, rss]),
            "MS": np.array([msr, mse]),
            "F": np.array([f_stat, np.nan]),
            "p": np.array([f_p_val, np.nan])
        }

        return {
            "std_errors": se_beta[1:],
            "robust_std_errors": robust_se[1:],
            "t_values": t_stats[1:],
            "p_values": p_vals[1:],
            "confidence_intervals": conf_ints[1:],
            "anova_table": anova_table,
            "residual_std_error": np.sqrt(mse),
            "r_squared": 1 - rss / tss,
            "adj_r_squared": 1 - (1 - (1 - rss / tss)) * (n - 1) / df_residual,
            "f_statistic": f_stat,
            "f_p_value": f_p_val,
            "df_residual": df_residual
        }

    def _robust_standard_errors(self, X: np.ndarray, residuals: np.ndarray, hc_type: str = "HC1") -> np.ndarray:
        n, p = X.shape
        hat_matrix = X @ np.linalg.pinv(X.T @ X) @ X.T
        h = np.diag(hat_matrix)
        u = residuals

        if hc_type == "HC0":
            omega = u ** 2
        elif hc_type == "HC1":
            omega = u ** 2 * (n / (n - p))
        elif hc_type == "HC2":
            omega = u ** 2 / (1 - h)
        elif hc_type == "HC3":
            omega = u ** 2 / (1 - h) ** 2
        elif hc_type == "HC4":
            delta = np.minimum(4, h / np.mean(h))
            omega = u ** 2 / (1 - h) ** delta
        else:
            raise ValueError(f"Unknown robust SE type: '{hc_type}'")

        S = np.diag(omega)
        XtX_inv = np.linalg.pinv(X.T @ X)
        cov_hc = XtX_inv @ X.T @ S @ X @ XtX_inv
        return np.sqrt(np.diag(cov_hc))

        