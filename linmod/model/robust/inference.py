from typing import Optional
from scipy import stats
import numpy as np


def compute_confidence_intervals(
    coefficients: np.ndarray,
    std_errors: np.ndarray,
    alpha: float = 0.05
) -> np.ndarray:
    """
    Compute confidence intervals for regression coefficients.

    Parameters
    ----------
    coefficients : np.ndarray
        Estimated regression coefficients.
    std_errors : np.ndarray
        Standard errors of the coefficients.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    np.ndarray
        Array of shape (n_features, 2) with lower and upper bounds of CI.
    """
    z = stats.norm.ppf(1 - alpha / 2)
    lower = coefficients - z * std_errors
    upper = coefficients + z * std_errors
    return np.column_stack([lower, upper])


def compute_t_and_p_values(
    coefficients: np.ndarray,
    std_errors: np.ndarray,
    df_residual: Optional[int] = None
) -> dict[str, np.ndarray]:
    """
    Compute t-values and two-sided p-values for regression coefficients.

    Parameters
    ----------
    coefficients : np.ndarray
        Estimated regression coefficients.
    std_errors : np.ndarray
        Standard errors of the coefficients.
    df_residual : int, optional
        Degrees of freedom of the residuals (n - p - 1). Required for t-distribution.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with 't_values' and 'p_values'.
    """
    if len(coefficients) != len(std_errors):
        raise ValueError(
            "Coefficient and standard error arrays must have the same length.")

    t_values = coefficients / std_errors

    if df_residual is not None and df_residual > 0:
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df=df_residual))
    else:
        # fallback to normal approximation
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_values)))

    return {"t_values": t_values, "p_values": p_values}


def build_anova_table(
    y: np.ndarray,
    y_hat: np.ndarray,
    df_model: int,
    df_residual: int
) -> dict[str, Any]:
    """
    Construct ANOVA-like table for robust regression.

    Parameters
    ----------
    y : np.ndarray
        Original response values.
    y_hat : np.ndarray
        Fitted values from model.
    df_model : int
        Degrees of freedom of the model (number of predictors).
    df_residual : int
        Degrees of freedom of the residuals.

    Returns
    -------
    dict
        ANOVA-like table with df, SS, MS, F, p.
    """
    ss_total = np.sum((y - y.mean())**2)
    ss_model = np.sum((y_hat - y.mean())**2)
    ss_resid = np.sum((y - y_hat)**2)

    ms_model = ss_model / df_model
    ms_resid = ss_resid / df_residual
    f_stat = ms_model / ms_resid if ms_resid > 0 else np.nan
    p_value = 1 - stats.f.cdf(f_stat, df_model, df_residual)

    return {
        "df": [df_model, df_residual],
        "SS": [ss_model, ss_resid],
        "MS": [ms_model, ms_resid],
        "F": [f_stat, np.nan],
        "p": [p_value, np.nan]
    }
