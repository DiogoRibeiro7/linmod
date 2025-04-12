import numpy as np
from scipy import stats
from typing import Union, Dict

def inference_summary(
    coefficients: np.ndarray,
    std_errors: np.ndarray,
    alpha: float = 0.05,
    df_resid: Union[int, None] = None
) -> Dict[str, np.ndarray]:
    """
    Compute inference statistics for model coefficients: t-values, p-values, and confidence intervals.

    Parameters
    ----------
    coefficients : np.ndarray
        Coefficient estimates (1D array).
    std_errors : np.ndarray
        Corresponding standard errors (1D array).
    alpha : float
        Significance level for confidence intervals (e.g., 0.05 for 95% CI).
    df_resid : int, optional
        Degrees of freedom for residuals. If not provided, defaults to len(coefficients) - 1.

    Returns
    -------
    dict
        Dictionary containing t-values, p-values, and confidence intervals.
    """
    if df_resid is None:
        df_resid = len(coefficients) - 1

    t_values = coefficients / std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df=df_resid))
    z = stats.t.ppf(1 - alpha / 2, df=df_resid)
    ci_lower = coefficients - z * std_errors
    ci_upper = coefficients + z * std_errors

    return {
        "t_values": t_values,
        "p_values": p_values,
        "confidence_intervals": np.column_stack((ci_lower, ci_upper))
    }


def anova_like_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p: int
) -> Dict[str, Union[float, np.ndarray, Dict[str, np.ndarray]]]:
    """
    Compute an ANOVA-like summary for a regression model.

    Parameters
    ----------
    y_true : np.ndarray
        Observed response values.
    y_pred : np.ndarray
        Fitted response values from the model.
    p : int
        Number of predictors (excluding intercept).

    Returns
    -------
    dict
        Dictionary containing residual std error, R², adjusted R², F-statistic,
        and a nested ANOVA table with degrees of freedom, sum of squares,
        mean squares, F-statistic, and p-value.
    """
    n = len(y_true)
    residuals = y_true - y_pred

    rss = np.sum(residuals ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    mss = tss - rss

    df_model = p
    df_resid = n - p - 1

    ms_model = mss / df_model if df_model > 0 else 0.0
    ms_resid = rss / df_resid if df_resid > 0 else np.nan

    f_statistic = ms_model / ms_resid if ms_resid > 0 else np.nan
    f_p_value = 1 - stats.f.cdf(f_statistic, df_model, df_resid) if not np.isnan(f_statistic) else np.nan

    r_squared = float(1 - rss / tss)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / df_resid
    residual_std_error = np.sqrt(ms_resid)

    return {
        "residual_std_error": residual_std_error,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "f_statistic": f_statistic,
        "f_p_value": f_p_value,
        "df_model": df_model,
        "df_residual": df_resid,
        "anova_table": {
            "df": np.array([df_model, df_resid]),
            "SS": np.array([mss, rss]),
            "MS": np.array([ms_model, ms_resid]),
            "F": np.array([f_statistic, np.nan]),
            "p": np.array([f_p_value, np.nan])
        }
    }