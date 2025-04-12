# linmod/diagnostics/heteroscedasticity.py

import numpy as np
from scipy import stats
from typing import Union

def white_test(X: np.ndarray, residuals: np.ndarray, alpha: float = 0.05) -> dict[str, float | str]:
    n, k = X.shape
    u2 = residuals**2

    Z = [np.ones(n)]
    Z.extend(X.T)
    Z.extend((X[:, i] * X[:, j] for i in range(k) for j in range(i, k)))
    Z = np.column_stack(Z)

    beta_aux = np.linalg.lstsq(Z, u2, rcond=None)[0]
    y_hat = Z @ beta_aux
    ssr = np.sum((y_hat - u2.mean()) ** 2)
    lm = n * ssr / np.sum((u2 - u2.mean())**2)

    df = Z.shape[1] - 1
    p_value = 1 - stats.chi2.cdf(lm, df)

    interpretation = (
        "Evidence of heteroscedasticity" if p_value < alpha else "No evidence of heteroscedasticity"
    )

    return {"LM": lm, "df": df, "p-value": p_value, "interpretation": interpretation}


def breusch_pagan_test(X: np.ndarray, residuals: np.ndarray, alpha: float = 0.05) -> dict[str, float | str]:
    n, k = X.shape
    u2 = residuals**2

    beta_aux = np.linalg.lstsq(X, u2, rcond=None)[0]
    y_hat = X @ beta_aux
    ssr = np.sum((y_hat - u2.mean()) ** 2)

    lm = 0.5 * n * ssr / np.var(u2, ddof=1)
    df = k - 1
    p_value = 1 - stats.chi2.cdf(lm, df)

    interpretation = (
        "Evidence of heteroscedasticity" if p_value < alpha else "No evidence of heteroscedasticity"
    )

    return {"LM": lm, "df": df, "p-value": p_value, "interpretation": interpretation}


def goldfeld_quandt_test(X: np.ndarray, y: np.ndarray, sort_by: int = 1, drop_fraction: float = 0.2,
                         alpha: float = 0.05) -> dict[str, Union[float, np.float64, np.ndarray, str]]:
    n = len(y)
    sorted_idx = np.argsort(X[:, sort_by])
    X_sorted = X[sorted_idx]
    y_sorted = y[sorted_idx]

    drop = int(drop_fraction * n)
    split = (n - drop) // 2

    X1, y1 = X_sorted[:split], y_sorted[:split]
    X2, y2 = X_sorted[-split:], y_sorted[-split:]

    def rss(X_part, y_part):
        Xb = np.hstack([np.ones((len(X_part), 1)), X_part])
        beta = np.linalg.lstsq(Xb, y_part, rcond=None)[0]
        residuals = y_part - Xb @ beta
        return np.sum(residuals**2)

    rss1 = rss(X1, y1)
    rss2 = rss(X2, y2)

    f_stat = max(rss1, rss2) / min(rss1, rss2)
    df1 = df2 = split - X.shape[1] - 1
    p = 1 - stats.f.cdf(f_stat, df1, df2) if df1 > 0 and df2 > 0 else float("nan")

    interpretation = "Evidence of heteroscedasticity" if p < alpha else "No evidence of heteroscedasticity"

    return {
        "F statistic": f_stat,
        "df1": df1,
        "df2": df2,
        "p-value": p,
        "interpretation": interpretation
    }


def park_test(X: np.ndarray, residuals: np.ndarray, predictor_index: int = 0, alpha: float = 0.05) -> dict[str, float | str]:
    u2 = residuals**2
    log_u2 = np.log(u2 + 1e-8)
    x = X[:, predictor_index]

    x_mean = np.mean(x)
    log_u2_mean = np.mean(log_u2)

    cov = np.sum((x - x_mean) * (log_u2 - log_u2_mean))
    var_x = np.sum((x - x_mean) ** 2)

    slope = cov / var_x
    intercept = log_u2_mean - slope * x_mean

    residuals = log_u2 - (intercept + slope * x)
    sse = np.sum(residuals**2)
    se_slope = np.sqrt(sse / (len(x) - 2) / var_x)

    t_stat = slope / se_slope
    p = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(x) - 2))

    interpretation = "Evidence of heteroscedasticity" if p < alpha else "No evidence of heteroscedasticity"

    return {
        "slope": slope,
        "t statistic": t_stat,
        "p-value": p,
        "interpretation": interpretation
    }


def glejser_test(X: np.ndarray, residuals: np.ndarray, predictor_index: int = 0,
                 transform: str = "sqrt", alpha: float = 0.05) -> dict[str, float | str]:
    y = np.abs(residuals)
    x = X[:, predictor_index]

    if transform == "raw":
        x_t = x
    elif transform == "sqrt":
        x_t = np.sqrt(np.abs(x))
    elif transform == "inverse":
        x_t = 1 / (np.abs(x) + 1e-8)
    else:
        raise ValueError("Unknown transform: choose 'raw', 'sqrt', or 'inverse'.")

    x_mean = np.mean(x_t)
    y_mean = np.mean(y)

    cov = np.sum((x_t - x_mean) * (y - y_mean))
    var_x = np.sum((x_t - x_mean) ** 2)

    slope = cov / var_x
    intercept = y_mean - slope * x_mean

    residuals = y - (intercept + slope * x_t)
    sse = np.sum(residuals ** 2)
    se_slope = np.sqrt(sse / (len(x_t) - 2) / var_x)

    t_stat = slope / se_slope
    p = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(x_t) - 2))

    interpretation = "Evidence of heteroscedasticity" if p < alpha else "No evidence of heteroscedasticity"

    return {
        "slope": slope,
        "t statistic": t_stat,
        "p-value": p,
        "transformation": transform,
        "interpretation": interpretation
    }


def variance_power_test(
    X: np.ndarray,
    residuals: np.ndarray,
    predictor_index: int = 0,
    alpha: float = 0.05
) -> dict[str, float | str]:
    """
    Estimate power model: Var(u²) = x^γ

    Parameters
    ----------
    X : np.ndarray
        Design matrix without intercept (n_samples, n_features)
    residuals : np.ndarray
        Residuals from a fitted model.
    predictor_index : int
        Index of predictor to use.
    alpha : float
        Significance level.

    Returns
    -------
    dict with gamma estimate, t-statistic, p-value, and interpretation.
    """
    x = X[:, predictor_index]
    u2 = residuals**2

    # Evita log(0) e divisão por zero
    x_safe = np.where(x <= 0, 1e-8, x)
    log_x = np.log(x_safe)
    log_u2 = np.log(u2 + 1e-8)

    x_mean = np.mean(log_x)
    y_mean = np.mean(log_u2)

    cov = np.sum((log_x - x_mean) * (log_u2 - y_mean))
    var_x = np.sum((log_x - x_mean)**2)

    gamma = cov / var_x
    intercept = y_mean - gamma * x_mean

    y_pred = intercept + gamma * log_x
    residuals_gamma = log_u2 - y_pred
    sse = np.sum(residuals_gamma**2)
    se_gamma = np.sqrt(sse / (len(x) - 2) / var_x)

    t_stat = gamma / se_gamma
    p = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(x) - 2))

    interpretation = (
        "Variance appears to follow a power function of the predictor"
        if p < alpha else "No strong power-form variance relationship"
    )

    return {
        "gamma (slope)": gamma,
        "t statistic": t_stat,
        "p-value": p,
        "interpretation": interpretation
    }
