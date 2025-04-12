# linmod/diagnostics/functional_form.py

import numpy as np
from scipy import stats

def reset_test(
    X_design: np.ndarray,
    y: np.ndarray,
    fitted_values: np.ndarray,
    residuals: np.ndarray,
    max_power: int = 3,
    alpha: float = 0.05
) -> dict[str, float | str]:
    """
    Ramsey RESET test for functional form misspecification.

    Parameters
    ----------
    X_design : np.ndarray
        Design matrix including intercept.
    y : np.ndarray
        Original response variable.
    fitted_values : np.ndarray
        Fitted values from the model.
    residuals : np.ndarray
        Residuals from the fitted model.
    max_power : int
        Maximum power of fitted values to include (e.g., 3 includes ŷ², ŷ³).
    alpha : float
        Significance level.

    Returns
    -------
    dict with F statistic, df, p-value, and interpretation.
    """
    n = len(y)
    powers = [fitted_values ** p for p in range(2, max_power + 1)]
    X_aug = np.column_stack([X_design] + powers)

    beta_aug = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    y_aug_hat = X_aug @ beta_aug
    residuals_aug = y - y_aug_hat

    rss_base = np.sum(residuals ** 2)
    rss_aug = np.sum(residuals_aug ** 2)

    df1 = max_power - 1
    df2 = n - X_aug.shape[1]
    f_stat = ((rss_base - rss_aug) / df1) / (rss_aug / df2)
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    interpretation = (
        "Possible functional form misspecification"
        if p_value < alpha else "No strong evidence of misspecification"
    )

    return {
        "F statistic": f_stat,
        "df1": df1,
        "df2": df2,
        "p-value": p_value,
        "interpretation": interpretation
    }


def harvey_collier_test(
    fitted_values: np.ndarray,
    residuals: np.ndarray,
    alpha: float = 0.05
) -> dict[str, float | str]:
    """
    Harvey–Collier test for linear functional form.

    Parameters
    ----------
    fitted_values : np.ndarray
        Fitted values from the model.
    residuals : np.ndarray
        Residuals from the fitted model.
    alpha : float
        Significance level.

    Returns
    -------
    dict with t statistic, p-value, and interpretation.
    """
    y = fitted_values + residuals
    y_hat = fitted_values
    Z = np.column_stack([np.ones(len(y_hat)), y_hat])

    beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    y_pred = Z @ beta
    residuals_new = y - y_pred

    sse = np.sum(residuals_new ** 2)
    mse = sse / (len(y) - 2)
    var_beta = mse * np.linalg.inv(Z.T @ Z)
    t_stat = beta[1] / np.sqrt(var_beta[1, 1])
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(y) - 2))

    interpretation = (
        "Possible linear functional form misspecification"
        if p_value < alpha else "No evidence of linear form misfit"
    )

    return {
        "t statistic": t_stat,
        "p-value": p_value,
        "interpretation": interpretation
    }


def white_nonlinearity_test(
    X_design: np.ndarray,
    fitted_values: np.ndarray,
    residuals: np.ndarray,
    alpha: float = 0.05
) -> dict[str, float | str]:
    """
    White’s test for non-linear specification.

    Parameters
    ----------
    X_design : np.ndarray
        Design matrix including intercept.
    fitted_values : np.ndarray
        Fitted values from the model.
    residuals : np.ndarray
        Residuals from the model.
    alpha : float
        Significance level.

    Returns
    -------
    dict with F statistic, df, p-value, and interpretation.
    """
    X = X_design[:, 1:]  # exclude intercept
    y = fitted_values + residuals
    n, k = X.shape

    nonlinear_terms = []
    for i in range(k):
        nonlinear_terms.append(X[:, i] ** 2)
        for j in range(i + 1, k):
            nonlinear_terms.append(X[:, i] * X[:, j])

    X_aug = np.column_stack([X_design] + nonlinear_terms)
    df1 = X_aug.shape[1] - X_design.shape[1]
    df2 = n - X_aug.shape[1]

    beta_full = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    rss_full = np.sum((y - X_aug @ beta_full) ** 2)
    rss_base = np.sum(residuals ** 2)

    f_stat = ((rss_base - rss_full) / df1) / (rss_full / df2)
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    interpretation = (
        "Model may be missing nonlinear structure"
        if p_value < alpha else "No evidence of nonlinear functional form"
    )

    return {
        "F statistic": f_stat,
        "df1": df1,
        "df2": df2,
        "p-value": p_value,
        "interpretation": interpretation
    }
