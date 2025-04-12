# linmod/diagnostics/normality.py

import math
import numpy as np
from scipy.stats import shapiro

def shapiro_wilk_test(residuals: np.ndarray, alpha: float = 0.05) -> dict[str, float | str]:
    """
    Perform the Shapiro–Wilk test for normality.

    Parameters
    ----------
    residuals : np.ndarray
        Vector of residuals.
    alpha : float
        Significance level.

    Returns
    -------
    dict with W statistic, p-value, and interpretation.
    """
    stat, p_value = shapiro(residuals)
    interpretation = (
        "Residuals are not normally distributed"
        if p_value < alpha else
        "Residuals appear normally distributed"
    )
    return {
        "W": stat,
        "p-value": p_value,
        "interpretation": interpretation
    }


def normality_heuristic(x: list[float], alpha: float = 0.05) -> dict[str, float | str]:
    """
    Approximate normality test using skewness and kurtosis (base Python only).

    Parameters
    ----------
    x : list[float]
        Sample data.
    alpha : float
        Significance threshold.

    Returns
    -------
    dict with skewness, kurtosis, and interpretation.
    """
    n = len(x)
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / n
    std = var ** 0.5

    skewness = sum((xi - mean) ** 3 for xi in x) / (n * std**3)
    kurtosis = sum((xi - mean) ** 4 for xi in x) / (n * std**4) - 3  # excess

    interpretation = (
        "Possible non-normality (|skew| > 1 or kurtosis > 1)"
        if abs(skewness) > 1 or abs(kurtosis) > 1
        else "Residuals appear approximately normal"
    )

    return {
        "skewness": skewness,
        "excess kurtosis": kurtosis,
        "interpretation": interpretation
    }


def dagostino_k2(x: list[float], alpha: float = 0.05) -> dict[str, float | str]:
    """
    Approximate D’Agostino’s K² test for normality (base Python implementation).

    Parameters
    ----------
    x : list[float]
        Sample data.
    alpha : float
        Significance level.

    Returns
    -------
    dict with K² statistic, p-value (approximate), skewness, and kurtosis.
    """
    n = len(x)
    if n < 8:
        raise ValueError("D'Agostino's K² test requires at least 8 observations.")

    mean = sum(x) / n
    m2 = sum((xi - mean) ** 2 for xi in x) / n
    m3 = sum((xi - mean) ** 3 for xi in x) / n
    m4 = sum((xi - mean) ** 4 for xi in x) / n

    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2) - 3

    Y = skew * math.sqrt((n + 1) * (n + 3) / (6.0 * (n - 2)))
    beta2 = 3.0 * (n**2 + 27 * n - 70) * (n + 1) * (n + 3) / (
        (n - 2) * (n + 5) * (n + 7) * (n + 9)
    )
    W2 = -1 + math.sqrt(2 * (beta2 - 1))
    delta = 1 / math.sqrt(math.log(math.sqrt(W2)))
    alpha_skew = math.sqrt(2 / (W2 - 1))
    Z1 = delta * math.log(Y / alpha_skew + math.sqrt((Y / alpha_skew) ** 2 + 1))

    E = 3.0 * (n - 1) / (n + 1)
    Var = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))
    X = (kurt - E) / math.sqrt(Var)
    B = (
        6.0
        * (n**2 - 5 * n + 2)
        / ((n + 7) * (n + 9))
        * math.sqrt(6.0 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
    )
    A = 6.0 + 8.0 / B * (2.0 / B + math.sqrt(1 + 4.0 / (B**2)))
    Z2 = (
        (1 - 2 / (9 * A))
        - ((1 - 2 / A) / (1 + X * math.sqrt(2 / (A - 4)))) ** (1 / 3)
    ) / math.sqrt(2 / (9 * A))

    K2 = Z1**2 + Z2**2
    p = math.exp(-0.5 * K2)  # approx chi2.sf(K2, df=2)

    interpretation = (
        "Residuals are not normally distributed"
        if p < alpha else
        "Residuals appear normally distributed"
    )

    return {
        "K² statistic": K2,
        "p-value (approx)": p,
        "interpretation": interpretation,
        "skewness": skew,
        "kurtosis": kurt
    }
