# linmod/model/robust_psi.py

import numpy as np

def psi_huber(u: np.ndarray, c: float = 1.345) -> np.ndarray:
    """
    Huber's psi function: linear for small residuals, constant for large ones.

    Parameters
    ----------
    u : np.ndarray
        Standardized residuals (residuals / scale).
    c : float
        Threshold parameter.

    Returns
    -------
    np.ndarray
        Psi values.
    """
    return np.where(np.abs(u) <= c, u, c * np.sign(u))


def psi_tukey(u: np.ndarray, c: float = 4.685) -> np.ndarray:
    """
    Tukey's bisquare psi function: redescending influence.

    Parameters
    ----------
    u : np.ndarray
        Standardized residuals.
    c : float
        Tuning constant.

    Returns
    -------
    np.ndarray
        Psi values.
    """
    u_abs = np.abs(u)
    mask = u_abs < c
    out = np.zeros_like(u)
    out[mask] = u[mask] * (1 - (u[mask] / c) ** 2) ** 2
    return out


def psi_andrews(u: np.ndarray, c: float = 1.339) -> np.ndarray:
    """
    Andrews' sine psi function.

    Parameters
    ----------
    u : np.ndarray
        Standardized residuals.
    c : float
        Tuning constant.

    Returns
    -------
    np.ndarray
        Psi values.
    """
    u_abs = np.abs(u)
    mask = u_abs < np.pi * c
    out = np.zeros_like(u)
    out[mask] = np.sin(u[mask] / c)
    return out


def psi_hampel(u: np.ndarray, a: float = 2.0, b: float = 4.0, c: float = 8.0) -> np.ndarray:
    """
    Hampel's three-part redescending psi function.

    Parameters
    ----------
    u : np.ndarray
        Standardized residuals.
    a, b, c : float
        Thresholds for influence decline.

    Returns
    -------
    np.ndarray
        Psi values.
    """
    abs_u = np.abs(u)
    psi = np.zeros_like(u)

    # linear part
    mask_a = abs_u <= a
    psi[mask_a] = u[mask_a]

    # declining part
    mask_b = (abs_u > a) & (abs_u <= b)
    psi[mask_b] = a * np.sign(u[mask_b])

    # redescending part
    mask_c = (abs_u > b) & (abs_u <= c)
    psi[mask_c] = a * (c - abs_u[mask_c]) / (c - b) * np.sign(u[mask_c])

    # zero influence beyond c
    return psi
