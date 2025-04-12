import pandas as pd
import numpy as np


def compute_weights_from_residuals(residuals: np.ndarray, method: str = "inverse") -> np.ndarray:
    """
    Compute weights based on residuals using the specified weighting method.

    Parameters:
    -----------
    residuals : np.ndarray
        An array of residuals for which weights are to be computed.
    method : str, optional
        The weighting method to use. Options are:
        - "inverse": Computes weights as the inverse of the absolute residuals.
        - "inverse_squared": Computes weights as the inverse of the squared absolute residuals.
        - "huber": Computes weights using the Huber weighting function, which is robust to outliers.
          The Huber method uses a threshold constant `c = 1.345`.

    Returns:
    --------
    np.ndarray
        An array of computed weights corresponding to the input residuals.

    Raises:
    -------
    ValueError
        If an unknown weighting method is specified.
    """
    eps = 1e-8
    abs_res = np.abs(residuals)

    if method == "inverse":
        return 1.0 / (abs_res + eps)
    elif method == "inverse_squared":
        return 1.0 / (abs_res ** 2 + eps)
    elif method == "huber":
        c = 1.345
        return np.where(abs_res <= c, 1.0, c / abs_res)
    else:
        raise ValueError(f"Unknown weighting method: '{method}'")


def hat_matrix(X_design: np.ndarray) -> np.ndarray:
    """
    Compute the hat matrix (projection matrix) from a design matrix.

    The hat matrix H = X(X'X)⁻¹X' projects y onto the column space of X.

    Note:
        Uses Cholesky decomposition when X'X is well-conditioned.
        Raises if the design matrix is ill-conditioned to avoid misleading results.

    Parameters
    ----------
    X_design : np.ndarray
        Design matrix (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Hat matrix H (n_samples, n_samples).
    """
    XtX = X_design.T @ X_design
    cond = np.linalg.cond(XtX)

    if cond > 1e12:
        raise np.linalg.LinAlgError(
            "Design matrix is ill-conditioned. Cannot compute reliable hat matrix.")

    try:
        # Use Cholesky when possible
        L = np.linalg.cholesky(XtX)
        XtX_inv = np.linalg.inv(L.T) @ np.linalg.inv(L)
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse if Cholesky fails
        XtX_inv = np.linalg.pinv(XtX)

    return X_design @ XtX_inv @ X_design.T


def leverage_scores(X_design: np.ndarray) -> np.ndarray:
    """
    Compute the leverage scores for a given design matrix.

    Leverage scores are the diagonal elements of the hat matrix, which
    measure the influence of each observation on the fitted values in
    a linear regression model.

    Args:
        X_design (np.ndarray): The design matrix (2D array) where rows
            represent observations and columns represent predictors.

    Returns:
        np.ndarray: A 1D array containing the leverage scores for each
        observation.
    """
    return np.diag(hat_matrix(X_design))


def studentized_residuals(residuals: np.ndarray, leverage: np.ndarray, sigma2: float) -> np.ndarray:
    """
    Calculate the studentized residuals for a regression model.

    Studentized residuals are used to identify outliers in regression analysis
    by standardizing the residuals while accounting for the leverage of each
    observation.

    Parameters:
        residuals (np.ndarray): A 1D array of residuals from the regression model
            with shape (n,), where n is the number of observations.
        leverage (np.ndarray): A 1D array of leverage values for each observation
            with shape (n,).
        sigma2 (float): The estimated variance of the residuals.

    Returns:
        np.ndarray: A 1D array of studentized residuals with shape (n,).
    """
    eps = 1e-8  # Small constant to prevent division by zero
    return residuals / np.sqrt(sigma2 * (1 - leverage + eps))


def cooks_distance(residuals: np.ndarray, leverage: np.ndarray, p: int, sigma2: float) -> np.ndarray:
    """
    Compute Cook's distance for each observation.

    Parameters
    ----------
    residuals : np.ndarray
        Residual vector.
    leverage : np.ndarray
        Leverage values (diagonal of hat matrix).
    p : int
        Number of predictors + 1 (including intercept).
    sigma2 : float
        Estimated variance of residuals.

    Returns
    -------
    np.ndarray
        Cook's distance per observation.
    """
    return (residuals ** 2 / (p * sigma2)) * (leverage / (1 - leverage) ** 2)


def dffits(residuals: np.ndarray, leverage: np.ndarray, sigma2: float) -> np.ndarray:
    """
    Compute DFFITS: influence of each point on its own fitted value.

    Parameters
    ----------
    residuals : np.ndarray
        Residual vector.
    leverage : np.ndarray
        Leverage values.
    sigma2 : float
        Estimated residual variance.

    Returns
    -------
    np.ndarray
        DFFITS values.
    """
    return studentized_residuals(residuals, leverage, sigma2) * np.sqrt(leverage / (1 - leverage))


def influence_summary(
    X_design: np.ndarray,
    residuals: np.ndarray,
    sigma2: float
) -> pd.DataFrame:
    """
    Create a summary table with influence diagnostics.

    Parameters
    ----------
    X_design : np.ndarray
        Design matrix (with intercept).
    residuals : np.ndarray
        Residual vector.
    sigma2 : float
        Estimated residual variance.

    Returns
    -------
    pd.DataFrame
        A DataFrame with leverage, studentized residuals,
        Cook's distance and DFFITS for each observation.
    """
    n, p = X_design.shape

    leverage = leverage_scores(X_design)
    stud_resid = studentized_residuals(residuals, leverage, sigma2)
    cooks = cooks_distance(residuals, leverage, p, sigma2)
    dffits_ = dffits(residuals, leverage, sigma2)

    return pd.DataFrame({
        "leverage": leverage,
        "studentized_resid": stud_resid,
        "cooks_distance": cooks,
        "dffits": dffits_
    })
