import numpy as np


def compute_sandwich_se(X: np.ndarray, residuals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute robust standard errors via the sandwich estimator.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n_samples, n_features), including intercept.
    residuals : np.ndarray
        Residuals from the model.
    weights : np.ndarray
        Weights from the ψ-function, used in IRLS.

    Returns
    -------
    np.ndarray
        Robust standard errors (standard deviations of coefficients).
    """
    # Compute weighted residuals
    u = residuals * weights

    # Bread: (XᵗX)⁻¹
    XtX_inv = np.linalg.pinv(X.T @ X)

    # Meat: Xᵗ diag(u²) X
    S = np.diag(u**2)
    middle = X.T @ S @ X

    # Sandwich: (XᵗX)⁻¹ Xᵗ diag(u²) X (XᵗX)⁻¹
    cov_matrix = XtX_inv @ middle @ XtX_inv
    std_errors = np.sqrt(np.diag(cov_matrix))

    return std_errors
