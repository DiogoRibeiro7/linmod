from ace_tools_open import display_dataframe_to_user
import numpy as np
import pandas as pd

def compute_leverage(X: np.ndarray) -> np.ndarray:
    """Compute leverage (hat values) from design matrix X (including intercept)."""
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    return np.diag(H)

def compute_cooks_distance(X: np.ndarray, residuals: np.ndarray, mse: float) -> np.ndarray:
    """Compute Cook's distance for each observation."""
    h = compute_leverage(X)
    return (residuals ** 2 / (X.shape[1] * mse)) * (h / (1 - h) ** 2)

def compute_dffits(X: np.ndarray, residuals: np.ndarray, mse: float) -> np.ndarray:
    """Compute DFFITS for each observation."""
    h = compute_leverage(X)
    return residuals * np.sqrt(h / (mse * (1 - h)))

def compute_dfbetas(X: np.ndarray, residuals: np.ndarray, mse: float) -> np.ndarray:
    """Compute DFBETAS for each coefficient and observation."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    h = compute_leverage(X)
    dfbetas = []
    for i in range(X.shape[0]):
        xi = X[i, :].reshape(-1, 1)
        hi = h[i]
        ri = residuals[i]
        denom = mse * (1 - hi)
        influence = (XtX_inv @ xi).flatten() * ri / np.sqrt(denom)
        dfbetas.append(influence)
    return np.array(dfbetas)

if __name__ == "__main__":
    # Example with mock data
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 3)
    X_design = np.column_stack([np.ones(n), X])
    beta = np.array([1.0, 0.5, -0.3, 0.2])
    y = X_design @ beta + np.random.randn(n)
    residuals = y - X_design @ beta
    mse = np.mean(residuals ** 2)

    leverage = compute_leverage(X_design)
    cooks_d = compute_cooks_distance(X_design, residuals, mse)
    dffits = compute_dffits(X_design, residuals, mse)
    dfbetas = compute_dfbetas(X_design, residuals, mse)

    influence_df = pd.DataFrame({
        "Leverage": leverage,
        "Cook's Distance": cooks_d,
        "DFFITS": dffits
    })
    display_dataframe_to_user("Influence Diagnostics Summary", influence_df.round(4))
