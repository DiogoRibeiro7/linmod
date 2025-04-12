# linmod/plots.py

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from linmod.utils import hat_matrix, cooks_distance


def plot_residuals_vs_fitted(fitted: np.ndarray, residuals: np.ndarray) -> None:
    """Scatter plot of residuals vs fitted values."""
    plt.figure()
    plt.scatter(fitted, residuals, edgecolor='k', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', lw=1)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.grid(True)
    plt.show()


def plot_standardized_residuals_leverage(X_design: np.ndarray, residuals: np.ndarray) -> None:
    """Plot of standardized residuals vs leverage with Cook's distance contours."""
    n = X_design.shape[0]
    h = np.diag(hat_matrix(X_design))
    mse = np.sum(residuals**2) / (n - X_design.shape[1])
    std_resid = residuals / np.sqrt(mse * (1 - h))

    fig, ax = plt.subplots()
    ax.scatter(h, std_resid, alpha=0.7, edgecolor='k')
    ax.axhline(0, linestyle='--', color='grey', linewidth=1)
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized Residuals")
    ax.set_title("Standardized Residuals vs Leverage")

    # Add Cook's distance contours
    p = X_design.shape[1]
    for level in [0.5, 1.0]:
        boundary = np.sqrt(level * p * (1 - h) / h)
        ax.plot(h, boundary, linestyle='--', color='red', linewidth=0.7)
        ax.plot(h, -boundary, linestyle='--', color='red', linewidth=0.7)
        ax.annotate(f"Cook’s D={level}", xy=(max(h)*0.9, max(boundary)*0.9),
                    color='red', fontsize=8)

    plt.grid(True)
    plt.show()


def plot_qq(residuals: np.ndarray) -> None:
    """QQ-plot of residuals against the normal distribution."""
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Normal Q–Q Plot")
    plt.grid(True)
    plt.show()


def plot_component_plus_residuals(X: np.ndarray, residuals: np.ndarray, coefficients: np.ndarray, feature_names: list[str] = None) -> None:
    """
    Component + Residual (Partial Residual) plots for each predictor.

    Parameters
    ----------
    X : np.ndarray
        Matrix of original predictors (without intercept).
    residuals : np.ndarray
        Residuals from the fitted model.
    coefficients : np.ndarray
        Estimated coefficients (excluding intercept).
    feature_names : list of str
        Optional names of features for labeling.
    """
    p = X.shape[1]
    feature_names = feature_names or [f"x{i+1}" for i in range(p)]

    for i in range(p):
        cr_values = residuals + X[:, i] * coefficients[i]
        plt.figure()
        plt.scatter(X[:, i], cr_values, alpha=0.7, edgecolor='k')
        plt.xlabel(feature_names[i])
        plt.ylabel("Component + Residual")
        plt.title(f"CR Plot: {feature_names[i]}")
        plt.axhline(0, color='grey', linestyle='--')
        plt.grid(True)
        plt.show()
