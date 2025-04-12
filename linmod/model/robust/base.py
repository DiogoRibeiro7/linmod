import tempfile
import subprocess
import os
from typing import Callable, Optional, Union
import numpy as np

from linmod.model.robust.se import compute_sandwich_se


def mad(residuals: np.ndarray) -> float:
    """Median Absolute Deviation (robust scale estimate)."""
    return float(np.median(np.abs(residuals - np.median(residuals))) / 0.6745)


class RobustLinearModel:
    """
    Robust linear regression using Iteratively Reweighted Least Squares (IRLS).
    Allows for custom psi functions (e.g., Huber, Tukey).

    Attributes
    ----------
    coefficients : np.ndarray
        Fitted regression coefficients.
    intercept : float
        Intercept term.
    fitted_values : np.ndarray
        Model predictions.
    residuals : np.ndarray
        Residuals of the fit.
    weights : np.ndarray
        Final weights applied during IRLS.
    """

    def __init__(
        self,
        psi: Callable[[np.ndarray], np.ndarray],
        scale_estimator: Callable[[np.ndarray], float] = mad,
        max_iter: int = 50,
        tol: float = 1e-6
    ):
        self.psi = psi
        self.scale_estimator = scale_estimator
        self.max_iter = max_iter
        self.tol = tol

        self.coefficients: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        self.fitted_values: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.X_design_: Optional[np.ndarray] = None
        self.std_errors_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the robust linear regression model via IRLS.

        Parameters
        ----------
        X : np.ndarray
            Predictor matrix, shape (n_samples, n_features)
        y : np.ndarray
            Response vector, shape (n_samples,)
        """
        n, p = X.shape
        X_design = np.column_stack([np.ones(n), X])
        beta = np.linalg.lstsq(X_design, y, rcond=None)[0]

        for _ in range(self.max_iter):
            y_pred = X_design @ beta
            residuals = y - y_pred
            scale = self.scale_estimator(residuals)

            if scale < 1e-8:
                break

            u = residuals / scale
            w = self.psi(u) / u
            w = np.where(np.abs(u) < 1e-8, 1.0, w)

            W = np.diag(w)
            beta_new = np.linalg.pinv(
                X_design.T @ W @ X_design) @ X_design.T @ W @ y

            if np.linalg.norm(beta_new - beta) < self.tol:
                beta = beta_new
                break

            beta = beta_new

        self.X_design_ = X_design
        self.coefficients = beta[1:]
        self.intercept = beta[0]
        self.fitted_values = X_design @ beta
        self.residuals = y - self.fitted_values
        self.weights = w
        if self.residuals is None:
            raise ValueError("Residuals are not available. Fit the model before computing standard errors.")
        if self.weights is None:
            raise ValueError("Weights are not available. Fit the model before computing standard errors.")
        self.std_errors_ = compute_sandwich_se(X, self.residuals, self.weights)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict response values for new input data.

        Parameters
        ----------
        X : np.ndarray
            New input matrix (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model must be fit before calling predict().")
        return self.intercept + X @ self.coefficients

    def summary(self) -> dict[str, Union[float, np.ndarray]]:
        """
        Return model summary statistics.

        Returns
        -------
        dict
            Dictionary with intercept, coefficients, residual statistics, and IRLS weights.
        """
        if self.coefficients is None or self.fitted_values is None:
            raise ValueError("Model must be fit before calling summary().")

        if self.X_design_ is None:
            raise ValueError(
                "X_design_ is not available. Fit the model before calling summary().")
        n = self.X_design_.shape[0]
        if self.X_design_ is None:
            raise ValueError(
                "X_design_ is not available. Fit the model before calling summary().")
        p = self.X_design_.shape[1] - 1

        if self.residuals is None:
            raise ValueError(
                "Residuals are not available. Fit the model before calling summary().")
        rss = np.sum(self.residuals**2)
        rse = np.sqrt(rss / (n - p - 1))
        weights_mean = np.mean(
            self.weights) if self.weights is not None else None

        return {
            "intercept": self.intercept if self.intercept is not None else 0.0,
            "coefficients": self.coefficients,
            "residual_std_error": rse,
            "degrees_of_freedom": n - p - 1,
            "mean_weight": float(weights_mean) if weights_mean is not None else 0.0,
            "n_iter": self.max_iter,
            "weighted": True
        }

    def summary_to_latex(self) -> str:
        """
        Export model summary to LaTeX tabular format.

        Returns
        -------
        str
            LaTeX string with model summary.
        """
        if self.coefficients is None or self.fitted_values is None:
            raise ValueError("Model must be fit before exporting summary.")

        summary = self.summary()
        coef_rows = []

        # Intercept
        coef_rows.append(f"(Intercept) & {summary['intercept']:.4f} \\\\")

        # Coefficients
        for i, coef in enumerate(summary["coefficients"]):
            coef_rows.append(f"x{i+1} & {coef:.4f} \\\\")

        body = "\n".join(coef_rows)
        extra = (
            f"Residual Std. Error & {summary['residual_std_error']:.4f} \\\\\n"
            f"Degrees of Freedom & {summary['degrees_of_freedom']} \\\\\n"
            f"Mean IRLS Weight & {summary['mean_weight']:.4f} \\\\\n"
            f"Max Iterations & {summary['n_iter']} \\\\\n"
        )

        return (
            "\\begin{tabular}{lr}\n"
            "\\toprule\n"
            "Term & Estimate \\\\\n"
            "\\midrule\n"
            f"{body}\n"
            "\\midrule\n"
            f"{extra}"
            "\\bottomrule\n"
            "\\end{tabular}"
        )

    def summary_report_to_latex(self) -> str:
        """
        Generate a complete LaTeX report block for the robust model summary.

        Returns
        -------
        str
            LaTeX report string.
        """
        if self.coefficients is None or self.fitted_values is None:
            raise ValueError("Model must be fit before generating report.")

        summary_table = self.summary_to_latex()

        return (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage[margin=1in]{geometry}\n"
            "\\begin{document}\n\n"
            "\\section*{Robust Linear Model Summary}\n"
            "This report summarizes the results from an M-estimator robust linear regression using IRLS.\n\n"
            f"{summary_table}\n\n"
            "\\end{document}"
        )


    def save_summary_pdf(self, filepath: str) -> None:
        """
        Save the model summary report as a PDF file using LaTeX.

        Parameters
        ----------
        filepath : str
            Full path where the PDF should be saved (must end with .pdf).
        """
        if not filepath.endswith(".pdf"):
            raise ValueError("Filepath must end with '.pdf'.")

        latex_code = self.summary_report_to_latex()

        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, "report.tex")

            # Write the LaTeX code to a temporary .tex file
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(latex_code)

            # Compile LaTeX using pdflatex
            try:
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", tex_path],
                    cwd=tmpdir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    "Failed to compile LaTeX to PDF. Make sure pdflatex is installed."
                ) from exc

            pdf_src = os.path.join(tmpdir, "report.pdf")
            os.replace(pdf_src, filepath)
