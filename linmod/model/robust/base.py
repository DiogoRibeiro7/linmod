import tempfile
import subprocess
import os
from typing import Callable, Optional, Union
import numpy as np

from linmod.model.robust.se import compute_sandwich_se
from linmod.model.robust.summary import inference_summary, anova_like_summary


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
        Fit the robust linear model to the given data.

        This method estimates the model coefficients using an iterative reweighted 
        least squares (IRLS) approach, which minimizes the influence of outliers 
        by applying a robust weighting function.

        Parameters:
        -----------
        X : np.ndarray
            The design matrix of shape (n_samples, n_features), where each row 
            represents a sample and each column represents a feature.
        y : np.ndarray
            The response vector of shape (n_samples,), containing the target values.

        Returns:
        --------
        None
            The method updates the following attributes of the object:
            - `coefficients` : np.ndarray
                The estimated coefficients for the features.
            - `intercept` : float
                The estimated intercept term.
            - `fitted_values` : np.ndarray
                The predicted values based on the fitted model.
            - `residuals` : np.ndarray
                The residuals (differences between observed and predicted values).
            - `weights` : np.ndarray
                The final weights applied to each observation.
            - `X_design_` : np.ndarray
                The design matrix with an added intercept column.

        Notes:
        ------
        - The method stops iterating when the change in coefficients is below 
          the specified tolerance (`self.tol`) or when the scale of residuals 
          becomes too small.
        - The weighting function (`self.psi`) and scale estimator (`self.scale_estimator`) 
          are user-defined and determine the robustness of the fitting process.
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
            beta_new = np.linalg.pinv(X_design.T @ W @ X_design) @ X_design.T @ W @ y
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

        self._compute_standard_errors()

    def _compute_standard_errors(self) -> None:
        if self.X_design_ is None or self.residuals is None or self.weights is None:
            raise ValueError("Fit the model before computing standard errors.")
        self.std_errors_ = compute_sandwich_se(self.X_design_[:, 1:], self.residuals, self.weights)

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
        Return extended robust model summary statistics.

        Returns
        -------
        dict
            Dictionary with intercept, coefficients, inference, residual stats, and weights.
        """
        if self.coefficients is None or self.fitted_values is None:
            raise ValueError("Model must be fit before calling summary().")

        if self.X_design_ is None or self.residuals is None:
            raise ValueError("Model must be fit before calling summary().")

        n, p_plus1 = self.X_design_.shape
        p = p_plus1 - 1
        df_residual = n - p - 1

        # Residual standard error (manually computed here)
        rss = np.sum(self.residuals**2)
        rse = np.sqrt(rss / df_residual)

        # Compute t-stats, p-values, CIs
        if self.std_errors_ is None:
            raise ValueError("Standard errors must be computed before calling inference_summary.")

        # Compute the standard error for the intercept
        intercept_se = np.sqrt(
            np.sum((self.residuals**2) / (self.X_design_[:, 0]**2)) / (len(self.residuals) - len(self.coefficients) - 1)
        )

        # Concatenate the intercept's standard error with the predictors' standard errors
        std_errors_with_intercept = np.concatenate(([intercept_se], self.std_errors_))

        infer = inference_summary(
            coefficients=np.concatenate((np.array([self.intercept]), self.coefficients)),
            std_errors=std_errors_with_intercept,
            # Removed df_residual as it is not a valid parameter
            alpha=0.05
        )

        # ANOVA-like metrics (R², adjusted R², F-statistic)
        anova = anova_like_summary(
            y_true=self.fitted_values + self.residuals,
            y_pred=self.fitted_values,
            p=len(self.coefficients)
        )

        return {
            "intercept": self.intercept,
            "coefficients": self.coefficients,
            "residual_std_error": rse,
            "degrees_of_freedom": df_residual,
            "mean_weight": float(np.mean(self.weights)) if self.weights is not None else 0.0,
            "n_iter": self.max_iter,
            "weighted": True,
            "std_errors": self.std_errors_,
            "t_values": infer["t_values"],
            "p_values": infer["p_values"],
            "confidence_intervals": infer["confidence_intervals"],
            "r_squared": anova["r_squared"],
            "adj_r_squared": anova["adj_r_squared"],
            "f_statistic": anova["f_statistic"],
            "f_p_value": anova["f_p_value"]
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
