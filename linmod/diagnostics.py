# linmod/diagnostics/mixin.py

from typing import Any, Union
import numpy as np
from linmod.diagnostics.heteroscedasticity import (
    white_test,
    breusch_pagan_test,
    goldfeld_quandt_test,
    park_test,
    glejser_test,
    variance_power_test
)
from linmod.diagnostics.functional_form import (
    reset_test,
    harvey_collier_test,
    white_nonlinearity_test
)

class DiagnosticsMixin:
    def diagnostics(
        self,
        alpha: float = 0.05,
        gq_sort_by: int = 1,
        park_index: int = 0,
        glejser_index: int = 0,
        glejser_transform: str = "sqrt"
    ) -> dict[str, dict[str, float | str]]:
        """
        Run diagnostic tests for heteroscedasticity and residual structure.

        Returns
        -------
        dict[str, dict[str, float | str]]
            A dictionary of test results.
        """
        if self.residuals is None or self.X_design_ is None or self.fitted_values is None:
            raise ValueError("Model must be fit before running diagnostics.")

        X_ = self.X_design_[:, 1:]  # exclude intercept
        residuals = self.residuals
        y_raw = self.fitted_values + self.residuals

        return {
            "breusch_pagan": breusch_pagan_test(X_, residuals, alpha=alpha),
            "white": white_test(X_, residuals, alpha=alpha),
            "goldfeld_quandt": goldfeld_quandt_test(X_, y_raw, sort_by=gq_sort_by, alpha=alpha),
            "park": park_test(X_, residuals, predictor_index=park_index, alpha=alpha),
            "glejser": glejser_test(X_, residuals, predictor_index=glejser_index, transform=glejser_transform, alpha=alpha),
            "reset": reset_test(self.X_design_, y_raw, self.fitted_values, self.residuals, alpha=alpha),
            "variance_power": variance_power_test(X_, residuals=self.residuals, predictor_index=park_index, alpha=alpha),
            "box_cox": self.box_cox_suggestion(),
            "harvey_collier": harvey_collier_test(self.fitted_values, self.residuals, alpha=alpha),
            "white_nonlinearity": white_nonlinearity_test(self.X_design_, self.fitted_values, self.residuals, alpha=alpha)
        }

    def print_diagnostics(self, alpha: float = 0.05) -> None:
        """
        Print formatted diagnostics for heteroscedasticity and functional form.

        Parameters
        ----------
        alpha : float
            Significance level for interpretations.
        """
        diag = self.diagnostics(alpha=alpha)

        def print_section(title: str, result: dict[str, float | str]) -> None:
            print(f"\n{title}")
            print("-" * len(title))
            for key, value in result.items():
                label = key.replace('_', ' ').capitalize()
                print(f"{label:<20}: {value:.4f}" if isinstance(value, float) else f"{label:<20}: {value}")

        print("\nDiagnostics Summary")
        print("=" * 20)
        print_section("Breusch–Pagan Test", diag["breusch_pagan"])
        print_section("White Test", diag["white"])
        print_section("D'Agostino K² Normality Test", diag.get("dagostino_k2", {}))
        print_section("Goldfeld–Quandt Test", diag["goldfeld_quandt"])
        print_section("Park Test", diag["park"])
        print_section("Glejser Test", diag["glejser"])

    def diagnostics_to_latex(self, alpha: float = 0.05) -> str:
        """
        Export diagnostics summary as a LaTeX tabular environment.

        Returns
        -------
        str
            LaTeX string.
        """
        diag = self.diagnostics(alpha=alpha)

        def format_row(test: str, metric: str, value: float | str) -> str:
            return f"{test} & {metric} & {value:.4f} \\\\" if isinstance(value, float) else f"{test} & {metric} & {value} \\\\"

        rows = []
        for test, results in diag.items():
            test_name = test.replace("_", " ").title()
            for key, val in results.items():
                metric = key.replace("_", " ").capitalize()
                rows.append(format_row(test_name, metric, val))

        return (
            "\\begin{tabular}{lll}\n"
            "\\toprule\n"
            "Test & Metric & Value \\\\\n"
            "\\midrule\n"
            + "\n".join(rows) +
            "\n\\bottomrule\n"
            "\\end{tabular}"
        )


def diagnostics_report(model) -> str:
    """
    Generate a plain text diagnostics report of the linear model.
    """
    lines = []

    lines.append("=== Linear Model Diagnostics ===")
    lines.append(f"Residual Std. Error: {model.residual_std_error:.4f}")
    lines.append(f"R-squared: {model.r_squared:.4f}")
    lines.append(f"Adjusted R-squared: {model.adj_r_squared:.4f}")
    lines.append(f"F-statistic: {model.f_statistic:.4f}, p-value: {model.f_p_value:.4g}")
    lines.append("")

    lines.append("--- Inference Configuration ---")
    lines.append(f"Confidence level: {1 - model._inference_alpha:.2%}")
    lines.append(f"Robust SE type: {model._inference_robust}")
    lines.append("")

    return "\n".join(lines)
