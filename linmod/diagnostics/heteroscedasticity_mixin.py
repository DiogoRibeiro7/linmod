import numpy as np
from typing import Any

from linmod.diagnostics.heteroscedasticity import (
    white_test,
    breusch_pagan_test,
    goldfeld_quandt_test,
    park_test,
    glejser_test,
    variance_power_test
)


class HeteroscedasticityMixin:
    def diagnostics_heteroscedasticity(
        self,
        alpha: float = 0.05,
        gq_sort_by: int = 1,
        park_index: int = 0,
        glejser_index: int = 0,
        glejser_transform: str = "sqrt"
    ) -> dict[str, dict[str, float | str]]:
        """
        Run heteroscedasticity-related tests.

        Returns
        -------
        dict
            Dictionary with results for White, Breusch–Pagan, and other tests.
        """
        if self.residuals is None or self.X_design_ is None or self.fitted_values is None:
            raise ValueError("Model must be fit before diagnostics.")

        X_ = self.X_design_[:, 1:]
        residuals = self.residuals
        y_raw = self.fitted_values + self.residuals

        return {
            "breusch_pagan": breusch_pagan_test(X_, residuals, alpha=alpha),
            "white": white_test(X_, residuals, alpha=alpha),
            "goldfeld_quandt": goldfeld_quandt_test(X_, y_raw, sort_by=gq_sort_by, alpha=alpha),
            "park": park_test(X_, residuals, predictor_index=park_index, alpha=alpha),
            "glejser": glejser_test(X_, residuals, predictor_index=glejser_index, transform=glejser_transform, alpha=alpha),
            "variance_power": variance_power_test(X_, residuals, predictor_index=park_index, alpha=alpha),
        }

    def print_heteroscedasticity_summary(self, alpha: float = 0.05) -> None:
        """
        Print results of heteroscedasticity diagnostics.
        """
        results = self.diagnostics_heteroscedasticity(alpha=alpha)

        print("\nHeteroscedasticity Summary")
        print("=" * 30)

        def print_section(title: str, result: dict[str, float | str]) -> None:
            print(f"\n{title}")
            print("-" * len(title))
            for key, value in result.items():
                label = key.replace('_', ' ').capitalize()
                print(f"{label:<20}: {value:.4f}" if isinstance(value, float) else f"{label:<20}: {value}")

        for test_name, result in results.items():
            title = test_name.replace("_", " ").title()
            print_section(title, result)

    def heteroscedasticity_summary_to_latex(self, alpha: float = 0.05) -> str:
        """
        Export heteroscedasticity diagnostics to LaTeX tabular format.
        """
        results = self.diagnostics_heteroscedasticity(alpha=alpha)

        def format_row(test: str, metric: str, value: float | str) -> str:
            return f"{test} & {metric} & {value:.4f} \\\\" if isinstance(value, float) else f"{test} & {metric} & {value} \\\\"

        rows = []
        for test, metrics in results.items():
            test_name = test.replace("_", " ").title()
            for key, val in metrics.items():
                metric = key.replace("_", " ").capitalize()
                rows.append(format_row(test_name, metric, val))

        return (
            "\\begin{tabular}{lll}\n"
            "\\toprule\n"
            "Test & Metric & Value \\\\\n"
            "\\midrule\n"
            + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}"
        )
