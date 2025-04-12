import numpy as np
from typing import Any
from linmod.diagnostics.normality import (
    shapiro_wilk_test,
    dagostino_k2,
    normality_heuristic
)


class NormalityMixin:
    def normality_summary(self, alpha: float = 0.05) -> dict[str, dict[str, float | str]]:
        """
        Summarize residual normality using multiple tests (Shapiro–Wilk, D'Agostino K², and heuristic).
        """
        if self.residuals is None:
            raise ValueError("Model must be fit before checking normality.")

        return {
            "shapiro_wilk": shapiro_wilk_test(self.residuals, alpha=alpha),
            "dagostino_k2": dagostino_k2(self.residuals.tolist(), alpha=alpha),
            "heuristic": normality_heuristic(self.residuals.tolist(), alpha=alpha)
        }

    def print_normality_summary(self, alpha: float = 0.05) -> None:
        """
        Print formatted normality test results.
        """
        results = self.normality_summary(alpha=alpha)

        print("\nNormality Summary")
        print("=" * 20)
        for test, values in results.items():
            title = test.replace("_", " ").title()
            print(f"\n{title}")
            print("-" * len(title))
            for key, value in values.items():
                label = key.replace('_', ' ').capitalize()
                print(f"{label:<20}: {value:.4f}" if isinstance(value, float) else f"{label:<20}: {value}")

    def normality_summary_to_latex(self, alpha: float = 0.05) -> str:
        """
        Export normality test results to a LaTeX tabular format.
        """
        results = self.normality_summary(alpha=alpha)

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
