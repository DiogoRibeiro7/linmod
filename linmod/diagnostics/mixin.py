from typing import Any
import numpy as np
from linmod.diagnostics.normality import (
    shapiro_wilk_test,
    dagostino_k2,
    normality_heuristic,
)
from linmod.diagnostics.heteroscedasticity import (
    white_test,
    breusch_pagan_test,
    goldfeld_quandt_test,
    park_test,
    glejser_test,
    variance_power_test,
)
from linmod.diagnostics.functional_form import (
    reset_test,
    harvey_collier_test,
    white_nonlinearity_test,
)


class DiagnosticsMixin:
    def diagnostics(
        self,
        alpha: float = 0.05,
        gq_sort_by: int = 1,
        park_index: int = 0,
        glejser_index: int = 0,
        glejser_transform: str = "sqrt",
    ) -> dict[str, dict[str, float | str]]:
        if self.residuals is None or self.X_design_ is None or self.fitted_values is None:
            raise ValueError("Model must be fit before running diagnostics.")

        X_ = self.X_design_[:, 1:]
        residuals = self.residuals
        y_raw = self.fitted_values + self.residuals

        return {
            "breusch_pagan": breusch_pagan_test(X_, residuals, alpha=alpha),
            "white": white_test(X_, residuals, alpha=alpha),
            "shapiro_wilk": shapiro_wilk_test(residuals, alpha=alpha),
            "dagostino_k2": dagostino_k2(residuals.tolist(), alpha=alpha),
            "goldfeld_quandt": goldfeld_quandt_test(X_, y_raw, sort_by=gq_sort_by, alpha=alpha),
            "park": park_test(X_, residuals, predictor_index=park_index, alpha=alpha),
            "glejser": glejser_test(
                X_,
                residuals,
                predictor_index=glejser_index,
                transform=glejser_transform,
                alpha=alpha,
            ),
            "reset": reset_test(
                self.X_design_,
                y_raw,
                self.fitted_values,
                self.residuals,
                alpha=alpha,
            ),
            "variance_power": variance_power_test(
                X_, residuals=self.residuals, predictor_index=park_index, alpha=alpha
            ),
            "box_cox": self.box_cox_suggestion(),
            "harvey_collier": harvey_collier_test(
                self.fitted_values, self.residuals, alpha=alpha
            ),
            "white_nonlinearity": white_nonlinearity_test(
                self.X_design_, self.fitted_values, self.residuals, alpha=alpha
            ),
        }

    def normality_summary(self, alpha: float = 0.05) -> dict[str, dict[str, float | str]]:
        if self.residuals is None:
            raise ValueError("Model must be fit before checking normality.")

        return {
            "shapiro_wilk": shapiro_wilk_test(self.residuals, alpha=alpha),
            "dagostino_k2": dagostino_k2(self.residuals.tolist(), alpha=alpha),
            "heuristic": normality_heuristic(self.residuals.tolist(), alpha=alpha),
        }

    def print_normality_summary(self, alpha: float = 0.05) -> None:
        results = self.normality_summary(alpha=alpha)

        print("\nNormality Summary")
        print("=" * 20)
        for test, values in results.items():
            title = test.replace("_", " ").title()
            print(f"\n{title}")
            print("-" * len(title))
            for key, value in values.items():
                label = key.replace("_", " ").capitalize()
                if isinstance(value, float):
                    print(f"{label:<20}: {value:.4f}")
                else:
                    print(f"{label:<20}: {value}")

    def normality_summary_to_latex(self, alpha: float = 0.05) -> str:
        results = self.normality_summary(alpha=alpha)

        def format_row(test: str, metric: str, value: float | str) -> str:
            if isinstance(value, float):
                return f"{test} & {metric} & {value:.4f} \\\\"
            return f"{test} & {metric} & {value} \\\\"

        rows = []
        for test, metrics in results.items():
            test_name = test.replace("_", " ").title()
            for key, val in metrics.items():
                metric = key.replace("_", " ").capitalize()
                rows.append(format_row(test_name, metric, val))

        body = "\n".join(rows)
        table = (
            "\\begin{tabular}{lll}\n"
            "\\toprule\n"
            "Test & Metric & Value \\\\\n"
            "\\midrule\n"
            f"{body}\n"
            "\\bottomrule\n"
            "\\end{tabular}"
        )
        return table

    def print_diagnostics(self, alpha: float = 0.05) -> None:
        diag = self.diagnostics(alpha=alpha)

        def print_section(title: str, result: dict[str, float | str]) -> None:
            print(f"\n{title}")
            print("-" * len(title))
            for key, value in result.items():
                label = key.replace("_", " ").capitalize()
                if isinstance(value, float):
                    print(f"{label:<20}: {value:.4f}")
                else:
                    print(f"{label:<20}: {value}")

        print("\nDiagnostics Summary")
        print("=" * 20)
        print_section("Breusch–Pagan Test", diag["breusch_pagan"])
        print_section("White Test", diag["white"])
        print_section("D'Agostino K² Normality Test", diag["dagostino_k2"])
        print_section("Goldfeld–Quandt Test", diag["goldfeld_quandt"])
        print_section("Park Test", diag["park"])
        print_section("Glejser Test", diag["glejser"])

    def diagnostics_to_latex(self, alpha: float = 0.05) -> str:
        diag = self.diagnostics(alpha=alpha)

        def format_row(test: str, metric: str, value: float | str) -> str:
            if isinstance(value, float):
                return f"{test} & {metric} & {value:.4f} \\\\"
            return f"{test} & {metric} & {value} \\\\"

        rows = []
        for test, results in diag.items():
            test_name = test.replace("_", " ").title()
            for key, val in results.items():
                metric = key.replace("_", " ").capitalize()
                rows.append(format_row(test_name, metric, val))

        body = "\n".join(rows)
        return (
            "\\begin{tabular}{lll}\n"
            "\\toprule\n"
            "Test & Metric & Value \\\\\n"
            "\\midrule\n"
            f"{body}\n"
            "\\bottomrule\n"
            "\\end{tabular}"
        )
