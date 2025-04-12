import numpy as np
from scipy import stats
from typing import Any

from linmod.base import BaseLinearModel
from linmod.inference import LinearInferenceMixin
from linmod.stats.wls import WeightedLinearModel
from linmod.stats.gls import GeneralizedLinearModel
from linmod.regularization.ridge import RidgeLinearModel
from linmod.regularization.lasso import LassoLinearModel
from linmod.regularization.elasticnet import ElasticNetLinearModel
from linmod.evaluation.crossval import cross_val_score

from linmod.diagnostics.normality import (
    shapiro_wilk_test,
    dagostino_k2,
    normality_heuristic
)

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

from linmod.transforms.build import suggest_transformed_model


class LinearModel(BaseLinearModel, LinearInferenceMixin):
    def __init__(self):
        BaseLinearModel.__init__(self)
        LinearInferenceMixin.__init__(self)

        self.std_errors = None
        self.robust_std_errors = None
        self.t_values = None
        self.p_values = None
        self.confidence_intervals = None
        self.anova_table = None
        self.residual_std_error = None
        self.r_squared = None
        self.adj_r_squared = None
        self.f_statistic = None
        self.f_p_value = None
        self.df_residual = None
        self._transformation_steps = None  # Initialize transformation steps
        self._transformed_features = None  # Initialize transformed features

    def fit(self, X: np.ndarray, y: np.ndarray, alpha: float | None = None, robust: str | None = None) -> None:
        """
        Fit the linear regression model using Ordinary Least Squares.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (n_samples, n_features) excluding intercept.
        y : np.ndarray
            Response variable (n_samples,).
        alpha : float, optional
            Confidence level (default uses self._inference_alpha).
        robust : str, optional
            Robust SE type: 'HC0', ..., 'HC4' (default uses self._inference_robust).
        """
        beta, y_pred, residuals = self._fit_ols_via_pinv(X, y)

        alpha = alpha if alpha is not None else self._inference_alpha
        robust = robust if robust is not None else self._inference_robust

        inference = self.compute_inference(
            beta, self.X_design_, y, residuals,
            alpha=alpha,
            robust=robust
        )

        self.std_errors = inference["std_errors"]
        self.robust_std_errors = inference["robust_std_errors"]
        self.t_values = inference["t_values"]
        self.p_values = inference["p_values"]
        self.confidence_intervals = inference["confidence_intervals"]
        self.anova_table = inference["anova_table"]
        self.residual_std_error = inference["residual_std_error"]
        self.r_squared = inference["r_squared"]
        self.adj_r_squared = inference["adj_r_squared"]
        self.f_statistic = inference["f_statistic"]
        self.f_p_value = inference["f_p_value"]
        self.df_residual = inference["df_residual"]

    def summary(self) -> dict[str, Any]:
        """
        Return model summary statistics, including inference configuration.

        Returns
        -------
        dict[str, Any]
            A dictionary with coefficients, standard errors, statistics, and fit info.
        """
        if self.coefficients is None:
            raise ValueError("Model is not fit yet.")

        return {
            "intercept": self.intercept,
            "coefficients": self.coefficients,
            "std_errors": self.std_errors,
            "robust_std_errors": self.robust_std_errors,
            "t_values": self.t_values,
            "p_values": self.p_values,
            "confidence_intervals": self.confidence_intervals,
            "residual_std_error": self.residual_std_error,
            "df_residual": self.df_residual,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "f_statistic": self.f_statistic,
            "f_p_value": self.f_p_value,
            "anova_table": self.anova_table,
            "inference_config": {
                "alpha": self._inference_alpha,
                "robust_type": self._inference_robust
            }
        }

    def anova(self) -> dict[str, Any]:
        """
        Return ANOVA table from the fitted model.

        Returns
        -------
        dict[str, Any]
            Dictionary with degrees of freedom, sum of squares, mean squares, F and p-values.
        """
        if self.anova_table is None:
            raise ValueError("Model is not fit yet.")
        return self.anova_table

    def compute_robust_se(self, X: np.ndarray, residuals: np.ndarray, hc_type: str = "HC1") -> np.ndarray:
        """
        Compute heteroscedasticity-consistent robust standard errors.

        Parameters
        ----------
        X : np.ndarray
            Design matrix with intercept (n_samples, n_features + 1).
        residuals : np.ndarray
            Residual vector.
        hc_type : str
            Type of robust estimator: 'HC0', 'HC1', 'HC2', 'HC3', 'HC4'.

        Returns
        -------
        np.ndarray
            Vector of robust standard errors.
        """
        n, p = X.shape
        hat_matrix = X @ np.linalg.pinv(X.T @ X) @ X.T
        h = np.diag(hat_matrix)
        u = residuals

        if hc_type == "HC0":
            omega = u ** 2
        elif hc_type == "HC1":
            omega = u ** 2 * (n / (n - p))
        elif hc_type == "HC2":
            omega = u ** 2 / (1 - h)
        elif hc_type == "HC3":
            omega = u ** 2 / (1 - h) ** 2
        elif hc_type == "HC4":
            delta = np.minimum(4, h / np.mean(h))
            omega = u ** 2 / (1 - h) ** delta
        else:
            raise ValueError(f"Unknown hc_type: {hc_type}")

        S = np.diag(omega)
        XtX_inv = np.linalg.pinv(X.T @ X)
        cov_hc = XtX_inv @ X.T @ S @ X @ XtX_inv
        return np.sqrt(np.diag(cov_hc))

    def white_test(self, alpha: float = 0.05) -> dict[str, float | str]:
        """
        Perform White's test for heteroscedasticity.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05).

        Returns
        -------
        dict with 'LM', 'df', 'p-value', and 'interpretation'.
        """
        if self.residuals is None:
            raise ValueError("Model must be fit before calling white_test.")

        X_design = self.X_design_[:, 1:]  # exclude intercept
        n, k = X_design.shape
        u2 = self.residuals**2

        Z = [np.ones(n)]
        Z.extend(X_design.T)
        Z.extend(list(X_design[:, i] * X_design[:, j]
                      for i in range(k) for j in range(i, k)))
        Z = np.column_stack(Z)

        beta_aux = np.linalg.lstsq(Z, u2, rcond=None)[0]
        y_hat = Z @ beta_aux
        ssr = np.sum((y_hat - u2.mean())**2)
        lm = n * ssr / np.sum((u2 - u2.mean())**2)

        df = Z.shape[1] - 1
        p_value = 1 - stats.chi2.cdf(lm, df)

        interpretation = (
            "Evidence of heteroscedasticity (non-constant variance)"
            if p_value < alpha
            else "No evidence of heteroscedasticity"
        )

        return {"LM": lm, "df": df, "p-value": p_value, "interpretation": interpretation}

    def diagnostics(
        self,
        alpha: float = 0.05,
        gq_sort_by: int = 1,
        park_index: int = 0,
        glejser_index: int = 0,
        glejser_transform: str = "sqrt"
    ) -> dict[str, dict[str, float | str]]:
        """
        Run diagnostic tests for heteroscedasticity and residual normality.

        Parameters
        ----------
        alpha : float
            Significance level for hypothesis tests.

        Returns
        -------
        dict with results for Breusch–Pagan, White, and normality tests.
        """
        if self.residuals is None or self.X_design_ is None or self.fitted_values is None:
            raise ValueError("Model must be fit before running diagnostics.")

        # Remove intercept column
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
            "glejser": glejser_test(X_, residuals, predictor_index=glejser_index, transform=glejser_transform, alpha=alpha),
            "reset": reset_test(
                self.X_design_,
                self.fitted_values + self.residuals,
                self.fitted_values,
                self.residuals,
                alpha=alpha
            ),
            "variance_power": variance_power_test(X_, residuals=self.residuals, predictor_index=park_index, alpha=alpha),
            "box_cox": self.box_cox_suggestion(),
            "harvey_collier": harvey_collier_test(
                    self.fitted_values,
                    self.residuals,
                    alpha=alpha
                ),

            "white_nonlinearity": white_nonlinearity_test(
                    self.X_design_,
                    self.fitted_values,
                    self.residuals,
                    alpha=alpha
                )
        }

    def normality_summary(self, alpha: float = 0.05) -> dict[str, dict[str, float | str]]:
        """
        Summarize residual normality using multiple tests (Shapiro–Wilk, D'Agostino K², and heuristic).

        Parameters
        ----------
        alpha : float
            Significance level for interpretations.

        Returns
        -------
        dict
            Dictionary with results from Shapiro–Wilk, D'Agostino K², and normality heuristic.
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

        Parameters
        ----------
        alpha : float
            Significance level for interpretation.
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
                if isinstance(value, float):
                    print(f"{label:<20}: {value:.4f}")
                else:
                    print(f"{label:<20}: {value}")

    def normality_summary_to_latex(self, alpha: float = 0.05) -> str:
        """
        Export normality test results to a LaTeX tabular format.

        Parameters
        ----------
        alpha : float
            Significance level for interpretation.

        Returns
        -------
        str
            LaTeX tabular string.
        """
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
        """
        Print formatted diagnostics for heteroscedasticity and normality.

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
        """
        Export diagnostics summary as a LaTeX tabular environment.

        Parameters
        ----------
        alpha : float
            Significance level.

        Returns
        -------
        str
            LaTeX string representing the diagnostics table.
        """
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

    def variance_power_test(self, predictor_index: int = 0, alpha: float = 0.05) -> dict[str, float | str]:
        """
        Estimate power model: Var(u²) = x^gamma

        Parameters
        ----------
        predictor_index : int
            Index of predictor to test.
        alpha : float
            Significance level.

        Returns
        -------
        dict with gamma estimate, t-statistic, p-value, and interpretation.
        """
        if self.X_design_ is None or self.residuals is None:
            raise ValueError("Model must be fit before variance power test.")

        x = self.X_design_[:, predictor_index + 1]
        u2 = self.residuals**2

        x_pos = np.where(x > 0, x, 1e-8)
        log_u2 = np.log(u2 + 1e-8)
        log_x = np.log(x_pos)

        x_mean = np.mean(log_x)
        y_mean = np.mean(log_u2)

        slope = np.sum((log_x - x_mean) * (log_u2 - y_mean)) / \
            np.sum((log_x - x_mean)**2)
        intercept = y_mean - slope * x_mean

        y_pred = intercept + slope * log_x
        residuals = log_u2 - y_pred
        sse = np.sum(residuals ** 2)
        se_slope = np.sqrt(sse / (len(x) - 2) / np.sum((log_x - x_mean) ** 2))

        t_stat = slope / se_slope
        p = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(x) - 2))

        interpretation = (
            "Variance appears to follow a power function of the predictor"
            if p < alpha
            else "No strong power-form variance relationship"
        )

        return {
            "gamma (slope)": slope,
            "t statistic": t_stat,
            "p-value": p,
            "interpretation": interpretation
        }

    def box_cox_suggestion(self, lambdas: list[float] = [-1, -0.5, 0, 0.5, 1]) -> dict[str, float | str]:
        """
        Approximate Box–Cox transformation recommendation by minimizing residual variance.

        Parameters
        ----------
        lambdas : list of float
            Power transformation candidates (0 means log).

        Returns
        -------
        dict with best lambda, residual variance, and recommendation.
        """
        if self.X_design_ is None or self.fitted_values is None or self.residuals is None:
            raise ValueError("Model must be fit before Box–Cox suggestion.")

        y = self.fitted_values + self.residuals
        best_lambda = None
        best_rss = float("inf")

        for lmb in lambdas:
            if lmb == 0:
                y_trans = np.log(y + 1e-8)
            else:
                y_trans = (y ** lmb - 1) / lmb

            beta = np.linalg.lstsq(self.X_design_, y_trans, rcond=None)[0]
            y_hat = self.X_design_ @ beta
            rss = np.sum((y_trans - y_hat) ** 2)

            if rss < best_rss:
                best_rss = rss
                best_lambda = lmb

        interpretation = f"Suggest Box–Cox transform with lambda = {best_lambda}"

        return {
            "lambda": best_lambda,
            "RSS": best_rss,
            "interpretation": interpretation
        }

    def recommend_transformations(self, alpha: float = 0.05) -> dict[str, list[str]]:
        """
        Recommend response and/or predictor transformations based on diagnostic tests.

        Parameters
        ----------
        alpha : float
            Significance level used for interpretation.

        Returns
        -------
        dict with recommendations for 'response' and 'predictors'.
        """
        if self.residuals is None or self.X_design_ is None:
            raise ValueError(
                "Model must be fit before making transformation suggestions.")

        diag = self.diagnostics(alpha=alpha)
        response_recommendations = []
        predictor_recommendations = []

        # Box-Cox transformation for y
        boxcox = diag["box_cox"]
        lmb = boxcox["lambda"]
        if lmb < -0.5:
            response_recommendations.append("1/y (inverse)")
        elif -0.5 <= lmb < 0.3:
            response_recommendations.append("log(y)")
        elif 0.3 <= lmb < 0.8:
            response_recommendations.append("sqrt(y)")
        elif lmb > 1.2:
            response_recommendations.append("y^2")
        elif abs(lmb - 1.0) > 0.1:
            response_recommendations.append(f"Box–Cox: y^{lmb:.2f}")

        # RESET / Harvey–Collier / White-nonlinearity → suggest polynomials or interactions
        if diag["reset"]["p-value"] < alpha or diag["harvey_collier"]["p-value"] < alpha:
            predictor_recommendations.append(
                "Include polynomial terms (e.g., x², x³)")

        if diag["white_nonlinearity"]["p-value"] < alpha:
            predictor_recommendations.append(
                "Include interaction terms (e.g., x₁ * x₂)")

        # Park / Glejser / Variance power
        if diag["park"]["p-value"] < alpha:
            predictor_recommendations.append(
                "Apply log(x) or sqrt(x) to stabilize variance")

        if diag["glejser"]["p-value"] < alpha:
            predictor_recommendations.append(
                f"Try transformation: {diag['glejser']['transformation']}(x)")

        if diag["variance_power"]["p-value"] < alpha:
            gamma = diag["variance_power"]["gamma (slope)"]
            if gamma < -0.5:
                predictor_recommendations.append("Apply inverse(x)")
            elif gamma < 0.5:
                predictor_recommendations.append("Apply log(x) or sqrt(x)")
            elif gamma > 1.5:
                predictor_recommendations.append(
                    "Consider x² or polynomial variance modeling")

        return {
            "response": list(set(response_recommendations)),
            "predictors": list(set(predictor_recommendations))
        }

    def suggest_transformed_model(self, alpha: float = 0.05) -> dict[str, Any]:
        """
        Suggest and build a transformed model based on diagnostics.

        Parameters
        ----------
        alpha : float
            Significance threshold for transformation suggestions.

        Returns
        -------
        dict containing:
            - y_trans: transformed response vector
            - X_trans: transformed predictor matrix (with intercept)
            - description: list of applied transformations
        """
        diag = self.diagnostics(alpha=alpha)
        X = self.X_design_[:, 1:]  # predictors only
        y = self.fitted_values + self.residuals
        n, p = X.shape

        transforms = {"description": [], "X_trans": [], "y_trans": y.copy()}

        # Transform y
        boxcox = diag["box_cox"]
        lmb = boxcox["lambda"]
        if abs(lmb - 1.0) > 0.1:
            if lmb == 0:
                y_t = np.log(y + 1e-8)
                transforms["description"].append("log(y)")
            else:
                y_t = (y ** lmb - 1) / lmb
                transforms["description"].append(
                    f"Box–Cox transform on y (λ={lmb:.2f})")
            transforms["y_trans"] = y_t

        # Start with original predictors
        X_parts = [np.ones((n, 1))]  # intercept
        X_parts.append(X)
        desc = [f"x{i+1}" for i in range(p)]

        # Add polynomial terms
        if diag["reset"]["p-value"] < alpha or diag["harvey_collier"]["p-value"] < alpha:
            for i in range(p):
                X_parts.append(X[:, i] ** 2)
                desc.append(f"x{i+1}^2")
            transforms["description"].append("Added squared terms (x^2)")

        # Add interactions
        if diag["white_nonlinearity"]["p-value"] < alpha:
            for i in range(p):
                for j in range(i + 1, p):
                    X_parts.append(X[:, i] * X[:, j])
                    desc.append(f"x{i+1}*x{j+1}")
            transforms["description"].append(
                "Added interaction terms (x_i * x_j)")

        # Log, sqrt, inverse transforms on predictors
        if diag["park"]["p-value"] < alpha or diag["glejser"]["p-value"] < alpha:
            for i in range(p):
                xi = X[:, i]
                X_parts.append(np.log(np.abs(xi) + 1e-8))
                desc.append(f"log(x{i+1})")
                X_parts.append(np.sqrt(np.abs(xi)))
                desc.append(f"sqrt(x{i+1})")
                X_parts.append(1 / (np.abs(xi) + 1e-8))
                desc.append(f"1/x{i+1}")
            transforms["description"].append(
                "Log, sqrt, and inverse transforms of predictors")

        # Final design matrix
        X_new = np.column_stack(X_parts)
        transforms["X_trans"] = X_new
        transforms["features"] = desc

        return transforms

    def transformation_recommendations_to_latex(self, alpha: float = 0.05) -> str:
        """
        Generate LaTeX table with recommended transformations.

        Parameters
        ----------
        alpha : float
            Significance level.

        Returns
        -------
        str
            LaTeX tabular string.
        """
        recs = self.recommend_transformations(alpha=alpha)
        lines = []

        for r in recs["response"]:
            lines.append(f"Response & {r} \\\\")

        for p in recs["predictors"]:
            lines.append(f"Predictors & {p} \\\\")

        table = (
            "\\begin{tabular}{ll}\n"
            "\\toprule\n"
            "Target & Recommended Transformation \\\\\n"
            "\\midrule\n"
            + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}"
        )

        return table

    def fit_transformed(self, alpha: float = 0.05) -> "LinearModel":
        """
        Automatically transform response and predictors based on diagnostics and fit a new model.

        Parameters
        ----------
        alpha : float
            Significance level used for transformation decisions.

        Returns
        -------
        LinearModel
            A new model trained on transformed data.
        """
        if self.X_design_ is None or self.residuals is None or self.fitted_values is None:
            raise ValueError("Fit the model before calling fit_transformed().")

        diag = self.diagnostics(alpha=alpha)

        suggestion = suggest_transformed_model(
            self.X_design_,
            self.residuals,
            self.fitted_values,
            diag,
            alpha=alpha
        )

        X_trans = suggestion["X_trans"]
        y_trans = suggestion["y_trans"]

        model = LinearModel()
        model.fit(X_trans[:, 1:], y_trans)  # exclude intercept

        # Store extra info for interpretation
        model._transformation_steps = suggestion["description"]
        model._transformed_features = suggestion["features"]

        return model

    def fit_wls(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        model = WeightedLinearModel(weights=weights)
        model.fit(X, y)
        self._copy_from(model)

    def fit_gls(self, X: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> None:
        model = GeneralizedLinearModel(sigma=sigma)
        model.fit(X, y)
        self._copy_from(model)

    def fit_ridge(self, X: np.ndarray, y: np.ndarray, lambda_: float = 1.0) -> None:
        model = RidgeLinearModel(lambda_=lambda_)
        model.fit(X, y)
        self._copy_from(model)

    def fit_lasso(self, X: np.ndarray, y: np.ndarray, lambda_: float = 1.0, max_iter: int = 1000, tol: float = 1e-4) -> None:
        model = LassoLinearModel(lambda_=lambda_, max_iter=max_iter, tol=tol)
        model.fit(X, y)
        self._copy_from(model)

    def fit_elasticnet(self, X: np.ndarray, y: np.ndarray, lambda_: float = 1.0, alpha: float = 0.5, max_iter: int = 1000, tol: float = 1e-4) -> None:
        model = ElasticNetLinearModel(
            lambda_=lambda_, alpha=alpha, max_iter=max_iter, tol=tol)
        model.fit(X, y)
        self._copy_from(model)

    def fit_ridge_cv(self, X: np.ndarray, y: np.ndarray, lambda_grid: list, k: int = 5) -> None:
        results = cross_val_score(X, y, lambda lambda_: RidgeLinearModel(
            lambda_), {"lambda_": lambda_grid}, k)
        best_lambda = max(results.items(), key=lambda item: item[1])[0][0]
        self.fit_ridge(X, y, lambda_=best_lambda)
        self._best_lambda_ = best_lambda

    def fit_lasso_cv(self, X: np.ndarray, y: np.ndarray, lambda_grid: list, k: int = 5) -> None:
        results = cross_val_score(X, y, lambda lambda_: LassoLinearModel(
            lambda_), {"lambda_": lambda_grid}, k)
        best_lambda = max(results.items(), key=lambda item: item[1])[0][0]
        self.fit_lasso(X, y, lambda_=best_lambda)
        self._best_lambda_ = best_lambda

    def fit_elasticnet_cv(self, X: np.ndarray, y: np.ndarray, lambda_grid: list, alpha_grid: list, k: int = 5) -> None:
        results = cross_val_score(
            X, y,
            model_fn=lambda lambda_, alpha: ElasticNetLinearModel(
                lambda_, alpha),
            param_grid={"lambda_": lambda_grid, "alpha": alpha_grid},
            k=k
        )
        best = max(results.items(), key=lambda item: item[1])[0]
        self.fit_elasticnet(X, y, lambda_=best[0], alpha=best[1])
        self._best_lambda_ = best[0]
        self._best_alpha_ = best[1]

    def _copy_from(self, model: Any) -> None:
        """Copy core fitted attributes from another model."""
        self.coefficients = model.coefficients
        self.intercept = model.intercept
        self.fitted_values = model.fitted_values
        self.residuals = model.residuals
        self.X_design_ = model.X_design_


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = 4 + 2.5 * X[:, 0] - 1.7 * X[:, 1] + np.random.randn(100)

    model = LinearModel()
    model.fit(X, y)

    summary = model.summary()
    anova = model.anova()

    print("\nCoefficients with 95% CI and robust SE:")
    for i, name in enumerate(["(Intercept)"] + [f"x{i+1}" for i in range(X.shape[1])]):
        ci = summary['confidence_intervals'][i]
        print(f"{name:<12} {summary['coefficients'][i]:>8.4f}  "
              f"SE: {summary['std_errors'][i]:>8.4f}  "
              f"Robust SE: {summary['robust_std_errors'][i]:>8.4f}  "
              f"t: {summary['t_values'][i]:>8.2f}  "
              f"p: {summary['p_values'][i]:>8.4f}  "
              f"CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    print(f"\nResidual Std. Error: {summary['residual std error']:.4f} "
          f"on {summary['df.residual']} degrees of freedom")
    print(
        f"R-squared: {summary['r.squared']:.4f}, Adjusted R-squared: {summary['adj.r.squared']:.4f}")
    print(
        f"F-statistic: {summary['f.statistic']:.2f}, p-value: {summary['f.p.value']:.4g}")

    print("\nANOVA Table:")
    print(f"{'Source':<12} {'Df':>5} {'SS':>12} {'MS':>12} {'F':>8} {'Pr(>F)':>10}")
    for name, i in zip(["Model", "Residuals"], range(2)):
        print(f"{name:<12} {anova['df'][i]:>5} {anova['SS'][i]:>12.4f} "
              f"{anova['MS'][i]:>12.4f} {anova['F'][i]:>8.2f} {anova['p'][i]:>10.4g}")
