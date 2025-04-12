import math
from scipy.stats import shapiro
import numpy as np
from scipy import stats
from typing import Any, Union

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

    def diagnostics(self, alpha: float = 0.05, gq_sort_by: int = 1,
                park_index: int = 0,
                glejser_index: int = 0,
                glejser_transform: str = "sqrt") -> dict[str, dict[str, float | str]]:
        """
        Run diagnostic tests for heteroscedasticity and residual normality.

        Parameters
        ----------
        alpha : float
            Significance level for hypothesis tests.

        Returns
        -------
        dict with results for Breusch–Pagan, White, and Shapiro–Wilk tests.
        """
        if self.residuals is None or self.X_design_ is None:
            raise ValueError("Model must be fit before diagnostics.")

        bp = self.breusch_pagan_test(alpha=alpha)
        white = self.white_test(alpha=alpha)

        # Shapiro–Wilk Test for Normality of Residuals
        stat, p_value = shapiro(self.residuals)
        shapiro_result = {
            "W": stat,
            "p-value": p_value,
            "interpretation": (
                "Residuals are not normally distributed"
                if p_value < alpha
                else "Residuals appear normally distributed"
            )
        }
        k2 = dagostino_k2(self.residuals.tolist(), alpha=alpha)

        return {
            "breusch_pagan": bp,
            "white": white,
            "shapiro_wilk": shapiro_result,
            "dagostino_k2": k2,
            "goldfeld_quandt": self.goldfeld_quandt_test(sort_by=gq_sort_by, alpha=alpha),
            "park": self.park_test(predictor_index=park_index, alpha=alpha),
            "glejser": self.glejser_test(predictor_index=glejser_index, transform=glejser_transform, alpha=alpha),
            "reset": self.reset_test(alpha=alpha),
            "variance_power": self.variance_power_test(alpha=alpha),
            "box_cox": self.box_cox_suggestion(),
            "harvey_collier": self.harvey_collier_test(alpha=alpha),
            "white_nonlinearity": self.white_nonlinearity_test(alpha=alpha)
        }

    def goldfeld_quandt_test(self, sort_by: int = 1, drop_fraction: float = 0.2, alpha: float = 0.05) -> dict[str, Union[float, np.float64, np.ndarray, str]]:
        """
        Goldfeld–Quandt test for heteroscedasticity.

        Parameters
        ----------
        sort_by : int
            Index of predictor to sort observations (default: 1st column).
        drop_fraction : float
            Fraction of middle observations to drop when splitting data.
        alpha : float
            Significance level.

        Returns
        -------
        dict with F statistic, p-value (approximate), and interpretation.
        """
        import math

        if self.X_design_ is None or self.residuals is None:
            raise ValueError(
                "Model must be fit before running Goldfeld–Quandt test.")

        X = self.X_design_[:, 1:]  # remove intercept
        y = self.fitted_values + self.residuals
        n = len(y)

        sorted_idx = np.argsort(X[:, sort_by])
        X_sorted = X[sorted_idx]
        y_sorted = y[sorted_idx]

        drop = int(drop_fraction * n)
        split = (n - drop) // 2

        X1, y1 = X_sorted[:split], y_sorted[:split]
        X2, y2 = X_sorted[-split:], y_sorted[-split:]

        def rss(X_part, y_part):
            Xb = np.hstack([np.ones((len(X_part), 1)), X_part])
            beta = np.linalg.lstsq(Xb, y_part, rcond=None)[0]
            residuals = y_part - Xb @ beta
            return np.sum(residuals**2)

        rss1 = rss(X1, y1)
        rss2 = rss(X2, y2)

        f_stat = max(rss1, rss2) / min(rss1, rss2)
        df1 = df2 = split - X.shape[1] - 1
        p = 1 - stats.f.cdf(f_stat, df1,
                            df2) if df1 > 0 and df2 > 0 else float("nan")

        interpretation = "Evidence of heteroscedasticity" if p < alpha else "No evidence of heteroscedasticity"

        return {
            "F statistic": f_stat,
            "df1": df1,
            "df2": df2,
            "p-value": p,
            "interpretation": interpretation
        }

    def breusch_pagan_test(self, alpha: float = 0.05) -> dict[str, float | str]:
        """
        Perform Breusch–Pagan test for heteroscedasticity.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05).

        Returns
        -------
        dict with 'LM', 'df', 'p-value', and 'interpretation'.
        """
        if self.residuals is None:
            raise ValueError(
                "Model must be fit before calling breusch_pagan_test.")

        n, k = self.X_design_.shape
        u2 = self.residuals**2

        beta_aux = np.linalg.lstsq(self.X_design_, u2, rcond=None)[0]
        y_hat = self.X_design_ @ beta_aux
        ssr = np.sum((y_hat - u2.mean())**2)

        lm = 0.5 * n * ssr / np.var(u2, ddof=1)
        df = k - 1
        p_value = 1 - stats.chi2.cdf(lm, df)

        interpretation = (
            "Evidence of heteroscedasticity (non-constant variance)"
            if p_value < alpha
            else "No evidence of heteroscedasticity"
        )

        return {"LM": lm, "df": df, "p-value": p_value, "interpretation": interpretation}

    def park_test(self, predictor_index: int = 0, alpha: float = 0.05) -> dict[str, float | str]:
        """
        Park test for heteroscedasticity.

        Parameters
        ----------
        predictor_index : int
            Index of the predictor to test.
        alpha : float
            Significance level.

        Returns
        -------
        dict with slope, t-statistic, and interpretation.
        """
        import math

        if self.X_design_ is None or self.residuals is None:
            raise ValueError("Model must be fit before Park test.")

        u2 = self.residuals ** 2
        log_u2 = np.log(u2 + 1e-8)  # prevent log(0)
        x = self.X_design_[:, predictor_index + 1]  # +1 due to intercept

        x_mean = np.mean(x)
        log_u2_mean = np.mean(log_u2)

        cov = np.sum((x - x_mean) * (log_u2 - log_u2_mean))
        var_x = np.sum((x - x_mean) ** 2)

        slope = cov / var_x
        intercept = log_u2_mean - slope * x_mean

        residuals = log_u2 - (intercept + slope * x)
        sse = np.sum(residuals**2)
        se_slope = np.sqrt(sse / (len(x) - 2) / var_x)

        t_stat = slope / se_slope
        p = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(x) - 2))

        interpretation = "Evidence of heteroscedasticity" if p < alpha else "No evidence of heteroscedasticity"

        return {
            "slope": float(slope),
            "t statistic": float(t_stat),
            "p-value": float(p),
            "interpretation": str(interpretation)
        }

    def glejser_test(self, predictor_index: int = 0, transform: str = "sqrt", alpha: float = 0.05) -> dict[str, float | str]:
        """
        Glejser test for heteroscedasticity.

        Parameters
        ----------
        predictor_index : int
            Index of the predictor to test.
        transform : str
            Transformation: 'raw', 'sqrt', or 'inverse'.
        alpha : float
            Significance level.

        Returns
        -------
        dict with slope, t-statistic, and interpretation.
        """
        import math

        if self.X_design_ is None or self.residuals is None:
            raise ValueError("Model must be fit before Glejser test.")

        y = np.abs(self.residuals)
        x = self.X_design_[:, predictor_index + 1]

        if transform == "raw":
            x_t = x
        elif transform == "sqrt":
            x_t = np.sqrt(np.abs(x))
        elif transform == "inverse":
            x_t = 1 / (np.abs(x) + 1e-8)
        else:
            raise ValueError(
                "Unknown transform: choose 'raw', 'sqrt', or 'inverse'.")

        x_mean = np.mean(x_t)
        y_mean = np.mean(y)

        cov = np.sum((x_t - x_mean) * (y - y_mean))
        var_x = np.sum((x_t - x_mean) ** 2)

        slope = cov / var_x
        intercept = y_mean - slope * x_mean

        residuals = y - (intercept + slope * x_t)
        sse = np.sum(residuals ** 2)
        se_slope = np.sqrt(sse / (len(x_t) - 2) / var_x)

        t_stat = slope / se_slope
        p = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(x_t) - 2))

        interpretation = "Evidence of heteroscedasticity" if p < alpha else "No evidence of heteroscedasticity"

        return {
            "slope": slope,
            "t statistic": t_stat,
            "p-value": p,
            "transformation": transform,
            "interpretation": interpretation
        }
    
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
    
    def reset_test(self, max_power: int = 3, alpha: float = 0.05) -> dict[str, float | str]:
        """
        RESET test for functional form misspecification.

        Parameters
        ----------
        max_power : int
            Highest power of fitted values to include (e.g., 2 means include ŷ²).
        alpha : float
            Significance level.

        Returns
        -------
        dict with F statistic, df, p-value, and interpretation.
        """
        if self.fitted_values is None or self.X_design_ is None or self.residuals is None:
            raise ValueError("Model must be fit before running RESET test.")

        n = len(self.fitted_values)
        base_X = self.X_design_
        powers = [self.fitted_values ** p for p in range(2, max_power + 1)]
        augmented_X = np.column_stack([base_X] + powers)

        beta_aug = np.linalg.lstsq(augmented_X, self.fitted_values + self.residuals, rcond=None)[0]
        y_aug_hat = augmented_X @ beta_aug
        residuals_aug = self.fitted_values + self.residuals - y_aug_hat

        rss_base = np.sum(self.residuals ** 2)
        rss_aug = np.sum(residuals_aug ** 2)

        df1 = max_power - 1
        df2 = n - augmented_X.shape[1]
        f_stat = ((rss_base - rss_aug) / df1) / (rss_aug / df2)
        p = 1 - stats.f.cdf(f_stat, df1, df2)

        interpretation = (
            "Possible functional form misspecification"
            if p < alpha
            else "No strong evidence of misspecification"
        )

        return {
            "F statistic": f_stat,
            "df1": df1,
            "df2": df2,
            "p-value": p,
            "interpretation": interpretation
        }


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

        slope = np.sum((log_x - x_mean) * (log_u2 - y_mean)) / np.sum((log_x - x_mean)**2)
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
        
    def harvey_collier_test(self, alpha: float = 0.05) -> dict[str, float | str]:
        """
        Harvey–Collier test for linear functional form misspecification.

        Parameters
        ----------
        alpha : float
            Significance level.

        Returns
        -------
        dict with t statistic, p-value, and interpretation.
        """
        if self.fitted_values is None or self.residuals is None:
            raise ValueError("Model must be fit before Harvey–Collier test.")

        y = self.fitted_values + self.residuals
        y_hat = self.fitted_values

        Z = np.column_stack([np.ones(len(y_hat)), y_hat])
        beta = np.linalg.lstsq(Z, y, rcond=None)[0]
        y_pred = Z @ beta
        residuals = y - y_pred

        sse = np.sum(residuals ** 2)
        mse = sse / (len(y) - 2)
        var_beta = mse * np.linalg.inv(Z.T @ Z)
        t_stat = beta[1] / np.sqrt(var_beta[1, 1])

        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(y) - 2))

        interpretation = (
            "Possible linear functional form misspecification"
            if p_value < alpha
            else "No evidence of linear form misfit"
        )

        return {
            "t statistic": t_stat,
            "p-value": p_value,
            "interpretation": interpretation
        }
        
    def white_nonlinearity_test(self, alpha: float = 0.05) -> dict[str, float | str]:
        """
        White’s expansion test for nonlinear functional form misspecification.

        Parameters
        ----------
        alpha : float
            Significance level.

        Returns
        -------
        dict with F statistic, df, p-value, and interpretation.
        """
        if self.X_design_ is None or self.fitted_values is None or self.residuals is None:
            raise ValueError("Model must be fit before test.")

        X = self.X_design_[:, 1:]  # exclude intercept
        y = self.fitted_values + self.residuals
        n, k = X.shape

        nonlinear_terms = []
        for i in range(k):
            nonlinear_terms.append(X[:, i] ** 2)
            for j in range(i + 1, k):
                nonlinear_terms.append(X[:, i] * X[:, j])

        X_aug = np.column_stack([self.X_design_] + nonlinear_terms)
        df1 = X_aug.shape[1] - self.X_design_.shape[1]
        df2 = n - X_aug.shape[1]

        beta_full = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        rss_full = np.sum((y - X_aug @ beta_full) ** 2)
        rss_base = np.sum(self.residuals ** 2)

        f_stat = ((rss_base - rss_full) / df1) / (rss_full / df2)
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)

        interpretation = (
            "Model may be missing nonlinear structure"
            if p_value < alpha
            else "No evidence of nonlinear functional form"
        )

        return {
            "F statistic": f_stat,
            "df1": df1,
            "df2": df2,
            "p-value": p_value,
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
            raise ValueError("Model must be fit before making transformation suggestions.")

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
            predictor_recommendations.append("Include polynomial terms (e.g., x², x³)")

        if diag["white_nonlinearity"]["p-value"] < alpha:
            predictor_recommendations.append("Include interaction terms (e.g., x₁ * x₂)")

        # Park / Glejser / Variance power
        if diag["park"]["p-value"] < alpha:
            predictor_recommendations.append("Apply log(x) or sqrt(x) to stabilize variance")

        if diag["glejser"]["p-value"] < alpha:
            predictor_recommendations.append(f"Try transformation: {diag['glejser']['transformation']}(x)")

        if diag["variance_power"]["p-value"] < alpha:
            gamma = diag["variance_power"]["gamma (slope)"]
            if gamma < -0.5:
                predictor_recommendations.append("Apply inverse(x)")
            elif gamma < 0.5:
                predictor_recommendations.append("Apply log(x) or sqrt(x)")
            elif gamma > 1.5:
                predictor_recommendations.append("Consider x² or polynomial variance modeling")

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
                transforms["description"].append(f"Box–Cox transform on y (λ={lmb:.2f})")
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
            transforms["description"].append("Added interaction terms (x_i * x_j)")

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
            transforms["description"].append("Log, sqrt, and inverse transforms of predictors")

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


# addapt to the class laetr on
# TODO:

def normality_heuristic(x: list[float], alpha: float = 0.05) -> dict[str, float | str]:
    """
    Approximate normality test using skewness and kurtosis (base Python only).

    Parameters
    ----------
    x : list[float]
        Sample data.
    alpha : float
        Significance threshold.

    Returns
    -------
    dict with skewness, kurtosis, and interpretation.
    """
    n = len(x)
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / n
    std = var ** 0.5

    skewness = sum((xi - mean) ** 3 for xi in x) / (n * std**3)
    kurtosis = sum((xi - mean) ** 4 for xi in x) / \
        (n * std**4) - 3  # excess kurtosis

    interpretation = (
        "Possible non-normality (|skew| > 1 or kurtosis > 1)"
        if abs(skewness) > 1 or abs(kurtosis) > 1
        else "Residuals appear approximately normal"
    )

    return {
        "skewness": skewness,
        "excess kurtosis": kurtosis,
        "interpretation": interpretation
    }


def dagostino_k2(x: list[float], alpha: float = 0.05) -> dict[str, float | str]:
    n = len(x)
    if n < 8:
        raise ValueError(
            "D'Agostino's K² test requires at least 8 observations.")

    mean = sum(x) / n
    m2 = sum((xi - mean) ** 2 for xi in x) / n
    m3 = sum((xi - mean) ** 3 for xi in x) / n
    m4 = sum((xi - mean) ** 4 for xi in x) / n

    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2) - 3

    Y = skew * math.sqrt((n + 1) * (n + 3) / (6.0 * (n - 2)))
    beta2 = 3.0 * (n ** 2 + 27 * n - 70) * (n + 1) * (n + 3) / \
        ((n - 2) * (n + 5) * (n + 7) * (n + 9))
    W2 = -1 + math.sqrt(2 * (beta2 - 1))
    delta = 1 / math.sqrt(math.log(math.sqrt(W2)))
    alpha_skew = math.sqrt(2 / (W2 - 1))
    Z1 = delta * math.log(Y / alpha_skew +
                          math.sqrt((Y / alpha_skew) ** 2 + 1))

    E = 3.0 * (n - 1) / (n + 1)
    Var = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))
    X = (kurt - E) / math.sqrt(Var)
    B = 6.0 * (n ** 2 - 5 * n + 2) / ((n + 7) * (n + 9)) * \
        math.sqrt(6.0 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
    A = 6.0 + 8.0 / B * (2.0 / B + math.sqrt(1 + 4.0 / (B ** 2)))
    Z2 = ((1 - 2 / (9 * A)) - ((1 - 2 / A) / (1 + X * math.sqrt(2 / (A - 4))))
          ** (1 / 3)) / math.sqrt(2 / (9 * A))

    K2 = Z1 ** 2 + Z2 ** 2
    p = math.exp(-0.5 * K2)  # approx for chi2.sf(K2, df=2)

    interpretation = (
        "Residuals are not normally distributed"
        if p < alpha
        else "Residuals appear normally distributed"
    )

    return {
        "K² statistic": K2,
        "p-value (approx)": p,
        "interpretation": interpretation,
        "skewness": skew,
        "kurtosis": kurt
    }
