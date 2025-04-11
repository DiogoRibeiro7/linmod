import math
from scipy.stats import shapiro
import numpy as np
from scipy import stats
from typing import Any

from linmod.stats.wls import WeightedLinearModel
from linmod.stats.gls import GeneralizedLinearModel
from linmod.regularization.ridge import RidgeLinearModel
from linmod.regularization.lasso import LassoLinearModel
from linmod.regularization.elasticnet import ElasticNetLinearModel
from linmod.evaluation.crossval import cross_val_score


class LinearModel:
    def __init__(self) -> None:
        self.coefficients: np.ndarray | None = None
        self.std_errors: np.ndarray | None = None
        self.robust_std_errors: np.ndarray | None = None
        self.t_values: np.ndarray | None = None
        self.p_values: np.ndarray | None = None
        self.confidence_intervals: np.ndarray | None = None
        self.anova_table: dict[str, Any] | None = None
        self.fitted_values: np.ndarray | None = None
        self.residuals: np.ndarray | None = None
        self.df_residual: int | None = None
        self.residual_std_error: float | None = None
        self.r_squared: float | None = None
        self.adj_r_squared: float | None = None
        self.f_statistic: float | None = None
        self.f_p_value: float | None = None
        self.X_design_: np.ndarray | None = None  # for internal diagnostics

    def fit(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> None:
        """
        Fit the linear regression model using Ordinary Least Squares.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (n_samples, n_features) excluding intercept.
        y : np.ndarray
            Response variable (n_samples,).
        alpha : float
            Confidence level for intervals (default=0.05).
        """
        n_samples, n_features = X.shape
        self.X_design_ = np.hstack([np.ones((n_samples, 1)), X])
        p = self.X_design_.shape[1]

        XtX = self.X_design_.T @ self.X_design_
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ self.X_design_.T @ y
        y_pred = self.X_design_ @ beta
        residuals = y - y_pred

        rss = np.sum(residuals**2)
        tss = np.sum((y - y.mean())**2)
        df_residual = n_samples - p
        df_model = p - 1

        mse = rss / df_residual
        se_beta = np.sqrt(np.diag(mse * XtX_inv))

        t_stats = beta / se_beta
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_residual))

        t_crit = stats.t.ppf(1 - alpha / 2, df=df_residual)
        ci_lower = beta - t_crit * se_beta
        ci_upper = beta + t_crit * se_beta
        conf_ints = np.column_stack([ci_lower, ci_upper])

        ms_model = (tss - rss) / df_model
        f_stat = ms_model / mse
        f_p_val = 1 - stats.f.cdf(f_stat, df_model, df_residual)

        # Robust standard errors (HC0 by default here)
        u = residuals.reshape(-1, 1)
        S = np.diagflat(u**2)
        sandwich = XtX_inv @ self.X_design_.T @ S @ self.X_design_ @ XtX_inv
        robust_se = np.sqrt(np.diag(sandwich))

        ssr = tss - rss
        msr = ssr / df_model

        self.anova_table = {
            "df": np.array([df_model, df_residual]),
            "SS": np.array([ssr, rss]),
            "MS": np.array([msr, mse]),
            "F": np.array([f_stat, np.nan]),
            "p": np.array([f_p_val, np.nan])
        }

        self.coefficients = beta
        self.std_errors = se_beta
        self.robust_std_errors = robust_se
        self.t_values = t_stats
        self.p_values = p_vals
        self.confidence_intervals = conf_ints
        self.fitted_values = y_pred
        self.residuals = residuals
        self.df_residual = df_residual
        self.residual_std_error = np.sqrt(mse)
        self.r_squared = 1 - rss / tss
        self.adj_r_squared = 1 - (1 - self.r_squared) * \
            (n_samples - 1) / df_residual
        self.f_statistic = f_stat
        self.f_p_value = f_p_val

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.intercept + X @ self.coefficients

    def summary(self) -> dict[str, Any]:
        """
        Return model summary statistics.

        Returns
        -------
        dict[str, Any]
            A dictionary with coefficients, standard errors, statistics, and fit info.
        """
        if self.coefficients is None:
            raise ValueError("Model is not fit yet.")

        return {
            "coefficients": self.coefficients,
            "std_errors": self.std_errors,
            "robust_std_errors": self.robust_std_errors,
            "t_values": self.t_values,
            "p_values": self.p_values,
            "confidence_intervals": self.confidence_intervals,
            "fitted.values": self.fitted_values,
            "residuals": self.residuals,
            "df.residual": self.df_residual,
            "residual std error": self.residual_std_error,
            "r.squared": self.r_squared,
            "adj.r.squared": self.adj_r_squared,
            "f.statistic": self.f_statistic,
            "f.p.value": self.f_p_value
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
        Z.extend((X_design[:, i] * X_design[:, j]
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

    def diagnostics(self, alpha: float = 0.05) -> dict[str, dict[str, float | str]]:
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
            "goldfeld_quandt": self.goldfeld_quandt_test(alpha=alpha),
            "park": self.park_test(alpha=alpha),
            "glejser": self.glejser_test(alpha=alpha)
        }

    def goldfeld_quandt_test(self, sort_by: int = 1, drop_fraction: float = 0.2, alpha: float = 0.05) -> dict[str, float | str]:
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
