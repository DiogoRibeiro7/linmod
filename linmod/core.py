import numpy as np
from scipy import stats
from typing import Any


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
        self.adj_r_squared = 1 - (1 - self.r_squared) * (n_samples - 1) / df_residual
        self.f_statistic = f_stat
        self.f_p_value = f_p_val

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

    def white_test(self) -> dict[str, float]:
        """
        Perform White's test for heteroscedasticity.

        Returns
        -------
        dict[str, float]
            Dictionary with test statistic, degrees of freedom, and p-value.
        """
        if self.residuals is None or self.X_design_ is None:
            raise ValueError("Model must be fit before calling white_test.")

        X = self.X_design_[:, 1:]  # remove intercept
        n, k = X.shape
        u2 = self.residuals ** 2

        Z = [np.ones(n)]
        Z.extend(X.T)
        Z.extend((X[:, i] * X[:, j] for i in range(k) for j in range(i, k)))
        Z = np.column_stack(Z)

        beta_aux = np.linalg.lstsq(Z, u2, rcond=None)[0]
        y_hat = Z @ beta_aux
        ssr = np.sum((y_hat - u2.mean()) ** 2)
        lm = n * ssr / np.sum((u2 - u2.mean()) ** 2)

        df = Z.shape[1] - 1
        p_value = 1 - stats.chi2.cdf(lm, df)
        return {"LM": lm, "df": df, "p-value": p_value}



def breusch_pagan_test(self) -> dict[str, float]:
    """
    Perform the Breusch–Pagan test for heteroscedasticity.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - 'LM': Lagrange Multiplier test statistic
        - 'df': Degrees of freedom
        - 'p-value': p-value from the chi-squared distribution
    """
    if self.residuals is None or self.X_design_ is None:
        raise ValueError("Model must be fit before calling breusch_pagan_test.")

    n, p = self.X_design_.shape
    u2 = self.residuals ** 2

    # Auxiliary regression: regress u² on all regressors excluding intercept
    X_aux = self.X_design_[:, 1:]  # drop intercept
    beta_aux = np.linalg.lstsq(X_aux, u2, rcond=None)[0]
    y_hat = X_aux @ beta_aux

    # Explained sum of squares
    ssr = np.sum((y_hat - u2.mean()) ** 2)

    # LM statistic
    lm = n * ssr / np.sum((u2 - u2.mean()) ** 2)

    df = X_aux.shape[1]  # number of explanatory variables in auxiliary regression
    p_value = 1 - stats.chi2.cdf(lm, df)

    return {"LM": lm, "df": df, "p-value": p_value}



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
