import numpy as np
from typing import Any

from linmod.base import BaseLinearModel
from linmod.inference import LinearInferenceMixin
from linmod.stats.wls import WeightedLinearModel
from linmod.stats.gls import GeneralizedLinearModel
from linmod.regularization.ridge import RidgeLinearModel
from linmod.regularization.lasso import LassoLinearModel
from linmod.regularization.elasticnet import ElasticNetLinearModel
from linmod.diagnostics.mixin import DiagnosticsMixin
from linmod.diagnostics.normality_mixin import NormalityMixin
from linmod.transforms.mixin import TransformMixin
from linmod.evaluation.crossval import cross_val_score


class LinearModel(BaseLinearModel,
                  LinearInferenceMixin,
                  DiagnosticsMixin,
                  NormalityMixin,
                  TransformMixin):
    def __init__(self):
        BaseLinearModel.__init__(self)
        LinearInferenceMixin.__init__(self)
        DiagnosticsMixin.__init__(self)
        NormalityMixin.__init__(self)
        TransformMixin.__init__(self)

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

    def fit(self, X: np.ndarray, y: np.ndarray, alpha: float | None = None, robust: str | None = None) -> None:
        beta, y_pred, residuals = self._fit_ols_via_pinv(X, y)
        alpha = alpha if alpha is not None else self._inference_alpha
        robust = robust if robust is not None else self._inference_robust

        inference = self.compute_inference(beta, self.X_design_, y, residuals, alpha=alpha, robust=robust)

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
        if self.anova_table is None:
            raise ValueError("Model is not fit yet.")
        return self.anova_table

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
        model = ElasticNetLinearModel(lambda_=lambda_, alpha=alpha, max_iter=max_iter, tol=tol)
        model.fit(X, y)
        self._copy_from(model)

    def fit_ridge_cv(self, X: np.ndarray, y: np.ndarray, lambda_grid: list, k: int = 5) -> None:
        results = cross_val_score(X, y, lambda lambda_: RidgeLinearModel(lambda_), {"lambda_": lambda_grid}, k)
        best_lambda = max(results.items(), key=lambda item: item[1])[0][0]
        self.fit_ridge(X, y, lambda_=best_lambda)
        self._best_lambda_ = best_lambda

    def fit_lasso_cv(self, X: np.ndarray, y: np.ndarray, lambda_grid: list, k: int = 5) -> None:
        results = cross_val_score(X, y, lambda lambda_: LassoLinearModel(lambda_), {"lambda_": lambda_grid}, k)
        best_lambda = max(results.items(), key=lambda item: item[1])[0][0]
        self.fit_lasso(X, y, lambda_=best_lambda)
        self._best_lambda_ = best_lambda

    def fit_elasticnet_cv(self, X: np.ndarray, y: np.ndarray, lambda_grid: list, alpha_grid: list, k: int = 5) -> None:
        results = cross_val_score(
            X, y,
            model_fn=lambda lambda_, alpha: ElasticNetLinearModel(lambda_, alpha),
            param_grid={"lambda_": lambda_grid, "alpha": alpha_grid},
            k=k
        )
        best = max(results.items(), key=lambda item: item[1])[0]
        self.fit_elasticnet(X, y, lambda_=best[0], alpha=best[1])
        self._best_lambda_ = best[0]
        self._best_alpha_ = best[1]

    def _copy_from(self, model: Any) -> None:
        self.coefficients = model.coefficients
        self.intercept = model.intercept
        self.fitted_values = model.fitted_values
        self.residuals = model.residuals
        self.X_design_ = model.X_design_
