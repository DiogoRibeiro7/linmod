# linmod/mixins/transform.py

from typing import Any, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from linmod.transforms.build import suggest_transformed_model

import numpy as np

class DiagnosticsProvider(ABC):
    @abstractmethod
    def diagnostics(self, alpha: float = 0.05) -> dict[str, Any]:
        ...


@runtime_checkable
class FitModelProtocol(Protocol):
    """
    Protocol for models that implement a fit method
    and provide access to core regression attributes.
    """
    X_design_: np.ndarray
    residuals: np.ndarray
    fitted_values: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...


class TransformMixin(DiagnosticsProvider, FitModelProtocol):
    _transformation_steps: list[str] | None = None
    _transformed_features: list[str] | None = None

    def recommend_transformations(self, alpha: float = 0.05) -> dict[str, list[str]]:
        diag = self.diagnostics(alpha=alpha)
        response_recs = []
        predictor_recs = []

        boxcox = diag.get("box_cox", {})
        lmb = boxcox.get("lambda", 1.0)
        if lmb < -0.5:
            response_recs.append("1/y (inverse)")
        elif -0.5 <= lmb < 0.3:
            response_recs.append("log(y)")
        elif 0.3 <= lmb < 0.8:
            response_recs.append("sqrt(y)")
        elif lmb > 1.2:
            response_recs.append("y^2")
        elif abs(lmb - 1.0) > 0.1:
            response_recs.append(f"Box–Cox: y^{lmb:.2f}")

        if diag.get("reset", {}).get("p-value", 1) < alpha or diag.get("harvey_collier", {}).get("p-value", 1) < alpha:
            predictor_recs.append("Include polynomial terms (e.g., x², x³)")

        if diag.get("white_nonlinearity", {}).get("p-value", 1) < alpha:
            predictor_recs.append("Include interaction terms (e.g., x₁ * x₂)")

        if diag.get("park", {}).get("p-value", 1) < alpha:
            predictor_recs.append("Apply log(x) or sqrt(x) to stabilize variance")

        if diag.get("glejser", {}).get("p-value", 1) < alpha:
            transformation = diag["glejser"].get("transformation", "unknown")
            predictor_recs.append(f"Try transformation: {transformation}(x)")

        if diag.get("variance_power", {}).get("p-value", 1) < alpha:
            gamma = diag["variance_power"].get("gamma (slope)", 0.0)
            if gamma < -0.5:
                predictor_recs.append("Apply inverse(x)")
            elif gamma < 0.5:
                predictor_recs.append("Apply log(x) or sqrt(x)")
            elif gamma > 1.5:
                predictor_recs.append("Consider x² or polynomial variance modeling")

        return {
            "response": list(set(response_recs)),
            "predictors": list(set(predictor_recs))
        }

    def transformation_recommendations_to_latex(self, alpha: float = 0.05) -> str:
        recs = self.recommend_transformations(alpha=alpha)
        rows = [f"Response & {r} \\\\" for r in recs["response"]] + \
               [f"Predictors & {p} \\\\" for p in recs["predictors"]]

        return (
            "\\begin{tabular}{ll}\n"
            "\\toprule\n"
            "Target & Recommended Transformation \\\\\n"
            "\\midrule\n"
            + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}"
        )

    def suggest_transformed_model(self, alpha: float = 0.05) -> dict[str, Any]:
        if not all(hasattr(self, attr) for attr in ("X_design_", "residuals", "fitted_values")):
            raise AttributeError("Model must be fit before calling suggest_transformed_model.")

        diag = self.diagnostics(alpha=alpha)
        return suggest_transformed_model(
            self.X_design_,
            self.residuals,
            self.fitted_values,
            diag,
            alpha=alpha
        )

    def fit_transformed(self, alpha: float = 0.05) -> Any:
        suggestion = self.suggest_transformed_model(alpha=alpha)
        X_trans = suggestion["X_trans"]
        y_trans = suggestion["y_trans"]

        model = self.__class__()  # expected to be a LinearModel subclass
        if not hasattr(model, "fit"):
            raise TypeError(f"{model.__class__.__name__} must implement a 'fit(X, y)' method.")

        model.fit(X_trans[:, 1:], y_trans)  # skip intercept
        model._transformation_steps = suggestion["description"]
        model._transformed_features = suggestion["features"]

        return model
