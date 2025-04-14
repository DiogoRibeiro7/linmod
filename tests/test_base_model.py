import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from linmod.base import BaseLinearModel

@pytest.fixture
def sample_data():
    """
    Provides a simple dataset with a linear relationship: y = 2x + 3
    """
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(100, 1))
    y = 2 * X[:, 0] + 3 + rng.normal(0, 0.1, size=100)
    return X, y

def test_fit_ols_via_pinv(sample_data):
    X, y = sample_data
    model = BaseLinearModel()
    
    beta, fitted, residuals = model._fit_ols_via_pinv(X, y)

    # Assert shapes
    assert beta.shape == (2,)
    assert fitted.shape == (100,)
    assert residuals.shape == (100,)

    # Assert model attributes are correctly assigned
    assert model.X_design_.shape == (100, 2)
    assert model.intercept is not None
    assert model.coefficients.shape == (1,)
    assert model.fitted_values.shape == (100,)
    assert model.residuals.shape == (100,)

    # The residuals should have mean close to 0
    assert np.abs(np.mean(residuals)) < 0.1

def test_predict(sample_data):
    X, y = sample_data
    model = BaseLinearModel()
    model._fit_ols_via_pinv(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.allclose(preds, model.fitted_values, atol=1e-6)

def test_predict_without_fit_raises():
    model = BaseLinearModel()
    X = np.random.rand(10, 1)

    with pytest.raises(ValueError, match="Model is not fit yet."):
        model.predict(X)
