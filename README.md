# **linmod** – Linear Modeling and Inference in Python  
*A modular NumPy-based framework for classical, regularized, and robust regression models*

[![PyPI version](https://img.shields.io/pypi/v/linmod)](https://pypi.org/project/linmod/)
[![Python versions](https://img.shields.io/pypi/pyversions/linmod)](https://pypi.org/project/linmod/)
<!-- [![CI](https://github.com/DiogoRibeiro7/linmod/actions/workflows/test.yml/badge.svg)](https://github.com/DiogoRibeiro7/linmod/actions/workflows/test.yml)
[![Coverage Report](https://github.com/DiogoRibeiro7/linmod/actions/workflows/test.yml/badge.svg?branch=main)](https://DiogoRibeiro7.github.io/linmod/) -->


📍 **Project Roadmap:** [See ROADMAP.md](./ROADMAP.md)

---

## 📦 Overview

**linmod** is a lightweight and extensible Python library for linear modeling and statistical inference. It is designed for users who need control and transparency over the modeling process, with pure NumPy implementations and modular design principles.

It supports:

- Ordinary Least Squares (OLS) with full inference
- Robust diagnostics and heteroscedasticity tests
- Weighted and Generalized Least Squares (WLS, GLS)
- Regularization (Ridge, Lasso, Elastic Net)
- Functional form and normality checks
- Cross-validation utilities for model selection
- Optional export to LaTeX and integration with Streamlit

---

## 📁 Project Structure

```text
linmod/
├── core.py                   # LinearModel (main coordinator class)
│
├── model/                    # Modeling strategies (OLS, WLS, GLS)
│   ├── ols.py                # Ordinary Least Squares implementation
│   ├── wls.py                # Weighted Least Squares model
│   └── gls.py                # Generalized Least Squares model
│
├── inference/                # Statistical inference methods
│   ├── base.py               # LinearInferenceMixin
│   ├── summary.py            # Summary statistics and ANOVA
│   └── robust_se.py          # HC0–HC4 robust SE computation
│
├── diagnostics/              # Diagnostic test modules
│   ├── normality.py          # Shapiro-Wilk, D’Agostino K², heuristic
│   ├── heteroscedasticity.py # BP, White, GQ, Park, Glejser, variance power
│   ├── functional_form.py    # RESET, White-nonlinearity, Harvey–Collier
│
├── transform/                # Transformation logic and mixins
│   ├── recommend.py          # recommend_transformations()
│   ├── build.py              # suggest_transformed_model()
│   ├── fit.py                # fit_transformed() logic (optional)
│   └── mixin.py              # TransformMixin
│
├── mixins/                   # General-purpose mixins
│   ├── diagnostics.py        # DiagnosticsMixin: .diagnostics(), .to_latex()
│   └── normality.py          # NormalityMixin: .normality_summary(), .to_latex()
│
├── regularization/           # Penalized regression models
│   ├── ridge.py              # RidgeLinearModel
│   ├── lasso.py              # LassoLinearModel
│   └── elasticnet.py         # ElasticNetLinearModel
│
├── evaluation/               # Evaluation tools
│   └── crossval.py           # Cross-validation for model selection
│
├── formula/                  # [Planned] Patsy-style formulas and parsing
│   └── ...
│
├── timeseries/               # [Planned] Time series diagnostics (e.g., DW)
│   └── ...
│
├── data/                     # [Optional] Sample datasets
│   └── __init__.py
│
├── tests/                    # Unit and integration tests
│   ├── test_linmod.py
│   ├── test_stats_models.py
│   └── test_regularization_models.py
│
├── notebooks/                # Interactive demos and exploration
│   └── demo.ipynb
│
├── app.py                    # Streamlit-based interactive dashboard
├── pyproject.toml            # Poetry config
├── README.md
└── LICENSE
```

---

## 🔍 Key Features

- 📐 **OLS with full inference**: standard/robust SEs, t-tests, confidence intervals, F-test
- 📊 **Influence diagnostics**: leverage, studentized residuals, Cook's D, DFFITS
- 📉 **Heteroscedasticity tests**: Breusch–Pagan, White
- 🧮 **Regularization**: Ridge, Lasso, and Elastic Net with optional cross-validation
- 📤 **Export support**: LaTeX summaries and textual diagnostics
- 📈 **Streamlit app**: explore model fits, residuals, and influence interactively

---

## ✅ Installation (once published to PyPI)

```bash
pip install linmod
```

Until then, clone the repo and use `poetry install`.

---

## 🧪 Examples

Run the included notebook:

```bash
jupyter lab notebooks/demo.ipynb
```

Or launch the dashboard:

```bash
streamlit run app.py
```

---

## 📌 Status

- ✅ OLS, Ridge, Lasso, ElasticNet
- ✅ Statistical inference (robust, classical)
- ✅ Visual diagnostics and plots
- 🟡 Planned: ANOVA tables via formula, time series support
- 🟡 Planned: Bayesian linear regression and bootstrapping

---

## 📄 License

MIT License. See `LICENSE` for details.
