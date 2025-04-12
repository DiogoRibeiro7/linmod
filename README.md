# **linmod** – Linear Modeling and Inference in Python  
*A modular NumPy-based framework for classical, regularized, and robust regression models*

[![PyPI version](https://img.shields.io/pypi/v/linmod)](https://pypi.org/project/linmod/)
[![Python versions](https://img.shields.io/pypi/pyversions/linmod)](https://pypi.org/project/linmod/)
[![CI](https://github.com/DiogoRibeiro7/linmod/actions/workflows/test.yml/badge.svg)](https://github.com/DiogoRibeiro7/linmod/actions/workflows/test.yml)


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
├── core.py                      # LinearModel core + fit_wls(), fit_ridge(), etc.
├── inference.py                 # Inference mixin: SEs, t-tests, ANOVA, robust errors
├── diagnostics.py               # Residual tests, functional form checks
├── transforms.py                # Transformation logic, fit_clean_transformed()
├── plots.py                     # Residuals, leverage, influence, component+residual
├── export.py                    # LaTeX/text summaries
├── utils.py                     # Weights, residuals, leverage, Cook's D, DFFITS

├── stats/
│   ├── wls.py                   # Weighted Least Squares model
│   ├── gls.py                   # Generalized Least Squares model
│   └── hypothesis.py            # Linear hypothesis testing (Rβ = q)

├── regularization/
│   ├── ridge.py                 # Ridge regression (closed-form)
│   ├── lasso.py                 # Lasso regression (coordinate descent)
│   ├── elasticnet.py            # ElasticNet regression

├── evaluation/
│   └── crossval.py              # Generic k-fold CV runner for regularized models

├── formula/                    # [Planned] Patsy-style formulas, ANOVA support
│   └── ...

├── timeseries/                 # [Planned] Lag models, DW test, autocorrelation
│   └── ...

├── data/                       # [Optional] Sample datasets
│   └── __init__.py

├── notebooks/                  # Interactive examples
│   └── demo.ipynb

├── tests/                      # Unit tests
│   ├── test_linmod.py
│   ├── test_stats_models.py
│   ├── test_regularization_models.py

├── app.py                      # Streamlit dashboard (interactive model explorer)
├── pyproject.toml              # Poetry-based project definition
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
