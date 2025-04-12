# **linmod** – Linear Modeling and Inference in Python  
*A modular NumPy-based framework for classical, regularized, and robust regression models*

[![PyPI version](https://img.shields.io/pypi/v/linmod)](https://pypi.org/project/linmod/)
[![Python versions](https://img.shields.io/pypi/pyversions/linmod)](https://pypi.org/project/linmod/)
[![CI](https://github.com/DiogoRibeiro7/linmod/actions/workflows/test.yml/badge.svg)](https://github.com/DiogoRibeiro7/linmod/actions/workflows/test.yml)
[![Coverage Report](https://github.com/DiogoRibeiro7/linmod/actions/workflows/test.yml/badge.svg?branch=main)](https://DiogoRibeiro7.github.io/linmod/)



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
├── core.py               # LinearModel (coordenador principal)
├── model/                # Estratégias de modelagem
│   ├── ols.py            # OLS logic
│   ├── wls.py            # WeightedLinearModel
│   ├── gls.py            # GeneralizedLinearModel
│
├── inference/            # Inferência estatística
│   ├── base.py           # LinearInferenceMixin
│   ├── summary.py        # .summary(), anova()
│   └── robust_se.py      # compute_robust_se

├── diagnostics/          # Diagnósticos e testes
│   ├── base.py           # diagnostics(), print_diagnostics(), to_latex()
│   ├── normality.py      # shapiro, dagostino_k2
│   ├── heteroscedasticity.py  # BP, White, GQ, Park, Glejser, Power
│   ├── functional_form.py     # RESET, White-nonlinearity, Harvey-Collier

├── transforms/           # Sugestões e aplicação de transformações
│   ├── recommend.py      # recommend_transformations()
│   ├── build.py          # suggest_transformed_model()
│   └── fit_transformed.py

├── regularization/
│   ├── ridge.py
│   ├── lasso.py
│   ├── elasticnet.py

├── evaluation/
│   └── crossval.py

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
