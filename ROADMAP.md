# linmod – Roadmap

This roadmap outlines the development milestones for **`linmod`**, a modular and transparent toolkit for linear modeling with NumPy. The work is structured across distinct stages of functionality, diagnostics, modeling features, and interface tools.

---

## ✅ Stage 1 – Core Modeling (COMPLETE)

- [x] `LinearModel` class with OLS support
- [x] Modular support for WLS, GLS, Ridge, Lasso, ElasticNet
- [x] `LinearInferenceMixin`: t-stats, p-values, ANOVA, CI, F-tests
- [x] Cross-validation support for penalized models (`crossval`)

---

## ⏳ Stage 1.1 – Core Modeling Extensions (PLANNED)

- [ ] **Generalized Linear Models (GLMs)**
  - [ ] Logistic regression
  - [ ] Poisson regression
  - [ ] Canonical links and deviance measures

- [ ] **Multivariate Linear Regression**
  - [ ] Multiple response variables `Y ∈ ℝⁿˣᵖ`
  - [ ] Per-output residuals and summaries

- [ ] **Robust Linear Regression**
  - [ ] M-estimators (Huber loss, etc.)
  - [ ] Configurable robustness parameters

- [ ] **Alternative Solvers**
  - [ ] QR decomposition
  - [ ] Gradient Descent (educational/benchmarking)

- [ ] **Linear Constraints**
  - [ ] Parameter constraints `Rβ = q`
  - [ ] Lagrangian-based optimization

- [ ] **Model Serialization**
  - [ ] `.to_dict()` and `.from_dict()`
  - [ ] `.save()` and `.load()` via JSON or Pickle

---

## ✅ Stage 2 – Diagnostics & Transformations (COMPLETE)

- [x] Residual-based tests: White, Breusch–Pagan, Goldfeld–Quandt, Park
- [x] Normality tests: Shapiro–Wilk, D’Agostino K², heuristic rule
- [x] Functional form tests: RESET, Harvey–Collier, White nonlinearity
- [x] Transformation suggestion system
- [x] Transformed model fitting (`fit_transformed()`)

---

## ⏳ Stage 2.1 – Extended Diagnostics & Transformation Tools (PLANNED)

### Influence & Outlier Measures
- [ ] Cook’s Distance with flagging
- [ ] DFFITS, DFBETAS
- [ ] Leverage diagnostics
- [ ] Unified `.influence_summary()`

### Residual-Aware Fitting
- [ ] `.fit_robust(trim='cook')` to exclude outliers
- [ ] Visual overlays for high-leverage points

### Multicollinearity Checks
- [ ] VIF computation
- [ ] Condition number filtering
- [ ] VIF in `.summary()` and `.diagnostics()`

### Auto Diagnostics Reporting
- [ ] `.diagnostics_report()` in text, LaTeX, or HTML
- [ ] Tagging of significant/warning thresholds

### Robust Transformation Suggestions
- [ ] Box–Cox via log-likelihood optimization
- [ ] Residual-feature correlation scan
- [ ] Improved targeted transformation logic

### Pipeline Comparison & Visualizations
- [ ] `.diagnostic_pipeline()` for before/after model comparison
- [ ] Heatmaps: residuals vs predictors
- [ ] Annotated plots with outlier tags

### (Optional)
- [ ] Breusch–Godfrey test (autocorrelation)
- [ ] Levene and Bartlett tests (group variance)

---

## ✅ Stage 3 – Visualization (COMPLETE)

- [x] Residuals vs Fitted
- [x] Standardized residuals vs leverage (Cook’s Distance)
- [x] Q–Q Plot for normality
- [x] Component + Residual plots
- [x] Streamlit-friendly plotting interface

---

## 🟨 Stage 4 – Packaging & Release (IN PROGRESS)

- [x] `pyproject.toml` with Poetry
- [x] PyPI metadata
- [x] MIT license
- [ ] `make-release.sh` automation
- [ ] PyPI badge for README
- [ ] GitHub release workflow

---

## 🟨 Stage 5 – Testing & CI

- [x] `pytest` test suite
- [ ] Test coverage metrics via `coverage`
- [ ] HTML reports for test coverage
- [ ] GitHub Pages publishing of test summary
- [ ] GitHub Actions pipeline (`.github/workflows/test.yml`)

---

## ⏳ Stage 6 – Model Expansion

- [ ] `formula/` module
  - [ ] Patsy-style parser `y ~ x1 + x2`
  - [ ] Dummy variable support
  - [ ] One-way ANOVA

- [ ] `timeseries/` module
  - [ ] Lagged predictor modeling
  - [ ] Durbin–Watson test
  - [ ] Autocorrelation metrics and PACF

---

## ⏳ Stage 7 – Reporting and Export

- [x] LaTeX exports for summaries and diagnostics
- [ ] HTML export of results
- [ ] Save plots to PNG/PDF
- [ ] PDF reports combining summaries and plots

---

## ⏳ Stage 8 – Interface & Tools

- [x] Streamlit dashboard (`app.py`)
- [ ] Model comparison UI
- [ ] Transformation selector GUI
- [ ] Export to `.pkl` or `.json`

---

## 🔮 Future Possibilities

- [ ] Native support for `pandas.DataFrame` inputs
- [ ] Microcontroller-ready linear models (e.g., TFLite-style export)
- [ ] scikit-learn or statsmodels wrapper
- [ ] Lightweight AutoML for linear models

---

**Last updated:** 2025-04-12
