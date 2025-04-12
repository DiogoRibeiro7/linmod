# linmod – Roadmap

This roadmap outlines the ongoing development plan for `linmod`, a modular and transparent toolkit for linear modeling built with NumPy. Progress is organized by technical milestones and development areas.

---

## ✅ Stage 1 – Core Modeling (COMPLETE)

- [x] LinearModel with OLS fitting
- [x] Support for WLS, GLS, Ridge, Lasso, and ElasticNet
- [x] Inference Mixin with SE, t, p, F-statistic, and ANOVA
- [x] Grid search via cross-validation for penalized models

---

## ⏳ Stage 1.1 – Core Modeling Extensions (PLANNED)

- [ ] Generalized Linear Models (GLMs)
  - [ ] Logistic regression (Binomial family)
  - [ ] Poisson regression
  - [ ] Canonical links and deviance measures

- [ ] Multivariate Linear Regression (multiple response variables)
  - [ ] Fit Y ∈ ℝⁿˣᵖ with shared X
  - [ ] Summary statistics and residuals per response

- [ ] Robust Linear Regression
  - [ ] M-estimators (e.g., Huber loss)
  - [ ] Tuning of breakdown point or threshold

- [ ] Alternative Solvers
  - [ ] QR decomposition (`X = QR ⇒ β = R⁻¹Qᵗy`)
  - [ ] Gradient Descent OLS (educational use)

- [ ] Linear Constraints on Parameters
  - [ ] Equality constraints: `Rβ = q`
  - [ ] Lagrangian method for solving under constraints

- [ ] Model Export & Interchange
  - [ ] `.to_dict()` and `.from_dict()` for lightweight serialization
  - [ ] `.save(filepath)` and `.load(filepath)` using pickle or JSON

---

## ✅ Stage 2 – Diagnostics & Transformations (COMPLETE)

- [x] Heteroscedasticity tests (White, BP, GQ, Park, etc.)
- [x] Normality tests (Shapiro–Wilk, D’Agostino K², heuristic)
- [x] Functional form tests (RESET, Harvey–Collier, White nonlinearity)
- [x] Transformation suggestions for predictors and response
- [x] Auto-fitting transformed models

---

## ⏳ Stage 2.1 – Extended Diagnostics & Transformation Tools (PLANNED)

### Outlier and Influence Measures
- [ ] Cook’s Distance thresholds and tagging
- [ ] DFFITS and DFBETAS statistics
- [ ] Leverage-based outlier detection
- [ ] Unified `.influence_summary()` returning DataFrame

### Residual-Aware Refitting
- [ ] `.fit_robust(trim="cook")` to exclude outliers before fitting
- [ ] Visual diagnostics highlighting high-leverage/influential points

### Automated Diagnostics Reporting
- [ ] `.diagnostics_report()` with output formats: text, LaTeX, HTML
- [ ] Summary tables with interpretation flags (e.g., warning, significant)

### Multicollinearity Diagnostics
- [ ] Variance Inflation Factor (VIF)
- [ ] Condition number thresholding
- [ ] Add VIF to `.summary()` and `.diagnostics()`

### Robust Transformation Suggestions
- [ ] Better Box–Cox lambda search via log-likelihood
- [ ] Residual-feature nonlinear correlation scan
- [ ] More targeted predictor transformation recommendations

### Pipeline Comparison
- [ ] `.diagnostic_pipeline()`:
  - Fit original model
  - Suggest + apply transformations
  - Fit transformed model
  - Compare metrics and residual tests side-by-side

### Diagnostic Visualization
- [ ] Annotated diagnostic plots with influence labels
- [ ] Heatmap: residuals vs. predictor correlations

### (Optional)
- [ ] Breusch–Godfrey test for autocorrelated residuals
- [ ] Levene and Bartlett tests for groupwise homoscedasticity


---

## ✅ Stage 3 – Visualization (COMPLETE)

- [x] Residuals vs Fitted plot
- [x] Standardized Residuals vs Leverage (with Cook’s Distance)
- [x] Normal Q–Q Plot
- [x] Component + Residual (CR) Plots
- [x] Streamlit-compatible version of all plots

---

## 🟨 Stage 4 – Packaging & Release (IN PROGRESS)

- [x] `pyproject.toml` with Poetry
- [x] PyPI metadata and description
- [x] MIT License
- [ ] `make-release.sh` script for publishing
- [ ] PyPI install/build badge for README
- [ ] GitHub Release automation for tags like `v*.*.*`

---

## 🟨 Stage 5 – Testing & CI

- [x] Unit test structure with `pytest`
- [ ] Test coverage via `coverage`
- [ ] HTML report generation
- [ ] Coverage publishing via GitHub Pages
- [ ] `.github/workflows/test.yml` with CI pipeline

---

## ⏳ Stage 6 – Model Expansion

- [ ] `formula/` submodule with:
  - [ ] Formula parsing (e.g., `y ~ x1 + x2`)
  - [ ] Categorical encoding (dummies)
  - [ ] ANOVA and factor models

- [ ] `timeseries/` submodule with:
  - [ ] Lagged predictors (`y_t ~ x_{t-1}`)
  - [ ] Durbin–Watson test
  - [ ] Residual autocorrelation and PACF

---

## ⏳ Stage 7 – Reporting and Export

- [x] Export to LaTeX summaries
- [ ] Export diagnostics to HTML
- [ ] Export plots as PNG/PDF
- [ ] Auto-generate PDF reports with summaries and diagnostics

---

## ⏳ Stage 8 – Interface & Tools

- [x] Streamlit app for interactive usage
- [ ] UI for model history and comparisons
- [ ] Interactive transformation selector
- [ ] Export models to `.pkl` or `.json`

---

## 🔮 Future Plans

- [ ] Native support for `pandas.DataFrame` inputs (`fit(df, y='target')`)
- [ ] Minimal build for microcontrollers / edge execution
- [ ] Integration with `statsmodels` or `scikit-learn` API
- [ ] Lightweight AutoML for linear models and transformations

---

**Last updated:** 2025-04-12
