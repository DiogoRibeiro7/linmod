# linmod - Linear Modeling and Inference in Python: A NumPy-Based Framework for Classical and Regularized Regression
```text
linmod/
├── __init__.py
├── core.py                      # LinearModel core + wrappers: WLS, GLS, Ridge, Lasso, ElasticNet
├── diagnostics.py               # Residual tests, normality, functional form
├── transforms.py                # Model suggestion, fit_transformed()
├── plots.py                     # Residual, leverage, influence, CR plots
├── export.py                    # LaTeX summaries
├── utils.py                     # Weight suggestion, residual ops

├── stats/                       # Statistical extensions
│   ├── wls.py                   # Weighted Linear Model
│   ├── gls.py                   # Generalized Least Squares
│   ├── hypothesis.py            # Linear constraint tests (Rβ = q)

├── regularization/             # Penalized linear models
│   ├── ridge.py                 # Ridge regression (closed-form)
│   ├── lasso.py                 # Lasso regression (coordinate descent)
│   ├── elasticnet.py            # ElasticNet regression

├── formula/                    # [Planned] Categorical encoding, formulas, ANOVA
│   └── ...

├── timeseries/                 # [Planned] Lag models, Durbin-Watson, autocorrelation tests
│   └── ...

├── evaluation/                 # Model comparison and scoring
│   ├── crossval.py             # Generic CV runner for Ridge, Lasso, ElasticNet

├── data/                       # [Optional] datasets
│   └── __init__.py

├── notebooks/                  # Interactive demos
│   └── demo.ipynb

├── tests/                      # Unit tests
│   ├── test_linmod.py
│   ├── test_stats_models.py
│   ├── test_regularization_models.py

├── app.py                      # Streamlit app
├── pyproject.toml              # Poetry build config
├── README.md
└── LICENSE
```