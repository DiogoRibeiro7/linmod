# app.py
import streamlit as st
import pandas as pd
import numpy as np
from linmod.core import LinearModel

st.set_page_config(page_title="linmod – Linear Model Explorer", layout="wide")

st.title("linmod – Linear Model Explorer")
st.write("Upload your dataset, choose variables, and explore diagnostics and transformations.")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    with st.form("variable_selection"):
        all_vars = list(data.columns)
        y_col = st.selectbox("Select target variable (y)", all_vars)
        x_cols = st.multiselect("Select predictor variables (X)", [v for v in all_vars if v != y_col])
        alpha = st.slider("Significance level (alpha)", 0.01, 0.20, 0.05, 0.01)
        submit = st.form_submit_button("Fit Linear Model")

    if submit and y_col and x_cols:
        X = data[x_cols].to_numpy()
        y = data[y_col].to_numpy()

        model = LinearModel()
        model.fit(X, y, alpha=alpha)

        st.subheader("Model Summary")
        summary = model.summary()
        coef_table = pd.DataFrame({
            "Coefficient": [summary["intercept"]] + list(summary["coefficients"]),
            "Std. Error": [np.nan] + list(summary["std_errors"]),
            "p-value": [np.nan] + list(summary["p_values"]),
        }, index=["Intercept"] + x_cols)
        st.dataframe(coef_table.style.format("{:.4f}"))

        st.write(f"**R-squared**: {summary['r_squared']:.4f}")
        st.write(f"**Adjusted R-squared**: {summary['adj_r_squared']:.4f}")
        st.write(f"**F-statistic**: {summary['f_statistic']:.2f}, "
                 f"**p-value**: {summary['f_p_value']:.4g}")

        st.subheader("ANOVA Table")
        anova = model.anova()
        anova_df = pd.DataFrame({
            "df": anova["df"],
            "SS": anova["SS"],
            "MS": anova["MS"],
            "F": anova["F"],
            "Pr(>F)": anova["p"]
        }, index=["Model", "Residual"])
        st.dataframe(anova_df)

        st.subheader("Diagnostics")
        diag = model.diagnostics(alpha=alpha)
        for name, test in diag.items():
            st.markdown(f"**{name.replace('_', ' ').title()}**")
            st.json(test)

        st.subheader("Recommended Transformations")
        rec = model.recommend_transformations(alpha=alpha)
        if rec["response"] or rec["predictors"]:
            st.write("**Response:**", ", ".join(rec["response"]) or "None")
            st.write("**Predictors:**", ", ".join(rec["predictors"]) or "None")
        else:
            st.success("No transformation strongly recommended.")

        if st.button("Fit Transformed Model"):
            transformed_model = model.fit_transformed(alpha=alpha)
            st.write("**Transformation steps applied:**")
            st.write(getattr(transformed_model, "_transformation_steps", []))
            st.write("**Transformed features:**")
            st.write(getattr(transformed_model, "_transformed_features", []))
            st.write("**New R²:**", f"{transformed_model.r_squared:.4f}")
            st.write("**New Adjusted R²:**", f"{transformed_model.adj_r_squared:.4f}")
