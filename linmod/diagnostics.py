def diagnostics_report(model) -> str:
    """
    Generate a plain text diagnostics report of the linear model.
    """
    lines = []

    lines.append("=== Linear Model Diagnostics ===")
    lines.append(f"Residual Std. Error: {model.residual_std_error:.4f}")
    lines.append(f"R-squared: {model.r_squared:.4f}")
    lines.append(f"Adjusted R-squared: {model.adj_r_squared:.4f}")
    lines.append(f"F-statistic: {model.f_statistic:.4f}, p-value: {model.f_p_value:.4g}")
    lines.append("")

    lines.append("--- Inference Configuration ---")
    lines.append(f"Confidence level: {1 - model._inference_alpha:.2%}")
    lines.append(f"Robust SE type: {model._inference_robust}")
    lines.append("")

    return "\n".join(lines)
