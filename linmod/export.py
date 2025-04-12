def export_latex(model) -> str:
    """
    Generate a LaTeX summary of the model results.
    """
    lines = []

    lines.append(r"\section*{Linear Model Summary}")
    lines.append(r"\begin{itemize}")
    lines.append(rf"\item Residual Std. Error: {model.residual_std_error:.4f}")
    lines.append(rf"\item $R^2$: {model.r_squared:.4f}, Adjusted $R^2$: {model.adj_r_squared:.4f}")
    lines.append(rf"\item F-statistic: {model.f_statistic:.4f}, $p$-value: {model.f_p_value:.4g}")
    lines.append(r"\end{itemize}")

    lines.append(r"\subsection*{Inference Configuration}")
    lines.append(r"\begin{itemize}")
    lines.append(rf"\item Confidence Level ($1 - \alpha$): {1 - model._inference_alpha:.2%}")
    lines.append(rf"\item Robust SE Type: {model._inference_robust}")
    lines.append(r"\end{itemize}")

    return "\n".join(lines)
