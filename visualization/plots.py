"""Plotly figure factories for scientific research visualizations.

Every function returns a ``plotly.graph_objects.Figure`` with the
project's dark colour theme applied.  Charts consume **actual stored
data** — no fabricated values.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from visualization.theme import COLORS

# ── colour sequences ─────────────────────────────────────────────────

_SEQ = [COLORS["accent"], COLORS["accent2"], COLORS["highlight"],
        "#F472B6", "#FB923C", "#A78BFA", "#34D399", "#FBBF24"]


# ── base layout ──────────────────────────────────────────────────────

def get_plotly_layout(**overrides) -> dict:
    """Return a base layout dict for Plotly figures."""
    base = dict(
        paper_bgcolor=COLORS["bg_secondary"],
        plot_bgcolor="#0a1120",
        font=dict(color=COLORS["text"], family="Inter, Segoe UI, sans-serif", size=12),
        xaxis=dict(gridcolor=COLORS["panel_border"], zerolinecolor=COLORS["panel_border"],
                   title_font=dict(size=13)),
        yaxis=dict(gridcolor=COLORS["panel_border"], zerolinecolor=COLORS["panel_border"],
                   title_font=dict(size=13)),
        colorway=_SEQ,
        margin=dict(l=60, r=24, t=48, b=48),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor=COLORS["panel"], font_color=COLORS["text"],
                        bordercolor=COLORS["panel_border"]),
    )
    base.update(overrides)
    return base


def _apply(fig: go.Figure, **kw) -> go.Figure:
    fig.update_layout(**get_plotly_layout(**kw))
    return fig


# ── gradient distribution histogram ─────────────────────────────────

def gradient_distribution(
    gradients: np.ndarray,
    title: str = "Gradient Distribution",
    log_y: bool = False,
    nbins: int = 40,
) -> go.Figure:
    """Histogram of gradient values with mean / median markers."""
    flat = np.asarray(gradients).ravel()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=flat, nbinsx=nbins,
        marker_color=COLORS["accent"], opacity=0.75,
        name="Gradient values",
        hovertemplate="Bin: %{x:.4f}<br>Count: %{y}<extra></extra>",
    ))
    mean_v = float(np.mean(flat))
    median_v = float(np.median(flat))
    for val, label, color, dash in [
        (mean_v, f"Mean = {mean_v:.4f}", COLORS["highlight"], "dash"),
        (median_v, f"Median = {median_v:.4f}", COLORS["accent2"], "dot"),
    ]:
        fig.add_vline(x=val, line=dict(color=color, width=2, dash=dash),
                      annotation_text=label, annotation_font_color=color)

    ytype = "log" if log_y else "linear"
    return _apply(fig, title=title,
                  xaxis_title="Gradient value", yaxis_title="Count",
                  yaxis_type=ytype, height=380)


# ── gradient variance vs depth / qubits ──────────────────────────────

def gradient_variance_plot(
    results: dict,
    qubit_counts: list[int],
    depth_counts: list[int],
    title: str = "Gradient Variance vs Circuit Depth",
) -> go.Figure:
    """Line plot with one trace per qubit count."""
    fig = go.Figure()
    for i, nq in enumerate(qubit_counts):
        variances = []
        for d in depth_counts:
            key = f"{nq}q_{d}L"
            if key in results:
                variances.append(results[key]["gradient_variance_per_param_mean"])
            else:
                variances.append(None)
        fig.add_trace(go.Scatter(
            x=depth_counts, y=variances,
            mode="lines+markers", name=f"{nq} qubits",
            line=dict(color=_SEQ[i % len(_SEQ)], width=2),
            marker=dict(size=7),
            hovertemplate=f"{nq}q depth=%{{x}}<br>var=%{{y:.2e}}<extra></extra>",
        ))
    return _apply(fig, title=title,
                  xaxis_title="Circuit Depth (layers)",
                  yaxis_title="Mean Gradient Variance",
                  yaxis_type="log", height=420)


# ── gradient heatmap ─────────────────────────────────────────────────

def gradient_heatmap(
    results: dict,
    qubit_counts: list[int],
    depth_counts: list[int],
    metric_key: str = "gradient_variance_per_param_mean",
    title: str = "Gradient Variance Heatmap",
) -> go.Figure:
    """2‑D heatmap: x = depth, y = qubits, colour = metric."""
    z = []
    for nq in qubit_counts:
        row = []
        for d in depth_counts:
            key = f"{nq}q_{d}L"
            val = results.get(key, {}).get(metric_key, None)
            row.append(val)
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(d) for d in depth_counts],
        y=[str(q) for q in qubit_counts],
        colorscale=[
            [0.0, "#0a1120"], [0.25, "#1e3a5f"],
            [0.5, COLORS["accent2"]], [0.75, COLORS["accent"]],
            [1.0, COLORS["highlight"]],
        ],
        colorbar=dict(title=dict(text="Variance", font=dict(color=COLORS["text"])),
                      tickfont=dict(color=COLORS["text_muted"])),
        hovertemplate="Depth: %{x}<br>Qubits: %{y}<br>Value: %{z:.2e}<extra></extra>",
    ))
    return _apply(fig, title=title,
                  xaxis_title="Circuit Depth (layers)",
                  yaxis_title="Number of Qubits",
                  height=400)


# ── comparison bar chart ─────────────────────────────────────────────

def comparison_bars(
    categories: list[str],
    values: list[float],
    title: str = "",
    y_label: str = "",
    colors: list[str] | None = None,
    log_y: bool = False,
) -> go.Figure:
    """Grouped / single bar chart for strategy or architecture comparisons."""
    colors = colors or [_SEQ[i % len(_SEQ)] for i in range(len(categories))]
    fig = go.Figure(go.Bar(
        x=categories, y=values,
        marker_color=colors,
        text=[f"{v:.2e}" if abs(v) < 0.01 else f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(color=COLORS["text_muted"], size=10),
        hovertemplate="%{x}<br>%{y:.4e}<extra></extra>",
    ))
    ytype = "log" if log_y else "linear"
    return _apply(fig, title=title,
                  xaxis_title="", yaxis_title=y_label,
                  yaxis_type=ytype, height=380)


# ── training curves ──────────────────────────────────────────────────

def training_curves(
    history: list[dict],
    title: str = "Training Convergence",
) -> go.Figure:
    """Loss and accuracy over epochs (dual y‑axis)."""
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    accs = [h["accuracy"] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=losses, mode="lines+markers", name="Loss (BCE)",
        line=dict(color=COLORS["accent"], width=2), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=accs, mode="lines+markers", name="Accuracy",
        line=dict(color=COLORS["highlight"], width=2), marker=dict(size=4),
        yaxis="y2",
    ))
    return _apply(fig, title=title,
                  xaxis_title="Epoch",
                  yaxis=dict(title="Loss", gridcolor=COLORS["panel_border"],
                             zerolinecolor=COLORS["panel_border"]),
                  yaxis2=dict(title="Accuracy", overlaying="y", side="right",
                              range=[0, 1.05], gridcolor="rgba(0,0,0,0)",
                              titlefont=dict(color=COLORS["highlight"]),
                              tickfont=dict(color=COLORS["highlight"])),
                  height=400)


# ── classification scatter ───────────────────────────────────────────

def classification_scatter(
    X: np.ndarray,
    y: np.ndarray,
    predictions: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    title: str = "Dataset",
) -> go.Figure:
    """2‑D scatter plot coloured by class label."""
    fig = go.Figure()

    if predictions is not None:
        correct = predictions == y
        # errors
        err_mask = ~correct
        if np.any(err_mask):
            fig.add_trace(go.Scatter(
                x=X[err_mask, 0], y=X[err_mask, 1], mode="markers",
                marker=dict(size=10, color=COLORS["error"], symbol="x",
                            line=dict(width=1, color=COLORS["error"])),
                name="Misclassified",
            ))
        # correct
        for cls, color in [(0, COLORS["accent"]), (1, COLORS["accent2"])]:
            mask = correct & (y == cls)
            fig.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1], mode="markers",
                marker=dict(size=6, color=color, opacity=0.8),
                name=f"Class {cls} (correct)",
            ))
    else:
        for cls, color, sym in [(0, COLORS["accent"], "circle"),
                                 (1, COLORS["accent2"], "diamond")]:
            mask = y == cls
            fig.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1], mode="markers",
                marker=dict(size=7, color=color, symbol=sym, opacity=0.8),
                name=f"Class {cls}",
            ))

    return _apply(fig, title=title,
                  xaxis_title="Feature 1 (scaled)",
                  yaxis_title="Feature 2 (scaled)",
                  height=420)


# ── state‑vector bar chart ───────────────────────────────────────────

def state_vector_bars(
    state: np.ndarray,
    n_qubits: int,
    title: str = "Quantum State",
) -> go.Figure:
    """Amplitude and probability bar chart for a quantum state."""
    from plotly.subplots import make_subplots

    labels = [f"|{i:0{n_qubits}b}⟩" for i in range(len(state))]
    real = np.real(state)
    imag = np.imag(state)
    probs = np.abs(state) ** 2

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Amplitudes", "Probabilities"),
                        horizontal_spacing=0.12)

    fig.add_trace(go.Bar(x=labels, y=real, name="Real",
                         marker_color=COLORS["accent"], opacity=0.85), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=imag, name="Imaginary",
                         marker_color=COLORS["accent2"], opacity=0.85), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=probs, name="P(|x⟩)",
                         marker_color=COLORS["highlight"], opacity=0.85,
                         text=[f"{p:.3f}" for p in probs],
                         textposition="outside",
                         textfont=dict(size=9, color=COLORS["text_muted"])),
                  row=1, col=2)

    fig.update_layout(**get_plotly_layout(
        title=title, height=350, barmode="group",
    ))
    fig.update_annotations(font=dict(color=COLORS["text_muted"], size=12))
    for i in [1, 2]:
        fig.update_xaxes(tickfont=dict(size=10, color=COLORS["text_muted"]),
                         gridcolor=COLORS["panel_border"], row=1, col=i)
        fig.update_yaxes(gridcolor=COLORS["panel_border"], row=1, col=i)
    return fig


# ── near‑zero gradient fraction ─────────────────────────────────────

def near_zero_fraction_plot(
    categories: list[str],
    fractions: list[float],
    threshold: float,
    title: str = "Fraction of Near‑Zero Gradients",
) -> go.Figure:
    """Bar chart of near‑zero gradient fractions with threshold annotation."""
    fig = comparison_bars(categories, fractions, title=title,
                          y_label=f"Fraction |∂C/∂θ| < {threshold:.0e}")
    fig.add_annotation(
        text=f"Threshold = {threshold:.0e}  (operational definition)",
        xref="paper", yref="paper", x=0.98, y=1.06,
        showarrow=False, font=dict(size=10, color=COLORS["text_muted"]),
    )
    fig.update_yaxes(range=[0, 1.05])
    return fig
