"""Interactive 3D Bloch sphere using Plotly.

The Bloch sphere represents the state of a *single* qubit.  For
multi‑qubit systems, the visualised vector is obtained by tracing out
all other qubits (reduced density matrix), which may place the vector
*inside* the sphere for entangled states.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from visualization.theme import COLORS


def state_to_bloch(state: np.ndarray, qubit: int = 0) -> tuple[float, float, float]:
    """Convert a multi‑qubit state to Bloch‑sphere coordinates of one qubit.

    Returns (x, y, z) where  x² + y² + z² ≤ 1.
    Equality holds for pure, unentangled states.
    """
    rho_q = _partial_trace_single(state, qubit)

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    rx = float(np.real(np.trace(sx @ rho_q)))
    ry = float(np.real(np.trace(sy @ rho_q)))
    rz = float(np.real(np.trace(sz @ rho_q)))
    return rx, ry, rz


def _partial_trace_single(state: np.ndarray, keep_qubit: int) -> np.ndarray:
    """Return the 2×2 reduced density matrix for *keep_qubit*."""
    state = np.asarray(state, dtype=complex)
    if state.ndim != 1 or state.size == 0:
        raise ValueError("state must be a non-empty one-dimensional array")

    n_qubits = int(np.log2(state.size))
    if 2**n_qubits != state.size:
        raise ValueError("state length must be a power of two")
    if not 0 <= keep_qubit < n_qubits:
        raise ValueError(
            f"keep_qubit={keep_qubit} out of range for {n_qubits} qubits"
        )

    state_tensor = state.reshape((2,) * n_qubits)
    traced_qubits = [q for q in range(n_qubits) if q != keep_qubit]
    if not traced_qubits:
        return np.outer(state, state.conj())

    reduced = np.tensordot(
        state_tensor,
        state_tensor.conj(),
        axes=(traced_qubits, traced_qubits),
    )
    return reduced.reshape(2, 2)


def bloch_sphere_figure(
    bx: float = 0.0,
    by: float = 0.0,
    bz: float = 1.0,
    title: str = "",
) -> go.Figure:
    """Create a Plotly 3D Bloch sphere with the state vector at (bx, by, bz)."""

    fig = go.Figure()

    # ── sphere wireframe ──────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 60)
    phi = np.linspace(0, np.pi, 30)

    # Equatorial circle (XY plane)
    fig.add_trace(go.Scatter3d(
        x=np.cos(theta), y=np.sin(theta), z=np.zeros_like(theta),
        mode="lines", line=dict(color=COLORS["panel_border"], width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    # XZ circle
    fig.add_trace(go.Scatter3d(
        x=np.cos(theta), y=np.zeros_like(theta), z=np.sin(theta),
        mode="lines", line=dict(color=COLORS["panel_border"], width=1),
        showlegend=False, hoverinfo="skip",
    ))
    # YZ circle
    fig.add_trace(go.Scatter3d(
        x=np.zeros_like(theta), y=np.cos(theta), z=np.sin(theta),
        mode="lines", line=dict(color=COLORS["panel_border"], width=1),
        showlegend=False, hoverinfo="skip",
    ))

    # ── axes ──────────────────────────────────────────────────────
    for axis_vec, label, color in [
        ([1.3, 0, 0], "X", COLORS["error"]),
        ([0, 1.3, 0], "Y", COLORS["highlight"]),
        ([0, 0, 1.3], "Z", COLORS["accent"]),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[0, axis_vec[0]], y=[0, axis_vec[1]], z=[0, axis_vec[2]],
            mode="lines+text",
            line=dict(color=color, width=2),
            text=["", label],
            textfont=dict(size=14, color=color),
            textposition="top center",
            showlegend=False, hoverinfo="skip",
        ))

    # ── basis labels ──────────────────────────────────────────────
    for pos, label in [([0, 0, 1.15], "|0⟩"), ([0, 0, -1.15], "|1⟩")]:
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode="text", text=[label],
            textfont=dict(size=12, color=COLORS["text"]),
            showlegend=False, hoverinfo="skip",
        ))

    # ── state vector ──────────────────────────────────────────────
    r = np.sqrt(bx**2 + by**2 + bz**2)
    purity_label = "pure" if np.isclose(r, 1.0, atol=0.05) else f"mixed (r={r:.2f})"

    fig.add_trace(go.Scatter3d(
        x=[0, bx], y=[0, by], z=[0, bz],
        mode="lines", line=dict(color=COLORS["accent"], width=5),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=[bx], y=[by], z=[bz],
        mode="markers+text",
        marker=dict(size=7, color=COLORS["accent"], symbol="diamond"),
        text=[f"({bx:.2f}, {by:.2f}, {bz:.2f})"],
        textfont=dict(size=10, color=COLORS["text"]),
        textposition="top center",
        name=f"|ψ⟩ ({purity_label})",
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
    ))

    # ── layout ────────────────────────────────────────────────────
    axis_common = dict(
        range=[-1.5, 1.5],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        showbackground=True,
        backgroundcolor="#0a1120",
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="", **axis_common),
            yaxis=dict(title="", **axis_common),
            zaxis=dict(title="", **axis_common),
            bgcolor="#0a1120",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
        ),
        paper_bgcolor=COLORS["bg_secondary"],
        margin=dict(l=0, r=0, t=30, b=0),
        height=420,
        title=dict(text=title, font=dict(color=COLORS["text"], size=14)),
        legend=dict(font=dict(color=COLORS["text"]), bgcolor="rgba(0,0,0,0)"),
    )
    return fig
