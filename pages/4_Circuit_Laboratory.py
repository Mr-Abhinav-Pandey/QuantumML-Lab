"""Page 4 — Circuit Laboratory.

Interactive circuit builder with live visualisation, state inspection,
Bloch sphere, and measurement probabilities.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import streamlit as st

st.set_page_config(page_title="Circuit Laboratory — QuantumML Lab", page_icon="🔬", layout="wide")

from visualization.theme import inject_custom_css, sidebar_config, page_header, COLORS
from visualization.circuits import render_circuit_svg
from visualization.bloch import bloch_sphere_figure, _partial_trace_single
from visualization.plots import state_vector_bars
from quantumml.simulator import StatevectorSimulator
from quantumml.circuits import VariationalAnsatz, angle_encoding, ENTANGLEMENT_TYPES
from quantumml.training.initialization import initialize_parameters, INIT_STRATEGIES

inject_custom_css()
sidebar_config()
page_header("Circuit Laboratory",
            "Build, inspect, and explore variational quantum circuits interactively", "🔬")

# ── sidebar controls ─────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Circuit Configuration")
    n_qubits = st.slider("Qubits", 1, 8, 2, key="lab_qubits")
    n_layers = st.slider("Variational layers", 1, 8, 2, key="lab_layers")
    entanglement = st.selectbox("Entanglement", list(ENTANGLEMENT_TYPES), key="lab_ent")
    init_strategy = st.selectbox("Initialisation", list(INIT_STRATEGIES), key="lab_init")
    seed = st.number_input("Random seed", 0, 99999, 42, key="lab_seed")

    st.markdown("### Data Encoding")
    encode_data = st.checkbox("Apply angle encoding", value=True, key="lab_encode")
    if encode_data:
        x0 = st.slider("Feature x₀", 0.0, 1.0, 0.5, 0.01, key="lab_x0")
        x1 = st.slider("Feature x₁", 0.0, 1.0, 0.5, 0.01, key="lab_x1")
    else:
        x0, x1 = 0.5, 0.5

    st.markdown("### Measurement")
    meas_mode = st.radio("Mode", ["Exact (statevector)", "Finite shots"], key="lab_meas")
    shots = 1000
    if meas_mode == "Finite shots":
        shots = st.select_slider("Shots", [100, 500, 1000, 5000, 10000], value=1000, key="lab_shots")

# ── build circuit ────────────────────────────────────────────────────

ansatz = VariationalAnsatz(n_qubits, n_layers, entanglement)
params = initialize_parameters(ansatz.parameter_shape, init_strategy, seed=int(seed))

# ── circuit diagram ──────────────────────────────────────────────────

st.markdown("### Circuit Diagram")
n_features = 2 if encode_data else 0
svg = render_circuit_svg(n_qubits, n_layers, entanglement, params, n_features=n_features)
st.markdown(f'<div style="overflow-x:auto; padding:0.5rem 0;">{svg}</div>',
            unsafe_allow_html=True)

# ── circuit info ─────────────────────────────────────────────────────

ic1, ic2, ic3, ic4 = st.columns(4)
ic1.metric("Qubits", n_qubits)
ic2.metric("Layers", n_layers)
ic3.metric("Parameters", ansatz.num_parameters)
ic4.metric("Entanglement", entanglement)

# ── run simulation ───────────────────────────────────────────────────

sim = StatevectorSimulator(n_qubits)
if encode_data:
    angle_encoding(sim, np.array([x0, x1]))
ansatz.apply(sim, params)

# ── state visualisation ──────────────────────────────────────────────

st.markdown("---")
t1, t2, t3, t4 = st.tabs(["State Vector", "Bloch Sphere", "Measurements", "Parameters"])

with t1:
    st.markdown("### Quantum State")
    fig_state = state_vector_bars(sim.state, n_qubits, title="Circuit Output State")
    st.plotly_chart(fig_state, use_container_width=True, key="lab_state_bars")

    st.markdown("#### Amplitude Table")
    import pandas as pd
    state = sim.state
    labels = [f"|{i:0{n_qubits}b}⟩" for i in range(len(state))]
    df = pd.DataFrame({
        "Basis": labels,
        "Real": np.real(state),
        "Imaginary": np.imag(state),
        "Magnitude": np.abs(state),
        "Probability": np.abs(state) ** 2,
    })
    st.dataframe(df.style.format({
        "Real": "{:.5f}", "Imaginary": "{:.5f}",
        "Magnitude": "{:.5f}", "Probability": "{:.5f}",
    }), use_container_width=True, hide_index=True)

with t2:
    st.markdown("### Bloch Sphere (Reduced State)")
    if n_qubits == 1:
        qubit_sel = 0
    else:
        qubit_sel = st.selectbox("Inspect qubit", list(range(n_qubits)), key="bloch_qubit")

    rho_q = _partial_trace_single(sim.state, qubit_sel)
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    bx = float(np.real(np.trace(sx @ rho_q)))
    by = float(np.real(np.trace(sy @ rho_q)))
    bz = float(np.real(np.trace(sz @ rho_q)))
    purity = float(np.real(np.trace(rho_q @ rho_q)))

    bc1, bc2 = st.columns([2, 1])
    with bc1:
        bfig = bloch_sphere_figure(bx, by, bz, title=f"Qubit {qubit_sel}")
        st.plotly_chart(bfig, use_container_width=True, key="lab_bloch")
    with bc2:
        st.metric("Bloch x", f"{bx:.4f}")
        st.metric("Bloch y", f"{by:.4f}")
        st.metric("Bloch z", f"{bz:.4f}")
        r_len = np.sqrt(bx**2 + by**2 + bz**2)
        st.metric("|r|", f"{r_len:.4f}")
        st.metric("Purity Tr(ρ²)", f"{purity:.4f}")
        if purity < 0.99:
            st.caption("⚠️ Purity < 1 indicates entanglement with other qubits.")

with t3:
    st.markdown("### Measurement Outcomes")
    mc1, mc2 = st.columns(2)
    with mc1:
        exp_z0 = sim.expectation_value_z(0)
        p_class1 = (1 - exp_z0) / 2
        st.metric("⟨Z₀⟩ (exact)", f"{exp_z0:.5f}")
        st.metric("p(class=1)", f"{p_class1:.5f}")

    with mc2:
        if meas_mode == "Finite shots":
            exp_z0_shots = sim.expectation_value_z_shots(0, shots, seed=int(seed))
            p_class1_shots = (1 - exp_z0_shots) / 2
            st.metric(f"⟨Z₀⟩ ({shots} shots)", f"{exp_z0_shots:.5f}")
            st.metric(f"p(class=1) ({shots} shots)", f"{p_class1_shots:.5f}")
            st.caption("Shot‑based values are stochastic approximations.")
        else:
            st.info("Switch to 'Finite shots' in the sidebar to compare.")

    st.markdown("#### Probability distribution")
    probs = sim.get_probabilities()
    import plotly.graph_objects as go
    from visualization.plots import get_plotly_layout
    pfig = go.Figure(go.Bar(
        x=[f"|{i:0{n_qubits}b}⟩" for i in range(len(probs))],
        y=probs,
        marker_color=COLORS["accent"],
        text=[f"{p:.3f}" for p in probs],
        textposition="outside",
        textfont=dict(size=9, color=COLORS["text_muted"]),
    ))
    pfig.update_layout(**get_plotly_layout(
        title="Measurement Probabilities",
        xaxis_title="Basis State", yaxis_title="Probability",
        yaxis_range=[0, max(probs) * 1.2 + 0.05], height=350,
    ))
    st.plotly_chart(pfig, use_container_width=True, key="lab_probs")

with t4:
    st.markdown("### Learned / Initialised Parameters")
    st.markdown(f"**Strategy:** {init_strategy} &emsp; **Seed:** {seed}")
    param_data = []
    for l in range(n_layers):
        for q in range(n_qubits):
            param_data.append({
                "Layer": l + 1,
                "Qubit": q,
                "RY angle": f"{params[l, q, 0]:.5f}",
                "RZ angle": f"{params[l, q, 1]:.5f}",
            })
    st.dataframe(pd.DataFrame(param_data), use_container_width=True, hide_index=True)
