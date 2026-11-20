"""Page 3 — Mathematical Foundations.

Interactive explanations of qubit states, quantum gates, rotation
gates, entanglement, variational circuits, expectation values,
parameter‑shift rule, and the barren‑plateau concept.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Mathematical Foundations — QuantumML Lab", page_icon="📐", layout="wide")

from visualization.theme import inject_custom_css, sidebar_config, page_header, COLORS
from visualization.bloch import bloch_sphere_figure, _partial_trace_single
from visualization.plots import state_vector_bars, get_plotly_layout
from quantumml.simulator import StatevectorSimulator, ry, rz, rx, hadamard, pauli_x, pauli_z

inject_custom_css()
sidebar_config()
page_header("Mathematical Foundations",
            "Core quantum computing concepts used in this research", "📐")

# ── tabs ─────────────────────────────────────────────────────────────

tabs = st.tabs([
    "Qubit States", "Quantum Gates", "Entanglement",
    "Variational Circuits", "Parameter‑Shift Rule", "Barren Plateaus",
])

# ── Tab 1: Qubit States ─────────────────────────────────────────────

with tabs[0]:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Single‑Qubit State")
        st.latex(r"|\psi\rangle = \cos\frac{\theta}{2}\,|0\rangle + e^{i\phi}\sin\frac{\theta}{2}\,|1\rangle")
        st.markdown(
            "A qubit lives in a two‑dimensional complex Hilbert space.  "
            "Its state can be visualised on the **Bloch sphere** where "
            "θ is the polar angle and φ is the azimuthal angle."
        )
        theta_deg = st.slider("θ (degrees)", 0, 180, 90, 5, key="bloch_theta")
        phi_deg = st.slider("φ (degrees)", 0, 360, 0, 5, key="bloch_phi")
        theta = np.radians(theta_deg)
        phi = np.radians(phi_deg)

        # Bloch coordinates
        bx = np.sin(theta) * np.cos(phi)
        by = np.sin(theta) * np.sin(phi)
        bz = np.cos(theta)
        st.markdown(
            f"**Bloch vector:** ({bx:.3f}, {by:.3f}, {bz:.3f})"
        )

        alpha = np.cos(theta / 2)
        beta = np.exp(1j * phi) * np.sin(theta / 2)
        st.markdown(
            f"**State:** {alpha:.3f} |0⟩ + ({beta:.3f}) |1⟩"
        )

    with c2:
        fig = bloch_sphere_figure(bx, by, bz, title="Interactive Bloch Sphere")
        st.plotly_chart(fig, use_container_width=True, key="bloch_main")

    st.markdown("---")
    st.markdown("### Multi‑Qubit States")
    st.latex(r"|\Psi\rangle = \sum_{i=0}^{2^n - 1} c_i\,|i\rangle, \qquad \sum_i |c_i|^2 = 1")
    st.markdown(
        "An *n*‑qubit system lives in a 2ⁿ‑dimensional Hilbert space.  "
        "The computational basis states are labelled by binary strings."
    )

# ── Tab 2: Quantum Gates ────────────────────────────────────────────

with tabs[1]:
    st.markdown("### Standard Gates")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown("**Pauli‑X (NOT)**")
        st.latex(r"X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}")
        st.caption("Bit flip: |0⟩ ↔ |1⟩")
    with g2:
        st.markdown("**Pauli‑Z**")
        st.latex(r"Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}")
        st.caption("Phase flip: |1⟩ → −|1⟩")
    with g3:
        st.markdown("**Hadamard**")
        st.latex(r"H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}")
        st.caption("Creates superposition")
    with g4:
        st.markdown("**CNOT**")
        st.latex(r"\text{CNOT} = \begin{pmatrix} 1&0&0&0\\0&1&0&0\\0&0&0&1\\0&0&1&0 \end{pmatrix}")
        st.caption("Controlled‑NOT")

    st.markdown("---")
    st.markdown("### Rotation Gates")
    st.latex(r"R_Y(\theta) = e^{-i\theta Y/2} = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}")
    st.latex(r"R_Z(\theta) = e^{-i\theta Z/2} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}")

    st.markdown("#### Gate Explorer")
    gate_choice = st.selectbox("Gate", ["RY", "RZ", "RX", "Hadamard", "Pauli‑X"], key="gate_sel")
    angle = st.slider("θ (radians)", -3.14, 3.14, 0.5, 0.01, key="gate_angle")

    sim = StatevectorSimulator(1)
    if gate_choice == "RY":
        sim.apply_single_qubit_gate(ry(angle), 0)
    elif gate_choice == "RZ":
        sim.apply_single_qubit_gate(rz(angle), 0)
    elif gate_choice == "RX":
        sim.apply_single_qubit_gate(rx(angle), 0)
    elif gate_choice == "Hadamard":
        sim.h(0)
    elif gate_choice == "Pauli‑X":
        sim.x(0)

    state = sim.state
    rho = _partial_trace_single(state, 0)
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    bx_ = float(np.real(np.trace(sx @ rho)))
    by_ = float(np.real(np.trace(sy @ rho)))
    bz_ = float(np.real(np.trace(sz @ rho)))

    gc1, gc2 = st.columns(2)
    with gc1:
        st.markdown(f"**Output state:** ({state[0]:.4f}) |0⟩ + ({state[1]:.4f}) |1⟩")
        probs = np.abs(state) ** 2
        st.markdown(f"P(|0⟩) = {probs[0]:.4f} &emsp; P(|1⟩) = {probs[1]:.4f}")
    with gc2:
        gfig = bloch_sphere_figure(bx_, by_, bz_, title=f"{gate_choice}(θ={angle:.2f}) applied to |0⟩")
        st.plotly_chart(gfig, use_container_width=True, key="gate_bloch")

# ── Tab 3: Entanglement ─────────────────────────────────────────────

with tabs[2]:
    st.markdown("### Bell State Demonstration")
    st.markdown(
        "Apply **H** to qubit 0 then **CNOT** (0→1) to produce a Bell state:"
    )
    st.latex(r"|00\rangle \xrightarrow{H \otimes I} \frac{|0\rangle + |1\rangle}{\sqrt{2}} \otimes |0\rangle \xrightarrow{\text{CNOT}} \frac{|00\rangle + |11\rangle}{\sqrt{2}}")

    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown("#### Product state (before CNOT)")
        sim_prod = StatevectorSimulator(2)
        sim_prod.h(0)
        fig_prod = state_vector_bars(sim_prod.state, 2, title="After H on qubit 0")
        st.plotly_chart(fig_prod, use_container_width=True, key="prod_state")

        rho_0_prod = _partial_trace_single(sim_prod.state, 0)
        purity_prod = float(np.real(np.trace(rho_0_prod @ rho_0_prod)))
        st.caption(f"Qubit 0 purity: Tr(ρ²) = {purity_prod:.4f} → pure (unentangled)")

    with bc2:
        st.markdown("#### Bell state (after CNOT)")
        sim_bell = sim_prod.copy()
        sim_bell.apply_cnot(0, 1)
        fig_bell = state_vector_bars(sim_bell.state, 2, title="Bell state |Φ⁺⟩")
        st.plotly_chart(fig_bell, use_container_width=True, key="bell_state")

        rho_0_bell = _partial_trace_single(sim_bell.state, 0)
        purity_bell = float(np.real(np.trace(rho_0_bell @ rho_0_bell)))
        st.caption(f"Qubit 0 purity: Tr(ρ²) = {purity_bell:.4f} → mixed (entangled)")

    st.markdown(
        "When the purity Tr(ρ²) < 1 for a subsystem, the state is **entangled**.  "
        "On the Bloch sphere, an entangled qubit's reduced state sits *inside* the sphere."
    )

# ── Tab 4: Variational Circuits ─────────────────────────────────────

with tabs[3]:
    st.markdown("### Variational Quantum Circuit (VQC)")
    st.markdown(
        "A VQC consists of:\n"
        "1. **Data encoding** — classical features mapped into quantum amplitudes.\n"
        "2. **Parameterised layers** — trainable rotation gates + entanglement.\n"
        "3. **Measurement** — expectation value used for prediction."
    )
    st.latex(r"U(\theta) = \prod_{\ell=1}^{L}\left[\text{Ent}\cdot\bigotimes_{q} R_Z(\theta_{\ell,q,1})\,R_Y(\theta_{\ell,q,0})\right]")
    st.markdown("#### Probability Mapping")
    st.latex(r"p(\text{class}=1 \mid x) = \frac{1 - \langle Z_0 \rangle}{2}")
    st.markdown(
        "⟨Z₀⟩ ∈ [−1, 1] is mapped to a probability in [0, 1].  "
        "The loss is binary cross‑entropy:"
    )
    st.latex(r"\mathcal{L} = -\frac{1}{N}\sum_{i}\left[y_i\log p_i + (1-y_i)\log(1-p_i)\right]")

# ── Tab 5: Parameter‑Shift Rule ─────────────────────────────────────

with tabs[4]:
    st.markdown("### The Parameter‑Shift Rule")
    st.markdown(
        "For rotation gates whose generator is a Pauli matrix "
        "(R_Y, R_Z, R_X), the gradient is computed **exactly** as:"
    )
    st.latex(r"\frac{\partial f}{\partial \theta_k} = \frac{f\!\left(\theta_k + \frac{\pi}{2}\right) - f\!\left(\theta_k - \frac{\pi}{2}\right)}{2}")
    st.markdown(
        "This requires **two circuit evaluations** per parameter — no "
        "approximation, no finite step size.  It is the standard method "
        "for computing gradients in variational quantum algorithms."
    )
    st.markdown(
        "The gradient of the full cost function is obtained by "
        "chaining the parameter‑shift derivative with the loss gradient:"
    )
    st.latex(r"\frac{\partial \mathcal{L}}{\partial \theta_k} = \frac{\partial \mathcal{L}}{\partial p} \cdot \frac{\partial p}{\partial \langle Z_0 \rangle} \cdot \frac{\partial \langle Z_0 \rangle}{\partial \theta_k}")

# ── Tab 6: Barren Plateaus ──────────────────────────────────────────

with tabs[5]:
    st.markdown("### The Barren Plateau Problem")
    st.markdown(
        "A **barren plateau** is a region in the cost landscape where the "
        "variance of the gradient vanishes exponentially with the number "
        "of qubits:"
    )
    st.latex(r"\text{Var}_{\theta}\!\left[\frac{\partial C}{\partial \theta_k}\right] \leq F(n)")
    st.markdown(
        "where F(n) decreases exponentially in *n* under certain conditions.  "
        "When this happens, the gradient is essentially zero for most "
        "random initialisations, making gradient‑based training impractical."
    )
    st.markdown("#### Known Contributing Factors")
    factors = {
        "Circuit depth": "Deep, highly expressive circuits can lead to Haar‑random behaviour.",
        "Global cost functions": "Observables acting on all qubits can suppress gradients faster.",
        "Random initialisation": "Sampling parameters uniformly over [0, 2π) can place the circuit in a flat region.",
        "Entanglement structure": "Highly connected ansatze may converge to 2‑designs faster.",
        "Noise": "Certain noise channels can induce noise‑induced barren plateaus.",
    }
    for factor, desc in factors.items():
        st.markdown(f"- **{factor}:** {desc}")

    st.markdown("#### What This Project Investigates")
    st.markdown(
        "We measure gradient variance across different configurations "
        "to observe whether barren‑plateau‑like behaviour appears.  "
        "We do **not** claim to prove new theoretical results; we conduct "
        "controlled experiments and compare with published findings."
    )
