"""Page 5 — Experiments.

Run experiments A, B, C, D with configurable parameters, progress
feedback, and automatic result persistence.
"""

from __future__ import annotations

import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import streamlit as st

st.set_page_config(page_title="Experiments — QuantumML Lab", page_icon="🧪", layout="wide")

from visualization.theme import inject_custom_css, sidebar_config, page_header, COLORS, panel
from visualization.circuits import render_circuit_svg
from quantumml.circuits.ansatz import VariationalAnsatz
from quantumml.training.initialization import INIT_STRATEGIES
from quantumml.circuits.entanglement import ENTANGLEMENT_TYPES

inject_custom_css()
sidebar_config()
page_header("Experiment Laboratory",
            "Configure and run barren‑plateau and classification experiments", "🧪")

exp_tabs = st.tabs([
    "A — Depth & Gradients",
    "B — Entanglement",
    "C — Initialisation",
    "D — Classification",
])

# ═══ Experiment A ════════════════════════════════════════════════════

with exp_tabs[0]:
    st.markdown("### Experiment A — Circuit Depth and Gradient Concentration")
    st.markdown(
        "Measure gradient statistics across different qubit counts and "
        "circuit depths to investigate gradient concentration."
    )

    ac1, ac2 = st.columns(2)
    with ac1:
        a_qubits = st.multiselect("Qubit counts", [2, 3, 4, 5, 6, 7, 8], default=[2, 4], key="a_q")
        a_depths = st.multiselect("Depth counts", [1, 2, 3, 4, 5, 6, 8], default=[1, 2, 4], key="a_d")
    with ac2:
        a_seeds = st.slider("Random seeds", 2, 50, 10, key="a_seeds")
        a_ent = st.selectbox("Entanglement", list(ENTANGLEMENT_TYPES), key="a_ent")
        a_init = st.selectbox("Initialisation", list(INIT_STRATEGIES), key="a_init")

    # Preview
    if a_qubits and a_depths:
        total_configs = len(a_qubits) * len(a_depths)
        total_evals = total_configs * a_seeds
        st.caption(f"📋 {total_configs} configurations × {a_seeds} seeds = {total_evals} gradient evaluations")

        with st.expander("Circuit preview (largest configuration)"):
            svg = render_circuit_svg(max(a_qubits), max(a_depths), a_ent, n_features=2)
            st.markdown(f'<div style="overflow-x:auto">{svg}</div>', unsafe_allow_html=True)

    if st.button("▶ Run Experiment A", key="run_a", type="primary", use_container_width=True):
        from quantumml.experiments.service import ExperimentService

        progress = st.progress(0, text="Starting Experiment A…")
        status = st.empty()

        def _cb(step, total, msg):
            progress.progress(step / total, text=f"[{step}/{total}] {msg}")

        t0 = time.perf_counter()
        result = ExperimentService.run(
            "depth",
            qubit_counts=sorted(a_qubits),
            depth_counts=sorted(a_depths),
            n_seeds=a_seeds,
            entanglement=a_ent,
            init_strategy=a_init,
            progress_callback=_cb,
        )
        elapsed = time.perf_counter() - t0
        progress.progress(1.0, text="Done!")

        st.success(f"✅ Experiment A completed in {elapsed:.1f}s — ID: `{result['experiment_id']}`")
        st.session_state["exp_a_result"] = result

# ═══ Experiment B ════════════════════════════════════════════════════

with exp_tabs[1]:
    st.markdown("### Experiment B — Entanglement Architecture Comparison")
    st.markdown(
        "Compare gradient statistics across linear, circular, and "
        "all‑to‑all entanglement with all other variables fixed."
    )

    bc1, bc2 = st.columns(2)
    with bc1:
        b_qubits = st.slider("Qubits", 2, 8, 4, key="b_q")
        b_layers = st.slider("Layers", 1, 8, 4, key="b_l")
    with bc2:
        b_seeds = st.slider("Random seeds", 2, 50, 10, key="b_seeds")
        b_init = st.selectbox("Initialisation", list(INIT_STRATEGIES), key="b_init")
    b_ents = st.multiselect("Architectures", list(ENTANGLEMENT_TYPES),
                             default=list(ENTANGLEMENT_TYPES), key="b_ents")

    if st.button("▶ Run Experiment B", key="run_b", type="primary", use_container_width=True):
        from quantumml.experiments.service import ExperimentService

        progress = st.progress(0, text="Starting Experiment B…")

        def _cb(step, total, msg):
            progress.progress(step / total, text=f"[{step}/{total}] {msg}")

        t0 = time.perf_counter()
        result = ExperimentService.run(
            "entanglement",
            n_qubits=b_qubits, n_layers=b_layers,
            entanglements=b_ents, n_seeds=b_seeds,
            init_strategy=b_init, progress_callback=_cb,
        )
        elapsed = time.perf_counter() - t0
        progress.progress(1.0, text="Done!")
        st.success(f"✅ Experiment B completed in {elapsed:.1f}s — ID: `{result['experiment_id']}`")
        st.session_state["exp_b_result"] = result

# ═══ Experiment C ════════════════════════════════════════════════════

with exp_tabs[2]:
    st.markdown("### Experiment C — Parameter Initialisation Comparison")
    st.markdown(
        "Compare gradient statistics across random, small‑angle, and "
        "identity‑block initialisation with all other variables fixed."
    )

    cc1, cc2 = st.columns(2)
    with cc1:
        c_qubits = st.slider("Qubits", 2, 8, 4, key="c_q")
        c_layers = st.slider("Layers", 1, 8, 4, key="c_l")
    with cc2:
        c_seeds = st.slider("Random seeds", 2, 50, 10, key="c_seeds")
        c_ent = st.selectbox("Entanglement", list(ENTANGLEMENT_TYPES), key="c_ent")
    c_strats = st.multiselect("Strategies", list(INIT_STRATEGIES),
                               default=list(INIT_STRATEGIES), key="c_strats")

    if st.button("▶ Run Experiment C", key="run_c", type="primary", use_container_width=True):
        from quantumml.experiments.service import ExperimentService

        progress = st.progress(0, text="Starting Experiment C…")

        def _cb(step, total, msg):
            progress.progress(step / total, text=f"[{step}/{total}] {msg}")

        t0 = time.perf_counter()
        result = ExperimentService.run(
            "initialization",
            n_qubits=c_qubits, n_layers=c_layers,
            strategies=c_strats, entanglement=c_ent,
            n_seeds=c_seeds, progress_callback=_cb,
        )
        elapsed = time.perf_counter() - t0
        progress.progress(1.0, text="Done!")
        st.success(f"✅ Experiment C completed in {elapsed:.1f}s — ID: `{result['experiment_id']}`")
        st.session_state["exp_c_result"] = result

# ═══ Experiment D ════════════════════════════════════════════════════

with exp_tabs[3]:
    st.markdown("### Experiment D — Binary Classification")
    st.markdown(
        "Train a VQC on the two‑moons dataset and compare with "
        "a classical baseline."
    )

    dc1, dc2 = st.columns(2)
    with dc1:
        d_qubits = st.slider("Qubits", 2, 6, 2, key="d_q")
        d_layers = st.slider("Layers", 1, 6, 2, key="d_l")
        d_ent = st.selectbox("Entanglement", list(ENTANGLEMENT_TYPES), key="d_ent")
        d_init = st.selectbox("Initialisation", list(INIT_STRATEGIES), key="d_init")
    with dc2:
        d_epochs = st.slider("Epochs", 5, 60, 25, 5, key="d_ep")
        d_lr = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01, key="d_lr")
        d_samples = st.slider("Dataset samples", 60, 300, 120, 20, key="d_samp")
        d_runs = st.slider("Independent runs (seeds)", 1, 5, 1, key="d_runs")

    seeds_list = list(range(21, 21 + d_runs))
    total_fw = d_runs * d_epochs * d_qubits * d_layers * 2 * (int(d_samples * 0.75))
    st.caption(f"📋 Estimated forward passes: ~{total_fw:,}  (may take a while for large configs)")

    if st.button("▶ Run Experiment D", key="run_d", type="primary", use_container_width=True):
        from quantumml.experiments.service import ExperimentService

        progress = st.progress(0, text="Starting Experiment D…")

        def _cb(step, total, msg):
            progress.progress(step / total, text=f"[{step}/{total}] {msg}")

        t0 = time.perf_counter()
        result = ExperimentService.run(
            "classification",
            n_qubits=d_qubits, n_layers=d_layers,
            entanglement=d_ent, init_strategy=d_init,
            learning_rate=d_lr, epochs=d_epochs,
            samples=d_samples, seeds=seeds_list,
            progress_callback=_cb,
        )
        elapsed = time.perf_counter() - t0
        progress.progress(1.0, text="Done!")
        st.success(f"✅ Experiment D completed in {elapsed:.1f}s — ID: `{result['experiment_id']}`")
        st.session_state["exp_d_result"] = result
