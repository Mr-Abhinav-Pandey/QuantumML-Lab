"""QuantumML Lab — Landing page and application entry point."""

from __future__ import annotations

import sys
import json
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="QuantumML Lab",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from visualization.theme import inject_custom_css, hero_section, sidebar_config, COLORS, panel
from quantumml.utils import original_experiments_dir, reference_library_dir

inject_custom_css()
sidebar_config()

# ── hero ─────────────────────────────────────────────────────────────

hero_section()

# ── research overview ────────────────────────────────────────────────

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### Research Objective")
    st.markdown(
        "This platform investigates **how quantum circuit architecture, "
        "circuit depth, entanglement connectivity, and parameter "
        "initialisation influence gradient concentration and trainability** "
        "in variational quantum classifiers."
    )
    st.markdown(
        "The central phenomenon under study is the **barren plateau** — "
        "a regime in which the gradient of the cost function becomes "
        "exponentially small with respect to system size, making "
        "optimisation impractical."
    )

    st.markdown("### Scope and Limitations")
    st.info(
        "🔬 This project simulates quantum circuits mathematically on a "
        "classical computer.  It is **not** a physical quantum processor, "
        "does **not** demonstrate quantum advantage, and does **not** "
        "claim to discover a new mitigation algorithm.  It is a "
        "controlled experimental investigation.",
        icon=None,
    )

with col2:
    st.markdown("### Experiment Overview")
    experiments = {
        "A — Depth & Gradient Concentration":
            "Vary qubits (2–8) and depth (1–8) to study gradient scaling.",
        "B — Entanglement Architecture":
            "Compare linear, circular, and all‑to‑all CNOT patterns.",
        "C — Initialisation Strategy":
            "Compare random, small‑angle, and identity‑block methods.",
        "D — Binary Classification":
            "Train VQC on two‑moons and compare with classical baseline.",
    }
    for name, desc in experiments.items():
        st.markdown(
            f'<div class="ql-panel" style="padding:0.9rem 1.1rem; margin-bottom:0.6rem;">'
            f'<span style="color:{COLORS["accent"]}; font-weight:600; font-size:0.9rem;">{name}</span>'
            f'<p style="color:{COLORS["text_muted"]}; font-size:0.82rem; margin:0.25rem 0 0;">{desc}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── research progress ───────────────────────────────────────────────

st.markdown("---")
st.markdown("### Research Progress")

# Count stored experiments
grad_dir = original_experiments_dir() / "gradient_results"
train_dir = original_experiments_dir() / "training_results"
ref_path = reference_library_dir() / "references.json"

n_grad = len(list(grad_dir.glob("EXP-*.json"))) if grad_dir.exists() else 0
n_train = len(list(train_dir.glob("EXP-*.json"))) if train_dir.exists() else 0
n_refs = 0
if ref_path.exists():
    try:
        n_refs = len(json.loads(ref_path.read_text(encoding="utf-8")))
    except Exception:
        pass

c1, c2, c3, c4 = st.columns(4)
c1.metric("Published References", n_refs)
c2.metric("Gradient Experiments", n_grad)
c3.metric("Training Experiments", n_train)
c4.metric("Simulator", "Statevector (n‑qubit)")

# ── data‑separation policy ───────────────────────────────────────────

st.markdown("---")
st.markdown("### Research Data Policy")
p1, p2, p3 = st.columns(3)
with p1:
    panel(
        f'<span style="color:{COLORS["accent"]}; font-weight:600;">📚 Reference Library</span>'
        f'<p class="ql-muted ql-small">Published research only.  Read‑only from the experiment interface.</p>',
        COLORS["accent"],
    )
with p2:
    panel(
        f'<span style="color:{COLORS["highlight"]}; font-weight:600;">🧪 Original Experiments</span>'
        f'<p class="ql-muted ql-small">Project‑generated simulation data.  Never contains published values.</p>',
        COLORS["highlight"],
    )
with p3:
    panel(
        f'<span style="color:{COLORS["accent2"]}; font-weight:600;">📐 Benchmarking</span>'
        f'<p class="ql-muted ql-small">Side‑by‑side comparison.  Qualitative unless conditions match exactly.</p>',
        COLORS["accent2"],
    )

st.caption(
    "QuantumML Lab v2.0 · Software‑only quantum simulation · "
    "All experiment records include unique IDs, timestamps, seeds, and configurations."
)
