"""Page 6 — Results and Analysis.

Visualises gradient distributions, variance scaling, heatmaps,
architecture comparisons, initialisation comparisons, training
curves, and classification metrics from stored experiment data.
"""

from __future__ import annotations

import sys, json, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import streamlit as st

st.set_page_config(page_title="Results & Analysis — QuantumML Lab", page_icon="📊", layout="wide")

from visualization.theme import inject_custom_css, sidebar_config, page_header, COLORS, empty_state
from visualization.plots import (
    gradient_distribution, gradient_variance_plot, gradient_heatmap,
    comparison_bars, training_curves, classification_scatter,
    near_zero_fraction_plot,
)
from quantumml.utils import original_experiments_dir

inject_custom_css()
sidebar_config()
page_header("Results & Analysis",
            "Visualise experiment results from stored data", "📊")


# ── helpers ──────────────────────────────────────────────────────────

def _load_latest(sub_dir: str, prefix: str = ""):
    """Load the most recent experiment JSON from a subdirectory."""
    base = original_experiments_dir() / sub_dir
    files = sorted(base.glob("EXP-*.json"), reverse=True)
    if prefix:
        files = [f for f in files if prefix in f.stem]
    return json.loads(files[0].read_text(encoding="utf-8")) if files else None


def _get_result(key: str, sub_dir: str, exp_name: str):
    """Return session result or latest file."""
    if key in st.session_state and st.session_state[key] is not None:
        return st.session_state[key]
    latest = _load_latest(sub_dir)
    if latest:
        return latest.get("results", latest)
    return None


# ── tabs ─────────────────────────────────────────────────────────────

tabs = st.tabs([
    "A — Depth & Gradients",
    "B — Entanglement",
    "C — Initialisation",
    "D — Classification",
])

# ═══ Results A ═══════════════════════════════════════════════════════

with tabs[0]:
    data = _get_result("exp_a_result", "gradient_results", "A")
    if data is None:
        empty_state("No depth‑gradient experiment results available.",
                    "Experiment A")
    else:
        results = data.get("results", data)
        raw = data.get("raw_gradients", {})

        # Determine qubit and depth counts from keys
        keys = list(results.keys())
        qubit_counts = sorted(set(int(k.split("q")[0]) for k in keys))
        depth_counts = sorted(set(int(k.split("_")[1].replace("L", "")) for k in keys))

        st.markdown("#### Gradient Variance vs Circuit Depth")
        fig_var = gradient_variance_plot(results, qubit_counts, depth_counts)
        st.plotly_chart(fig_var, use_container_width=True, key="a_var")

        st.markdown("#### Gradient Variance Heatmap")
        fig_heat = gradient_heatmap(results, qubit_counts, depth_counts)
        st.plotly_chart(fig_heat, use_container_width=True, key="a_heat")

        st.markdown("#### Gradient Distribution (select configuration)")
        sel_key = st.selectbox("Configuration", keys, key="a_sel")
        if sel_key in raw:
            grads = np.array(raw[sel_key])
            fig_dist = gradient_distribution(grads, title=f"Gradient Distribution — {sel_key}")
            st.plotly_chart(fig_dist, use_container_width=True, key="a_dist")

        # Near-zero fraction
        st.markdown("#### Near‑Zero Gradient Fraction")
        threshold = st.number_input("Threshold", value=1e-3, format="%.1e", key="a_thresh")
        cats, fracs = [], []
        for k in keys:
            frac = results[k].get("fraction_near_zero", 0)
            cats.append(k)
            fracs.append(frac)
        fig_nz = near_zero_fraction_plot(cats, fracs, threshold)
        st.plotly_chart(fig_nz, use_container_width=True, key="a_nz")

# ═══ Results B ═══════════════════════════════════════════════════════

with tabs[1]:
    data = _get_result("exp_b_result", "gradient_results", "B")
    if data is None:
        empty_state("No entanglement experiment results available.",
                    "Experiment B")
    else:
        results = data.get("results", data)
        ents = list(results.keys())

        st.markdown("#### Mean Gradient Variance by Architecture")
        vals = [results[e].get("gradient_variance_per_param_mean", 0) for e in ents]
        fig_bars = comparison_bars(ents, vals, title="Gradient Variance by Entanglement",
                                   y_label="Mean Gradient Variance", log_y=True)
        st.plotly_chart(fig_bars, use_container_width=True, key="b_bars")

        st.markdown("#### Mean |∂C/∂θ| by Architecture")
        abs_vals = [results[e].get("mean_abs_gradient", 0) for e in ents]
        fig_abs = comparison_bars(ents, abs_vals, title="Mean |Gradient| by Entanglement",
                                  y_label="Mean |∂C/∂θ|", log_y=True)
        st.plotly_chart(fig_abs, use_container_width=True, key="b_abs")

        st.markdown("#### Near‑Zero Fraction")
        fracs = [results[e].get("fraction_near_zero", 0) for e in ents]
        threshold = results[ents[0]].get("small_gradient_threshold", 1e-3)
        fig_nz = near_zero_fraction_plot(ents, fracs, threshold)
        st.plotly_chart(fig_nz, use_container_width=True, key="b_nz")

# ═══ Results C ═══════════════════════════════════════════════════════

with tabs[2]:
    data = _get_result("exp_c_result", "gradient_results", "C")
    if data is None:
        empty_state("No initialisation experiment results available.",
                    "Experiment C")
    else:
        results = data.get("results", data)
        strats = list(results.keys())

        st.markdown("#### Mean Gradient Variance by Initialisation")
        vals = [results[s].get("gradient_variance_per_param_mean", 0) for s in strats]
        fig_b = comparison_bars(strats, vals, title="Gradient Variance by Initialisation",
                                y_label="Mean Gradient Variance", log_y=True)
        st.plotly_chart(fig_b, use_container_width=True, key="c_bars")

        st.markdown("#### Mean |∂C/∂θ| by Initialisation")
        abs_vals = [results[s].get("mean_abs_gradient", 0) for s in strats]
        fig_a = comparison_bars(strats, abs_vals, title="Mean |Gradient| by Initialisation",
                                y_label="Mean |∂C/∂θ|", log_y=True)
        st.plotly_chart(fig_a, use_container_width=True, key="c_abs")

# ═══ Results D ═══════════════════════════════════════════════════════

with tabs[3]:
    data = _get_result("exp_d_result", "training_results", "D")
    if data is None:
        empty_state("No classification experiment results available.",
                    "Experiment D")
    else:
        results = data if isinstance(data, dict) else {}

        qruns = results.get("quantum_runs", [])
        classical = results.get("classical", {})
        summary = results.get("summary", {})

        if qruns:
            st.markdown("#### Training Convergence")
            for i, run in enumerate(qruns):
                if run.get("history"):
                    fig_tc = training_curves(run["history"],
                                              title=f"VQC Training — Seed {run.get('seed', i)}")
                    st.plotly_chart(fig_tc, use_container_width=True, key=f"d_tc_{i}")

            st.markdown("#### Classification Performance")
            mc1, mc2, mc3, mc4 = st.columns(4)
            q_acc = summary.get("quantum_mean_accuracy", 0)
            c_acc = summary.get("classical_accuracy", 0)
            mc1.metric("Quantum Accuracy (mean)", f"{q_acc:.1%}")
            mc2.metric("Quantum Std", f"{summary.get('quantum_std_accuracy', 0):.4f}")
            mc3.metric("Classical Accuracy", f"{c_acc:.1%}")
            mc4.metric("Runs", summary.get("n_runs", 0))

            # Comparison bar
            fig_cmp = comparison_bars(
                ["VQC (mean)", "Classical"],
                [q_acc, c_acc],
                title="Test Accuracy Comparison",
                y_label="Accuracy",
            )
            fig_cmp.update_yaxes(range=[0, 1.05])
            st.plotly_chart(fig_cmp, use_container_width=True, key="d_cmp")

            st.caption(
                "⚠️ The classical model is a baseline — not a proof that "
                "VQC is better or worse.  Compare across multiple seeds "
                "and configurations before drawing conclusions."
            )
