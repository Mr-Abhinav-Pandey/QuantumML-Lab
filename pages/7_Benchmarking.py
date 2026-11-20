"""Page 7 — Research Benchmarking.

Side‑by‑side comparison of published findings with original
experimental results.
"""

from __future__ import annotations

import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(page_title="Benchmarking — QuantumML Lab", page_icon="📐", layout="wide")

from visualization.theme import inject_custom_css, sidebar_config, page_header, COLORS, panel, empty_state
from quantumml.utils import reference_library_dir, original_experiments_dir

inject_custom_css()
sidebar_config()
page_header("Research Benchmarking",
            "Compare published findings with our original experimental results", "📐")

st.info(
    "🔬 Source classifications: **PUBLISHED_REFERENCE** (from literature), "
    "**ORIGINAL_EXPERIMENT** (our simulations), **REPRODUCED_REFERENCE_RESULT** "
    "(only after methodology is matched and executed).",
    icon=None,
)

# ── load data ────────────────────────────────────────────────────────

ref_path = reference_library_dir() / "references.json"
refs = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else []

grad_dir = original_experiments_dir() / "gradient_results"
train_dir = original_experiments_dir() / "training_results"
our_experiments = []
for d in [grad_dir, train_dir]:
    if d.exists():
        for f in sorted(d.glob("EXP-*.json"), reverse=True):
            try:
                our_experiments.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass

# ── comparison layout ────────────────────────────────────────────────

for ref in refs:
    with st.expander(f"📚 {ref.get('title', '')} ({ref.get('year', '')})"):
        col_pub, col_our = st.columns(2)

        with col_pub:
            st.markdown(
                f'<div class="ql-panel" style="border-left:3px solid {COLORS["accent"]};">'
                f'<span style="color:{COLORS["accent"]}; font-weight:700; font-size:0.85rem;">'
                f'PUBLISHED_REFERENCE</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Authors:** {', '.join(ref.get('authors', []))}")
            st.markdown(f"**Venue:** {ref.get('venue', '')}")
            st.markdown(f"**DOI:** [{ref.get('doi', '')}](https://doi.org/{ref.get('doi', '')})")
            st.markdown(f"**Study focus:** {ref.get('study_focus', 'N/A')}")
            st.markdown(f"**Reported finding:** {ref.get('reported_finding', 'N/A')}")
            st.markdown(f"**Limitations:** {ref.get('limitations_for_comparison', 'N/A')}")

        with col_our:
            st.markdown(
                f'<div class="ql-panel" style="border-left:3px solid {COLORS["highlight"]};">'
                f'<span style="color:{COLORS["highlight"]}; font-weight:700; font-size:0.85rem;">'
                f'ORIGINAL_EXPERIMENT</span></div>',
                unsafe_allow_html=True,
            )
            if our_experiments:
                latest = our_experiments[0]
                st.markdown(f"**Experiment ID:** `{latest.get('experiment_id', 'N/A')}`")
                st.markdown(f"**Name:** {latest.get('experiment_name', 'N/A')}")
                config = latest.get("configuration", {})
                st.markdown(f"**Qubits:** {config.get('qubit_counts', config.get('n_qubits', 'N/A'))}")
                st.markdown(f"**Depth:** {config.get('depth_counts', config.get('n_layers', 'N/A'))}")
                st.markdown(f"**Entanglement:** {config.get('entanglement', config.get('entanglements', 'N/A'))}")
                st.markdown(f"**Init:** {config.get('init_strategy', config.get('strategies', 'N/A'))}")
                st.markdown(f"**Seeds:** {config.get('n_seeds', 'N/A')}")
            else:
                st.markdown(f"<p style='color:{COLORS['text_muted']}'>No experiments run yet.  "
                            f"Run experiments from the Experiments page.</p>",
                            unsafe_allow_html=True)

        # ── condition matching ────────────────────────────────────
        st.markdown("---")
        st.markdown("**Condition Matching Assessment**")

        conditions = [
            "Number of qubits",
            "Circuit depth",
            "Circuit architecture / ansatz",
            "Entanglement pattern",
            "Cost function",
            "Initialisation",
            "Gradient estimation method",
            "Number of samples / repetitions",
            "Simulator or hardware conditions",
        ]

        st.markdown(
            f'<div class="ql-panel">'
            f'<p style="color:{COLORS["warning"]}; font-weight:600;">⚠ Comparison type: QUALITATIVE</p>'
            f'<p class="ql-muted ql-small">'
            f'The experimental configurations differ from the published study.  '
            f'Direct numerical reproduction requires matching all conditions listed below.</p>'
            f'<ul style="color:{COLORS["text_muted"]}; font-size:0.82rem;">',
            unsafe_allow_html=True,
        )
        for c in conditions:
            st.markdown(f"- {c}: *not yet matched*")
        st.markdown("</ul></div>", unsafe_allow_html=True)

# ── comparison template ──────────────────────────────────────────────

st.markdown("---")
st.markdown("### Reproduction Status")
csv_path = reference_library_dir() / "published_findings" / "reference_comparison_template.csv"
if csv_path.exists():
    import pandas as pd
    df = pd.read_csv(csv_path)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.caption("Comparison template not found.")

st.caption(
    "A result may only be labelled REPRODUCED_REFERENCE_RESULT after the "
    "corresponding methodology has been implemented, executed, and validated."
)
