"""Page 8 — Report Generator.

Generates a structured research report from actual saved experiment
data and verified references.  Sections with missing data are marked
as 'pending'.
"""

from __future__ import annotations

import sys, json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(page_title="Report Generator — QuantumML Lab", page_icon="📝", layout="wide")

from visualization.theme import inject_custom_css, sidebar_config, page_header, COLORS, panel
from quantumml.utils import reference_library_dir, original_experiments_dir
from quantumml import __version__

inject_custom_css()
sidebar_config()
page_header("Report Generator",
            "Generate a structured research report from actual experiment data", "📝")

# ── gather data ──────────────────────────────────────────────────────

ref_path = reference_library_dir() / "references.json"
refs = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else []

grad_dir = original_experiments_dir() / "gradient_results"
train_dir = original_experiments_dir() / "training_results"

grad_experiments = []
train_experiments = []
for d, store in [(grad_dir, grad_experiments), (train_dir, train_experiments)]:
    if d.exists():
        for f in sorted(d.glob("EXP-*.json"), reverse=True):
            try:
                store.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass

has_grad = len(grad_experiments) > 0
has_train = len(train_experiments) > 0

# ── controls ─────────────────────────────────────────────────────────

st.markdown("### Report Configuration")
c1, c2 = st.columns(2)
with c1:
    author = st.text_input("Author(s)", "QuantumML Lab Research Team")
    report_title = st.text_input("Title", "Architecture-Aware Investigation of Barren Plateaus in VQCs")
with c2:
    st.metric("Published References", len(refs))
    st.metric("Gradient Experiments", len(grad_experiments))
    st.metric("Training Experiments", len(train_experiments))

generate = st.button("📄 Generate Report", type="primary", use_container_width=True)

if not generate:
    st.info("Configure the report and click **Generate Report** to create a structured research document from your experiment data.")
    st.stop()

# ── generate report ──────────────────────────────────────────────────

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

report_parts = []

# 1. Title & Abstract
report_parts.append(f"# {report_title}\n")
report_parts.append(f"**Author(s):** {author}  ")
report_parts.append(f"**Date:** {now}  ")
report_parts.append(f"**Software version:** QuantumML Lab v{__version__}\n")
report_parts.append("---\n")

report_parts.append("## Abstract\n")
report_parts.append(
    "This report presents a computational investigation of barren plateaus "
    "in variational quantum circuits (VQCs) for binary classification.  "
    "Using a custom NumPy-based statevector simulator, we study how circuit "
    "depth, entanglement architecture, and parameter initialisation strategy "
    "influence gradient concentration and trainability.  "
    "All experiments are software-only simulations; no quantum hardware was used.\n"
)

# 2. Introduction
report_parts.append("## 1. Introduction\n")
report_parts.append(
    "Variational quantum algorithms are a leading paradigm for near-term "
    "quantum computing.  However, their training landscapes can exhibit "
    "*barren plateaus* — regions where the cost gradient vanishes exponentially "
    "with system size, making gradient-based optimisation impractical.\n"
)

# 3. Literature Review
report_parts.append("## 2. Literature Review\n")
if refs:
    for ref in refs:
        report_parts.append(f"- **{ref.get('title', '')}** ({ref.get('year', '')}), "
                            f"{', '.join(ref.get('authors', [])[:3])} et al.  "
                            f"*{ref.get('venue', '')}*.  "
                            f"DOI: {ref.get('doi', 'N/A')}.  "
                            f"{ref.get('reported_finding', '')}\n")
else:
    report_parts.append("*No published references loaded.*\n")

# 4. Research Gap
report_parts.append("## 3. Research Gap and Motivation\n")
report_parts.append(
    "While theoretical results establish conditions under which barren plateaus arise, "
    "controlled computational experiments comparing multiple architectural choices "
    "(entanglement patterns, initialisation strategies) on the same platform "
    "are valuable for understanding practical trainability.\n"
)

# 5-6. Methodology & Setup
report_parts.append("## 4. Methodology\n")
report_parts.append(
    "- **Simulator:** Custom n-qubit statevector simulator (NumPy).  \n"
    "- **Gradient method:** Parameter-shift rule (exact for Pauli rotations).  \n"
    "- **Ansatz:** RY-RZ layers with configurable entanglement.  \n"
    "- **Cost function:** ⟨Z₀⟩ for gradient analysis; BCE for classification.  \n"
    "- **Data encoding:** Angle encoding RY(πx).  \n"
)

# 7. Results
report_parts.append("## 5. Experimental Results\n")

if has_grad:
    report_parts.append("### 5.1 Gradient Experiments\n")
    for exp in grad_experiments[:3]:
        eid = exp.get("experiment_id", "N/A")
        ename = exp.get("experiment_name", "N/A")
        config = exp.get("configuration", {})
        report_parts.append(f"**{ename}** (`{eid}`)\n")
        report_parts.append(f"- Configuration: {json.dumps(config, default=str)}\n")
        results = exp.get("results", {})
        if isinstance(results, dict):
            for key, val in list(results.items())[:5]:
                if isinstance(val, dict):
                    var = val.get("gradient_variance_per_param_mean", "N/A")
                    frac = val.get("fraction_near_zero", "N/A")
                    report_parts.append(f"  - {key}: variance={var}, near-zero frac={frac}\n")
        report_parts.append("")
else:
    report_parts.append("### 5.1 Gradient Experiments\n")
    report_parts.append("*⏳ PENDING — Run Experiments A, B, or C first.*\n")

if has_train:
    report_parts.append("### 5.2 Classification Experiments\n")
    for exp in train_experiments[:2]:
        eid = exp.get("experiment_id", "N/A")
        config = exp.get("configuration", {})
        results = exp.get("results", {})
        summary = results.get("summary", {})
        report_parts.append(f"**Experiment D** (`{eid}`)\n")
        report_parts.append(f"- Quantum accuracy (mean): {summary.get('quantum_mean_accuracy', 'N/A')}\n")
        report_parts.append(f"- Classical accuracy: {summary.get('classical_accuracy', 'N/A')}\n")
        report_parts.append(f"- Runs: {summary.get('n_runs', 'N/A')}\n")
else:
    report_parts.append("### 5.2 Classification Experiments\n")
    report_parts.append("*⏳ PENDING — Run Experiment D first.*\n")

# 8. Discussion & Limitations
report_parts.append("## 6. Discussion\n")
if has_grad or has_train:
    report_parts.append(
        "The experimental results should be interpreted in the context of "
        "the specific configurations tested.  Gradient concentration patterns "
        "observed here are specific to the ansatz and cost function used.\n"
    )
else:
    report_parts.append("*⏳ PENDING — Run experiments to populate this section.*\n")

report_parts.append("## 7. Limitations\n")
report_parts.append(
    "- Software-only statevector simulation; no hardware noise.  \n"
    "- Limited qubit counts (up to ~10–12 due to classical memory).  \n"
    "- Binary classification on synthetic datasets only.  \n"
    "- The small-gradient threshold is an operational definition.  \n"
    "- No claim of quantum advantage.  \n"
)

# 9. Conclusion
report_parts.append("## 8. Conclusion\n")
report_parts.append(
    "This report presents a controlled investigation of gradient concentration "
    "in variational quantum circuits.  Further experiments with more qubit counts, "
    "deeper circuits, and additional cost functions are needed to strengthen "
    "the conclusions.\n"
)

# 10. References
report_parts.append("## 9. References\n")
for i, ref in enumerate(refs, 1):
    report_parts.append(
        f"[{i}] {', '.join(ref.get('authors', [])[:3])} et al. "
        f"({ref.get('year', '')}).  "
        f"*{ref.get('title', '')}*.  "
        f"{ref.get('venue', '')}.  "
        f"DOI: {ref.get('doi', 'N/A')}.\n"
    )

# ── display ──────────────────────────────────────────────────────────

report_text = "\n".join(report_parts)
st.markdown(report_text)

# ── download ─────────────────────────────────────────────────────────

st.markdown("---")
st.download_button(
    "⬇ Download Report (Markdown)",
    data=report_text,
    file_name=f"quantumml_report_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md",
    mime="text/markdown",
    use_container_width=True,
)
