"""SVG circuit diagram generator.

Generates crisp, interactive‑looking circuit diagrams as inline SVG
that can be rendered via ``st.markdown(..., unsafe_allow_html=True)``.
"""

from __future__ import annotations

import numpy as np
from visualization.theme import COLORS

# ── layout constants ─────────────────────────────────────────────────

_WIRE_Y0      = 60       # y of first wire
_WIRE_DY      = 60       # vertical spacing between wires
_LABEL_X      = 30       # x of qubit label
_ENC_X0       = 90       # x start of encoding gates
_GATE_W       = 48       # gate box width
_GATE_H       = 36       # gate box height
_LAYER_DX     = 72       # horizontal distance between layers
_ENT_DX       = 40       # extra space for entanglement
_RIGHT_PAD    = 50       # right padding
_FONT         = "font-family:'Inter','Segoe UI',system-ui,sans-serif"


def render_circuit_svg(
    n_qubits: int,
    n_layers: int,
    entanglement: str = "linear",
    params: np.ndarray | None = None,
    n_features: int = 2,
) -> str:
    """Return an SVG string for a variational quantum circuit.

    Parameters
    ----------
    n_qubits, n_layers : int
    entanglement : str
        ``"linear"``, ``"circular"``, or ``"all_to_all"``.
    params : ndarray, optional
        Shape ``(n_layers, n_qubits, 2)``.  If *None*, labels show
        generic θ names instead of numerical values.
    n_features : int
        Number of encoded features (angle‑encoding qubits).
    """
    # Determine how many columns we need
    # Columns:  [encoding] + [RY+RZ+entanglement] * n_layers
    enc_cols = 1  # one column for encoding
    per_layer = 3  # RY col, RZ col, entanglement col
    total_cols = enc_cols + per_layer * n_layers
    width = _LABEL_X + 50 + total_cols * _LAYER_DX + _RIGHT_PAD
    height = _WIRE_Y0 + (n_qubits - 1) * _WIRE_DY + 50

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'style="background:transparent">'
    )
    # Defs for glow filter
    parts.append("""<defs>
        <filter id="glow" x="-20%%" y="-20%%" width="140%%" height="140%%">
            <feGaussianBlur stdDeviation="2" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
    </defs>""")

    # ── qubit wires ──────────────────────────────────────────────
    for q in range(n_qubits):
        y = _wire_y(q)
        parts.append(
            f'<line x1="{_LABEL_X + 20}" y1="{y}" x2="{width - 20}" y2="{y}" '
            f'stroke="{COLORS["text_muted"]}" stroke-width="1.2" stroke-opacity="0.5"/>'
        )
        # label
        parts.append(
            f'<text x="{_LABEL_X}" y="{y + 5}" text-anchor="middle" '
            f'fill="{COLORS["text"]}" font-size="13" {_FONT}>|q{_sub(q)}⟩</text>'
        )

    # ── encoding column ──────────────────────────────────────────
    cx = _ENC_X0
    for q in range(min(n_features, n_qubits)):
        _gate_box(parts, cx, _wire_y(q), "RY", f"πx{_sub(q)}", COLORS["highlight"])
    # label
    parts.append(
        f'<text x="{cx + _GATE_W // 2}" y="{_WIRE_Y0 - 24}" text-anchor="middle" '
        f'fill="{COLORS["text_muted"]}" font-size="10" {_FONT}>Encoding</text>'
    )

    # ── variational layers ───────────────────────────────────────
    lx = _ENC_X0 + _LAYER_DX  # start after encoding
    for l in range(n_layers):
        # layer label
        mid_x = lx + _LAYER_DX
        parts.append(
            f'<text x="{mid_x}" y="{_WIRE_Y0 - 24}" text-anchor="middle" '
            f'fill="{COLORS["text_muted"]}" font-size="10" {_FONT}>Layer {l + 1}</text>'
        )

        # RY gates
        for q in range(n_qubits):
            lbl = _param_label(params, l, q, 0, "θ")
            _gate_box(parts, lx, _wire_y(q), "RY", lbl, COLORS["accent"])

        # RZ gates
        rz_x = lx + _LAYER_DX
        for q in range(n_qubits):
            lbl = _param_label(params, l, q, 1, "φ")
            _gate_box(parts, rz_x, _wire_y(q), "RZ", lbl, COLORS["accent2"])

        # Entanglement
        ent_x = rz_x + _LAYER_DX
        _draw_entanglement(parts, ent_x, n_qubits, entanglement)

        lx = ent_x + _LAYER_DX

    # ── measurement symbols ──────────────────────────────────────
    mx = width - 40
    for q in range(n_qubits):
        y = _wire_y(q)
        # meter icon (arc + arrow)
        parts.append(
            f'<circle cx="{mx}" cy="{y}" r="12" fill="none" '
            f'stroke="{COLORS["text_muted"]}" stroke-width="1.2"/>'
        )
        parts.append(
            f'<path d="M{mx-6} {y+4} A 8 8 0 0 1 {mx+6} {y+4}" '
            f'fill="none" stroke="{COLORS["text_muted"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{mx}" y1="{y+4}" x2="{mx+5}" y2="{y-6}" '
            f'stroke="{COLORS["text_muted"]}" stroke-width="1.2"/>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ── private helpers ──────────────────────────────────────────────────

def _wire_y(q: int) -> int:
    return _WIRE_Y0 + q * _WIRE_DY


def _sub(n: int) -> str:
    """Unicode subscript digits."""
    subs = "₀₁₂₃₄₅₆₇₈₉"
    return "".join(subs[int(d)] for d in str(n))


def _param_label(
    params: np.ndarray | None, layer: int, qubit: int, idx: int, sym: str
) -> str:
    if params is not None:
        try:
            val = params[layer, qubit, idx]
            return f"{val:.2f}"
        except (IndexError, TypeError):
            pass
    return f"{sym}{_sub(layer)},{_sub(qubit)}"


def _gate_box(
    parts: list[str], x: int, y: int, name: str, label: str, color: str
) -> None:
    """Draw a gate rectangle with name and parameter label."""
    rx, ry_ = x, y - _GATE_H // 2
    parts.append(
        f'<rect x="{rx}" y="{ry_}" width="{_GATE_W}" height="{_GATE_H}" '
        f'rx="6" fill="{COLORS["panel"]}" stroke="{color}" stroke-width="1.4"/>'
    )
    parts.append(
        f'<text x="{rx + _GATE_W // 2}" y="{y - 2}" text-anchor="middle" '
        f'fill="{color}" font-size="12" font-weight="600" {_FONT}>{name}</text>'
    )
    parts.append(
        f'<text x="{rx + _GATE_W // 2}" y="{y + 12}" text-anchor="middle" '
        f'fill="{COLORS["text_muted"]}" font-size="9" {_FONT}>{label}</text>'
    )


def _draw_entanglement(
    parts: list[str], x: int, n_qubits: int, pattern: str
) -> None:
    """Draw the CNOT connections for one entanglement layer."""
    if n_qubits < 2:
        return

    pairs: list[tuple[int, int]] = []
    if pattern == "linear":
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    elif pattern == "circular":
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        pairs.append((n_qubits - 1, 0))
    elif pattern == "all_to_all":
        pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

    cx = x + 20
    for ctrl, tgt in pairs:
        yc = _wire_y(ctrl)
        yt = _wire_y(tgt)
        # vertical line
        parts.append(
            f'<line x1="{cx}" y1="{yc}" x2="{cx}" y2="{yt}" '
            f'stroke="{COLORS["accent"]}" stroke-width="1.4" stroke-opacity="0.7"/>'
        )
        # control dot
        parts.append(
            f'<circle cx="{cx}" cy="{yc}" r="4" fill="{COLORS["accent"]}"/>'
        )
        # target ⊕
        parts.append(
            f'<circle cx="{cx}" cy="{yt}" r="8" fill="none" '
            f'stroke="{COLORS["accent"]}" stroke-width="1.4"/>'
        )
        parts.append(
            f'<line x1="{cx-5}" y1="{yt}" x2="{cx+5}" y2="{yt}" '
            f'stroke="{COLORS["accent"]}" stroke-width="1.2"/>'
        )
        parts.append(
            f'<line x1="{cx}" y1="{yt-5}" x2="{cx}" y2="{yt+5}" '
            f'stroke="{COLORS["accent"]}" stroke-width="1.2"/>'
        )
        cx += 18  # offset subsequent pairs horizontally
