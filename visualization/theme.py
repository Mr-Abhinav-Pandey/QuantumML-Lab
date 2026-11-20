"""Design system — colours, typography, global CSS, reusable components.

Call ``inject_custom_css()`` at the top of every page to apply the
dark‑theme design system.
"""

from __future__ import annotations
import streamlit as st

# ── colour palette ───────────────────────────────────────────────────

COLORS = {
    "bg":           "#080D18",
    "bg_secondary": "#101827",
    "panel":        "#131F30",
    "panel_border": "#1e3a5f",
    "accent":       "#39C6F4",
    "accent2":      "#8175FF",
    "highlight":    "#64E4C5",
    "text":         "#EAF2FF",
    "text_muted":   "#94A3B8",
    "success":      "#34D399",
    "warning":      "#FBBF24",
    "error":        "#F87171",
    "grid":         "rgba(57,198,244,0.07)",
}

# ── master CSS ───────────────────────────────────────────────────────

_CSS = """
<style>
/* ─ Root ─ */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: %(bg)s;
    color: %(text)s;
}
html, body { background-color: %(bg)s; }

/* ─ Sidebar ─ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, %(bg_secondary)s 0%%, #0b1120 100%%);
    border-right: 1px solid %(panel_border)s;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption p {
    color: %(text_muted)s;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: %(text)s;
}

/* ─ Header bar ─ */
header[data-testid="stHeader"] {
    background: transparent;
}

/* ─ Typography ─ */
.stApp h1 { color: %(text)s; font-weight: 700; letter-spacing: -0.02em; }
.stApp h2 { color: %(text)s; font-weight: 600; }
.stApp h3 { color: %(accent)s; font-weight: 600; }
.stApp p, .stApp li, .stApp span, .stApp label { color: %(text)s; }
.stApp .stCaption p { color: %(text_muted)s; }

/* ─ Metrics ─ */
[data-testid="stMetric"] {
    background: %(panel)s;
    border: 1px solid %(panel_border)s;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] { color: %(accent)s; font-weight: 700; }
[data-testid="stMetricLabel"] p { color: %(text_muted)s !important; font-size: .85rem; }

/* ─ Tabs ─ */
button[data-baseweb="tab"] {
    color: %(text_muted)s;
    background: transparent;
    border-radius: 8px 8px 0 0;
    font-weight: 500;
}
button[data-baseweb="tab"]:hover { color: %(accent)s; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: %(accent)s;
    border-bottom: 2px solid %(accent)s;
    background: rgba(57,198,244,0.06);
}
[data-baseweb="tab-panel"] { background: transparent; }
[data-baseweb="tab-highlight"] { background-color: %(accent)s !important; }
[data-baseweb="tab-border"] { background-color: %(panel_border)s !important; }

/* ─ Buttons ─ */
.stButton > button {
    background: linear-gradient(135deg, %(accent)s 0%%, #2ea8d8 100%%);
    color: %(bg)s;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(57,198,244,0.3);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button[kind="secondary"],
.stButton > button[data-testid="stFormSubmitButton"] {
    background: transparent;
    border: 1px solid %(panel_border)s;
    color: %(text)s;
}

/* ─ Inputs / Selects / Sliders ─ */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: %(panel)s;
    color: %(text)s;
    border: 1px solid %(panel_border)s;
    border-radius: 8px;
}
[data-baseweb="select"] > div {
    background-color: %(panel)s !important;
    border-color: %(panel_border)s !important;
    color: %(text)s;
}
[data-baseweb="popover"] > div {
    background-color: %(panel)s !important;
    border: 1px solid %(panel_border)s;
}
[data-baseweb="menu"] {
    background-color: %(panel)s !important;
}
[data-baseweb="menu"] li {
    color: %(text)s;
}
[data-baseweb="menu"] li:hover {
    background-color: rgba(57,198,244,0.1) !important;
}
div[data-baseweb="slider"] > div { background-color: %(panel_border)s; }
div[data-baseweb="slider"] > div > div { background-color: %(accent)s !important; }
div[data-baseweb="slider"] [role="slider"] {
    background-color: %(accent)s !important;
    border-color: %(accent)s !important;
}

/* ─ Expanders ─ */
[data-testid="stExpander"] {
    background-color: %(panel)s;
    border: 1px solid %(panel_border)s;
    border-radius: 10px;
}
[data-testid="stExpander"] summary p { color: %(text)s; font-weight: 500; }

/* ─ Code ─ */
.stCodeBlock, .stCode, pre {
    background-color: #0a1120 !important;
    border: 1px solid %(panel_border)s;
    border-radius: 8px;
}
code { color: %(accent)s; }

/* ─ Dataframes ─ */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ─ Alerts ─ */
[data-testid="stAlert"] {
    border-radius: 10px;
    border-left-width: 4px;
}

/* ─ Dividers ─ */
hr { border-color: %(panel_border)s; opacity: 0.4; }

/* ─ Scrollbar ─ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: %(bg_secondary)s; }
::-webkit-scrollbar-thumb { background: %(panel_border)s; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: %(accent)s; }

/* ─ Multiselect tags ─ */
[data-baseweb="tag"] {
    background-color: rgba(57,198,244,0.15) !important;
    color: %(accent)s !important;
}

/* ─ Radio / Checkbox ─ */
.stRadio label, .stCheckbox label { color: %(text)s !important; }

/* ─ Progress bar ─ */
.stProgress > div > div { background-color: %(accent)s; }

/* ─ Custom utility classes ─ */
.ql-panel {
    background: %(panel)s;
    border: 1px solid %(panel_border)s;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.ql-accent { color: %(accent)s; }
.ql-muted  { color: %(text_muted)s; }
.ql-small  { font-size: 0.85rem; }
.ql-mono   { font-family: 'JetBrains Mono', 'Fira Code', monospace; }

/* ─ Sidebar navigation ─ */
[data-testid="stSidebarNav"] a span {
    color: %(text_muted)s !important;
    font-weight: 500;
}
[data-testid="stSidebarNav"] a:hover span {
    color: %(accent)s !important;
}
[data-testid="stSidebarNav"] li[class*="active"] a span {
    color: %(accent)s !important;
    font-weight: 600;
}

/* ─ Plotly modebar ─ */
.modebar { background: transparent !important; }
.modebar-btn path { fill: %(text_muted)s !important; }
.modebar-btn:hover path { fill: %(accent)s !important; }
</style>
""" % COLORS


def inject_custom_css() -> None:
    """Inject the global dark‑theme CSS.  Call once per page."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ── reusable components ──────────────────────────────────────────────

def page_header(title: str, description: str = "", icon: str = "⚛️") -> None:
    """Render a consistent page header with optional description."""
    st.markdown(
        f"<h1 style='margin-bottom:0.2rem'>{icon} {title}</h1>",
        unsafe_allow_html=True,
    )
    if description:
        st.caption(description)
    st.markdown("<hr style='margin:0.5rem 0 1.5rem 0'>", unsafe_allow_html=True)


def panel(content_md: str, accent_color: str | None = None) -> None:
    """Render content inside a styled panel."""
    border = f"border-left: 3px solid {accent_color};" if accent_color else ""
    st.markdown(
        f'<div class="ql-panel" style="{border}">{content_md}</div>',
        unsafe_allow_html=True,
    )


def metric_row(metrics: dict[str, str | float], cols: int = 4) -> None:
    """Display up to *cols* metrics in a single row."""
    columns = st.columns(min(cols, len(metrics)))
    for col, (label, value) in zip(columns, metrics.items()):
        col.metric(label, value)


def empty_state(message: str, experiment: str = "") -> None:
    """Placeholder shown when results are not yet available."""
    st.markdown(
        f"""
        <div style="text-align:center; padding:3rem 1rem;
                    color:{COLORS['text_muted']}; border:1px dashed {COLORS['panel_border']};
                    border-radius:12px; margin:1rem 0;">
            <p style="font-size:1.3rem; margin-bottom:0.5rem;">📊 No data yet</p>
            <p>{message}</p>
            {"<p style='font-size:0.85rem; margin-top:0.5rem;'>Run <b>" + experiment + "</b> from the Experiments page.</p>" if experiment else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_config() -> None:
    """Standard sidebar branding applied to every page."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align:center; padding:0.5rem 0 0.8rem 0;">
                <span style="font-size:1.6rem; font-weight:800;
                             letter-spacing:0.05em; color:{COLORS['accent']};">
                    ⚛ QUANTUMLAB
                </span>
                <p style="font-size:0.75rem; color:{COLORS['text_muted']};
                          margin-top:0.2rem;">
                    Barren Plateau Research Platform
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()


# ── hero section (landing page) ──────────────────────────────────────

def hero_section() -> None:
    """Render the animated landing page hero."""
    st.markdown(_HERO_HTML, unsafe_allow_html=True)


_HERO_HTML = """
<style>
.hero-wrap{
    position:relative; overflow:hidden;
    border-radius:16px; padding:60px 40px 50px;
    background:linear-gradient(135deg,#080D18 0%%,#0a1428 50%%,#0d1a35 100%%);
    margin-bottom:2rem; text-align:center;
}
.hero-wrap::before{
    content:''; position:absolute; inset:0;
    background-image:radial-gradient(circle at 1px 1px,
        rgba(57,198,244,0.12) 1px, transparent 0);
    background-size:36px 36px;
    animation:gridPulse 5s ease-in-out infinite;
    pointer-events:none;
}
@keyframes gridPulse{0%%,100%%{opacity:.25}50%%{opacity:.55}}
.hero-orb{
    position:absolute; border-radius:50%%; filter:blur(80px);
    pointer-events:none; animation:orbFloat 10s ease-in-out infinite;
}
.orb-a{width:340px;height:340px;background:rgba(57,198,244,.07);top:-80px;left:8%%;animation-delay:0s}
.orb-b{width:260px;height:260px;background:rgba(129,117,255,.06);bottom:-60px;right:12%%;animation-delay:4s}
.orb-c{width:200px;height:200px;background:rgba(100,228,197,.05);top:40%%;left:55%%;animation-delay:7s}
@keyframes orbFloat{0%%,100%%{transform:translate(0,0)}50%%{transform:translate(18px,-16px)}}
.hero-title{
    font-size:3.6rem; font-weight:800; letter-spacing:.08em;
    background:linear-gradient(135deg,#39C6F4 0%%,#8175FF 50%%,#64E4C5 100%%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:.3rem; position:relative;
}
.hero-sub{
    font-size:1.08rem; color:#94A3B8; max-width:720px;
    margin:0 auto .6rem; line-height:1.6; position:relative;
}
.hero-badge{
    display:inline-block; padding:4px 14px; border-radius:20px;
    font-size:.75rem; font-weight:600; letter-spacing:.04em;
    border:1px solid rgba(57,198,244,.3); color:#39C6F4;
    margin-bottom:1.2rem; position:relative;
}
.hero-cards{
    display:flex; gap:16px; justify-content:center;
    flex-wrap:wrap; margin-top:2rem; position:relative;
}
.hero-card{
    background:rgba(19,31,48,.7); border:1px solid #1e3a5f;
    border-radius:12px; padding:20px 24px; width:200px;
    text-align:left; transition:all .25s;
}
.hero-card:hover{border-color:#39C6F4;transform:translateY(-3px);
    box-shadow:0 8px 30px rgba(57,198,244,.1)}
.hero-card h4{color:#EAF2FF;font-size:.95rem;margin:0 0 6px}
.hero-card p{color:#94A3B8;font-size:.78rem;margin:0;line-height:1.4}
.hero-card .ic{font-size:1.5rem;margin-bottom:8px;display:block}
</style>

<div class="hero-wrap">
    <div class="hero-orb orb-a"></div>
    <div class="hero-orb orb-b"></div>
    <div class="hero-orb orb-c"></div>

    <span class="hero-badge">SOFTWARE‑ONLY QUANTUM SIMULATION</span>
    <div class="hero-title">QUANTUMLAB</div>
    <p class="hero-sub">
        Architecture‑Aware Investigation of Barren Plateaus<br>
        in Variational Quantum Circuits for Binary Classification
    </p>
    <p class="hero-sub" style="font-size:.88rem;max-width:600px;margin-top:.4rem;">
        How do circuit depth, entanglement connectivity, and parameter
        initialisation influence gradient concentration in variational
        quantum classifiers?
    </p>

    <div class="hero-cards">
        <div class="hero-card">
            <span class="ic">🔬</span>
            <h4>Circuit Lab</h4>
            <p>Build and inspect variational quantum circuits interactively.</p>
        </div>
        <div class="hero-card">
            <span class="ic">📐</span>
            <h4>Experiments</h4>
            <p>Run gradient‑analysis and classification experiments.</p>
        </div>
        <div class="hero-card">
            <span class="ic">📊</span>
            <h4>Results</h4>
            <p>Analyse gradient distributions and barren‑plateau indicators.</p>
        </div>
        <div class="hero-card">
            <span class="ic">📚</span>
            <h4>References</h4>
            <p>Curated published research on barren plateaus in VQCs.</p>
        </div>
    </div>
</div>
"""
