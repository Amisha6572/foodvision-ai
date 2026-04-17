"""
FoodVision AI – Professional UI v3
Strategy: solid dark content area over blurred background.
All Streamlit native widgets render on #0E0A06 so they are always visible.
"""

import base64, os
import pandas as pd
import requests
import streamlit as st
from PIL import Image

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="FoodVision AI", page_icon="🍽️", layout="wide")

BG_PATH = os.path.join(os.path.dirname(__file__), "food_background.jpg")
with open(BG_PATH, "rb") as f:
    bg_b64 = base64.b64encode(f.read()).decode()

# ─────────────────────────────────────────────────────────────────────────────
#  CSS  –  every selector tested against Streamlit 1.28+
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Base font ── */
html, body, [class*="css"], .stApp, .stMarkdown,
p, span, div, label, li, small, a, button {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

/* ── Background image ── */
.stApp {{
    background-image: url("data:image/jpg;base64,{bg_b64}") !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
}}

/* ── Dark overlay on background only ── */
.stApp > [data-testid="stAppViewContainer"] {{
    background: rgba(8, 4, 1, 0.80) !important;
    backdrop-filter: blur(4px) !important;
    -webkit-backdrop-filter: blur(4px) !important;
}}

/* ── Main content block – solid dark so widgets are always readable ── */
.stApp > [data-testid="stAppViewContainer"] > .main > .block-container {{
    background: transparent !important;
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1180px !important;
}}

/* ── Hide chrome ── */
#MainMenu, footer, header, [data-testid="stDecoration"],
[data-testid="stToolbar"] {{ display: none !important; visibility: hidden !important; }}

/* ════════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════════ */
[data-testid="stSidebar"] {{
    background: #0E0A06 !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}}
[data-testid="stSidebar"] > div {{
    background: #0E0A06 !important;
    padding: 1.8rem 1.4rem !important;
}}
/* All sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small {{
    color: rgba(255,255,255,0.80) !important;
    font-family: 'Inter', sans-serif !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: #FFFFFF !important;
    font-weight: 700 !important;
}}

/* ════════════════════════════════════════════
   NUMBER INPUT  –  the white box fix
════════════════════════════════════════════ */
[data-testid="stNumberInput"] {{
    background: transparent !important;
}}
[data-testid="stNumberInput"] label {{
    color: rgba(255,255,255,0.60) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}}
/* The actual input wrapper */
[data-testid="stNumberInput"] [data-baseweb="input"] {{
    background: #1C1208 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}}
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {{
    border-color: #FF8C42 !important;
    box-shadow: 0 0 0 3px rgba(255,140,66,0.18) !important;
}}
/* The number text */
[data-testid="stNumberInput"] input {{
    background: #1C1208 !important;
    color: #FFFFFF !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    caret-color: #FF8C42 !important;
    border: none !important;
}}
/* +/- stepper buttons */
[data-testid="stNumberInput"] button {{
    background: #2A1A0A !important;
    color: #FF8C42 !important;
    border: none !important;
    border-left: 1px solid rgba(255,255,255,0.10) !important;
}}
[data-testid="stNumberInput"] button:hover {{
    background: #3A2510 !important;
}}

/* ════════════════════════════════════════════
   FILE UPLOADER
════════════════════════════════════════════ */
[data-testid="stFileUploader"] {{
    background: #1C1208 !important;
    border: 1.5px dashed rgba(255,140,66,0.45) !important;
    border-radius: 14px !important;
    padding: 6px !important;
}}
[data-testid="stFileUploader"] * {{
    color: rgba(255,255,255,0.75) !important;
    font-family: 'Inter', sans-serif !important;
}}
[data-testid="stFileUploader"] small {{
    color: rgba(255,255,255,0.35) !important;
}}
/* Browse files button inside uploader */
[data-testid="stFileUploader"] button {{
    background: rgba(255,140,66,0.15) !important;
    color: #FF8C42 !important;
    border: 1px solid rgba(255,140,66,0.4) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}}
[data-testid="stFileUploader"] button:hover {{
    background: rgba(255,140,66,0.25) !important;
}}
/* Uploaded file name row */
[data-testid="stFileUploaderFile"] {{
    background: rgba(255,255,255,0.05) !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
    margin-top: 6px !important;
}}

/* ════════════════════════════════════════════
   PRIMARY BUTTON
════════════════════════════════════════════ */
.stButton > button {{
    background: linear-gradient(135deg, #FF8C42 0%, #E8650A 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.4px !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 18px rgba(232,101,10,0.40) !important;
    transition: all 0.18s ease !important;
    font-family: 'Inter', sans-serif !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(232,101,10,0.55) !important;
}}
.stButton > button:active {{
    transform: translateY(0) !important;
}}

/* ════════════════════════════════════════════
   TABS
════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.10) !important;
    gap: 0 !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: rgba(255,255,255,0.40) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.3px !important;
    padding: 10px 22px !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-family: 'Inter', sans-serif !important;
}}
.stTabs [aria-selected="true"] {{
    color: #FF8C42 !important;
    border-bottom: 2px solid #FF8C42 !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    background: transparent !important;
    padding-top: 18px !important;
}}

/* ════════════════════════════════════════════
   PROGRESS BAR
════════════════════════════════════════════ */
[data-testid="stProgressBar"] > div {{
    background: rgba(255,255,255,0.10) !important;
    border-radius: 100px !important;
    height: 6px !important;
}}
[data-testid="stProgressBar"] > div > div {{
    background: linear-gradient(90deg, #FF8C42, #FFD166) !important;
    border-radius: 100px !important;
}}

/* ════════════════════════════════════════════
   SPINNER / ALERTS
════════════════════════════════════════════ */
[data-testid="stSpinner"] p {{ color: #FF8C42 !important; }}
[data-testid="stAlert"] {{
    background: rgba(239,71,111,0.12) !important;
    border: 1px solid rgba(239,71,111,0.35) !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
}}

/* ════════════════════════════════════════════
   HERO
════════════════════════════════════════════ */
.hero-wrap {{
    text-align: center;
    padding: 2rem 1rem 1.2rem;
}}
.hero-title {{
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 900;
    color: #FFFFFF;
    letter-spacing: -1px;
    line-height: 1.1;
    font-family: 'Inter', sans-serif;
}}
.hero-accent {{
    background: linear-gradient(135deg, #FF8C42, #FFD166);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.hero-sub {{
    font-size: 0.82rem;
    color: rgba(255,255,255,0.45);
    letter-spacing: 3.5px;
    text-transform: uppercase;
    margin-top: 0.7rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
}}
.hero-line {{
    width: 48px;
    height: 3px;
    background: linear-gradient(90deg, #FF8C42, #FFD166);
    border-radius: 2px;
    margin: 1rem auto 0;
}}

/* ════════════════════════════════════════════
   GLASS CARDS
════════════════════════════════════════════ */
.card {{
    background: rgba(14, 8, 2, 0.88);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.50);
}}
.sec-label {{
    font-size: 0.68rem;
    font-weight: 700;
    color: #FF8C42;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 12px;
    font-family: 'Inter', sans-serif;
}}
.food-name {{
    font-size: clamp(1.8rem, 3vw, 2.6rem);
    font-weight: 900;
    color: #FFFFFF;
    line-height: 1.1;
    letter-spacing: -0.5px;
    font-family: 'Inter', sans-serif;
}}
.conf-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,140,66,0.12);
    border: 1px solid rgba(255,140,66,0.35);
    color: #FF8C42;
    border-radius: 100px;
    padding: 5px 16px;
    font-size: 0.82rem;
    font-weight: 700;
    margin-top: 10px;
    letter-spacing: 0.3px;
    font-family: 'Inter', sans-serif;
}}

/* ── Health score ── */
.hs-icon {{ font-size: 2.8rem; line-height: 1; display: block; }}
.hs-label {{
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'Inter', sans-serif;
    display: block;
    margin-top: 6px;
}}
.hs-healthy   {{ color: #06D6A0 !important; }}
.hs-moderate  {{ color: #FFD166 !important; }}
.hs-indulgent {{ color: #EF476F !important; }}

/* ── Pills ── */
.pills-wrap {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.pill {{
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.82) !important;
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.80rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    white-space: nowrap;
}}

/* ── Macro grid ── */
.macro-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}}
.macro-box {{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 18px 10px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}}
.macro-box:hover {{
    border-color: rgba(255,140,66,0.45);
    transform: translateY(-2px);
}}
.macro-val {{
    font-size: 1.9rem;
    font-weight: 900;
    color: #FFFFFF !important;
    line-height: 1;
    font-family: 'Inter', sans-serif;
}}
.macro-unit {{
    font-size: 0.72rem;
    color: #FF8C42 !important;
    font-weight: 600;
    margin-top: 3px;
    font-family: 'Inter', sans-serif;
}}
.macro-lbl {{
    font-size: 0.68rem;
    color: rgba(255,255,255,0.38) !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
    font-family: 'Inter', sans-serif;
}}

/* ── Divider ── */
.hdivider {{
    height: 1px;
    background: rgba(255,255,255,0.07);
    margin: 1.2rem 0;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def macro_card(val, lbl, unit):
    return (
        f'<div class="macro-box">'
        f'<div class="macro-val">{val}</div>'
        f'<div class="macro-unit">{unit}</div>'
        f'<div class="macro-lbl">{lbl}</div>'
        f'</div>'
    )


def render_results(data: dict):
    st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)

    # Row 1 – food name + health score
    c1, c2 = st.columns([3, 1], gap="medium")
    with c1:
        conf = round(data["confidence"] * 100, 1)
        st.markdown(
            f'<div class="card">'
            f'<div class="sec-label">Detected Food</div>'
            f'<div class="food-name">{data["food"]}</div>'
            f'<div class="conf-badge">✦&nbsp; Confidence &nbsp;{conf}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.progress(data["confidence"])

    with c2:
        score = data["health_score"]
        css  = {"Healthy":"hs-healthy","Moderate":"hs-moderate","Indulgent":"hs-indulgent"}.get(score,"hs-moderate")
        icon = {"Healthy":"🥗","Moderate":"⚖️","Indulgent":"🍰"}.get(score,"⚖️")
        st.markdown(
            f'<div class="card" style="height:100%;display:flex;flex-direction:column;'
            f'align-items:center;justify-content:center;text-align:center;gap:4px">'
            f'<div class="sec-label">Health Score</div>'
            f'<span class="hs-icon">{icon}</span>'
            f'<span class="hs-label {css}">{score}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Row 2 – ingredients
    pills = "".join(
        f'<span class="pill">{i.replace("_"," ").title()}</span>'
        for i in data["ingredients"]
    )
    st.markdown(
        f'<div class="card">'
        f'<div class="sec-label">Ingredients</div>'
        f'<div class="pills-wrap">{pills}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Row 3 – nutrition
    base, adj, bw = data["nutrition"], data["adjusted_nutrition"], data["base_weight"]
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Nutrition Facts</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs([f"Base serving  ·  {bw:.0f}g", f"Your portion  ·  {data['portion_size']}g"])

    def nutrition_row(n):
        boxes = "".join([
            macro_card(n["calories"], "Calories", "kcal"),
            macro_card(n["protein"],  "Protein",  "g"),
            macro_card(n["carbs"],    "Carbs",    "g"),
            macro_card(n["fat"],      "Fat",      "g"),
        ])
        st.markdown(f'<div class="macro-grid">{boxes}</div>', unsafe_allow_html=True)

    with tab1:
        nutrition_row(base)
    with tab2:
        nutrition_row(adj)
    st.markdown("</div>", unsafe_allow_html=True)

    # Row 4 – chart
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Macro Comparison (g)</div>', unsafe_allow_html=True)
    df = pd.DataFrame({
        "Macro": ["Protein","Carbs","Fat"],
        f"Base {bw:.0f}g":              [base["protein"],base["carbs"],base["fat"]],
        f"Portion {data['portion_size']}g": [adj["protein"],adj["carbs"],adj["fat"]],
    }).set_index("Macro")
    st.bar_chart(df, color=["#FF8C42","#FFD166"])
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-wrap">'
    '<div class="hero-title">🍽️ Food<span class="hero-accent">Vision</span> AI</div>'
    '<div class="hero-sub">Food Detection &nbsp;·&nbsp; Ingredient Analysis &nbsp;·&nbsp; Nutrition Estimator</div>'
    '<div class="hero-line"></div>'
    '</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    portion_size = st.number_input("Portion size (grams)", min_value=10, max_value=2000, value=200, step=10)
    st.markdown("---")
    if st.button("📋 Supported Foods", width='stretch'):
        try:
            r = requests.get(f"{BACKEND_URL}/foods", timeout=5)
            if r.status_code == 200:
                for food in r.json().get("foods", []):
                    st.markdown(f"• {food.replace('_',' ').title()}")
        except Exception:
            st.warning("Backend not reachable.")
    st.markdown("---")
    st.caption(f"Backend: {BACKEND_URL}")
    st.caption("Upload a food image, set your portion size, and click Analyse Food.")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

col_up, col_prev = st.columns([1, 1], gap="large")

with col_up:
    uploaded_file = st.file_uploader(
        "Upload a food image",
        type=["jpg","jpeg","png","webp"],
    )

with col_prev:
    if uploaded_file:
        st.image(Image.open(uploaded_file), width='stretch')

if uploaded_file:
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    if st.button("🔍  Analyse Food", width='stretch'):
        with st.spinner("Analysing your food..."):
            try:
                uploaded_file.seek(0)
                resp = requests.post(
                    f"{BACKEND_URL}/predict",
                    files={"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)},
                    data={"portion_size": portion_size},
                    timeout=30,
                )
                if resp.status_code == 200:
                    render_results(resp.json())
                else:
                    st.error(resp.json().get("detail", "Prediction failed."))
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach backend at `{BACKEND_URL}`. Is the FastAPI server running?")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
