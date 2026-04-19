import datetime
import pathlib

import streamlit as st

from deception_mapper import DeceptionMapper
from inference import EmotionInference
from pages.about import render_about_page
from pages.analytics import render_analytics_page
from pages.image_analyzer import render_image_analyzer_page
from pages.live_detector import render_live_detector_page
from pages.training import render_training_page


st.set_page_config(
    page_title="EmotionLens",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .stApp {
            background:
              radial-gradient(1200px 600px at 10% -20%, rgba(0,200,83,0.12), transparent 60%),
              radial-gradient(1200px 600px at 100% 0%, rgba(0,229,255,0.10), transparent 60%),
              #0B0F17;
        }

        [data-testid="stSidebar"] {
            background: #0F1117;
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        .metric-card {
            background: #1E1E2E;
            border-radius: 14px;
            padding: 0.75rem 1rem;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .metric-value {
            color: #00E5FF;
            font-size: 1.8rem;
            font-weight: 700;
        }

        .emotion-badge {
            display: inline-block;
            color: #fff;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.92rem;
        }

        .truthlie-wrap {
            background: #1E1E2E;
            padding: 1rem;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .truthlie-bar {
            height: 30px;
            border-radius: 999px;
            overflow: hidden;
            display: flex;
            background: #111;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .truth-segment {
            background: #00C853;
            transition: width 0.3s ease;
            height: 100%;
        }

        .lie-segment {
            background: #D50000;
            transition: width 0.3s ease;
            height: 100%;
        }

        .quality-good { color: #00C853; }
        .quality-mid { color: #FF6D00; }
        .quality-bad { color: #D50000; }

        .stButton > button {
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            background: linear-gradient(180deg, #161b25, #111722) !important;
            color: #E8ECF2 !important;
            font-weight: 600 !important;
        }

        .stButton > button:hover {
            border-color: rgba(0,229,255,0.4) !important;
            box-shadow: 0 0 0 2px rgba(0,229,255,0.14) inset !important;
        }

        [data-testid="stMetric"] {
            background: rgba(30,30,46,0.75);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 8px 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_inference():
    try:
        return EmotionInference('models/best_model.pt')
    except FileNotFoundError:
        return None


def initialize_session_state() -> None:
    if 'inference' not in st.session_state:
        st.session_state.inference = None
    if 'mapper' not in st.session_state:
        st.session_state.mapper = DeceptionMapper()
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'low_quality_streak' not in st.session_state:
        st.session_state.low_quality_streak = 0
    if 'session_start' not in st.session_state:
        st.session_state.session_start = datetime.datetime.now()


def render_sidebar() -> str:
    with st.sidebar:
        st.title("🎭 EmotionLens")
        st.caption("Emotion-Based Deception Detector")
        st.divider()

        page = st.radio(
            "Navigate",
            [
                "🎥 Live Detector",
                "🖼️ Image Analyzer",
                "📊 Session Analytics",
                "🧠 Train Model",
                "ℹ️  About",
            ],
        )

        st.divider()
        with st.expander("Settings", expanded=True):
            conf_threshold = st.slider("Confidence threshold", 0.30, 0.90, 0.45, 0.05)
            ema_alpha = st.slider("EMA alpha", 0.10, 0.90, 0.30, 0.05)
            if st.button("Apply changes", use_container_width=True):
                st.session_state.mapper.conf_threshold = conf_threshold
                st.session_state.mapper.alpha = ema_alpha
                st.success("Settings updated.")

        model_path = pathlib.Path('models/best_model.pt')
        if model_path.exists():
            st.markdown("<span style='color:#00C853;'>●</span> Model ready", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#D50000;'>●</span> No model - train first", unsafe_allow_html=True)

        if st.button("Reset Session", use_container_width=True):
            st.session_state.mapper.reset()
            st.session_state.low_quality_streak = 0
            st.session_state.session_start = datetime.datetime.now()
            st.session_state.running = False
            st.success("Session reset complete.")

        st.divider()
        with st.expander("Ethical Disclaimer"):
            st.write(
                "For educational purposes only. Deception tendency scores are probabilistic "
                "estimates based on psycholinguistic research (Paul Ekman, 1969-2003) and are "
                "NOT scientifically absolute. Do not use for real-world lie detection."
            )

    return page


def main() -> None:
    inject_css()
    initialize_session_state()

    if st.session_state.inference is None:
        st.session_state.inference = load_inference()

    page = render_sidebar()

    if page == "🎥 Live Detector":
        render_live_detector_page(st.session_state.inference, st.session_state.mapper)
    elif page == "🖼️ Image Analyzer":
        render_image_analyzer_page(st.session_state.inference, st.session_state.mapper)
    elif page == "📊 Session Analytics":
        render_analytics_page(st.session_state.mapper)
    elif page == "🧠 Train Model":
        render_training_page()
    elif page == "ℹ️  About":
        render_about_page()


if __name__ == '__main__':
    main()
