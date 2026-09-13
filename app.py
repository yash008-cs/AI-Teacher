"""
AI Teacher Application Entrypoint.
Coordinates environment loading, CSS styling, session state, and view routing.
"""
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv(override=True)

# Load Streamlit Cloud secrets into environment variables
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

from utils.session_manager import init_session_state
from services import ZepService, MemoryService, get_ai_service, RAGService
from services.voice import ElevenLabsSTTService, ElevenLabsTTSService

# Configure Streamlit page parameters
st.set_page_config(
    page_title="AI Teacher • Your Personal AI Tutor",
    page_icon="👩‍🏫",
    layout="centered",
    initial_sidebar_state="expanded",
)


def load_custom_css():
    """Injects adaptive dark/light AI aesthetic styles and runtime theme overrides."""
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

    # Check active theme from session state or Streamlit context
    theme_type = getattr(st.context.theme, "type", None)
    is_light = (st.session_state.get("theme_mode") == "Light") or (theme_type == "light")
    if is_light:
        light_overrides = """
        <style>
        :root, .stApp, body, div[data-testid="stAppViewContainer"] {
          --bg-primary: #faf8ff !important;
          --bg-secondary: #f3effc !important;
          --bg-card: #ffffff !important;
          --bg-glass: rgba(255, 255, 255, 0.95) !important;
          --border-subtle: rgba(147, 51, 234, 0.18) !important;
          --border-focus: rgba(147, 51, 234, 0.55) !important;
          --accent-purple: #7c3aed !important;
          --accent-glow: #9333ea !important;
          --accent-violet: #6d28d9 !important;
          --accent-cyan: #0284c7 !important;
          --text-primary: #1e1b4b !important;
          --text-secondary: #334155 !important;
          --text-muted: #64748b !important;
          --app-gradient: radial-gradient(circle at 50% 10%, #f3effe 0%, #faf8ff 60%, #f5f3ff 100%) !important;
          --sidebar-bg: #fbf9ff !important;
          --msg-user-bg: rgba(124, 58, 237, 0.08) !important;
          --msg-assistant-bg: #ffffff !important;
          --chat-input-bg: #ffffff !important;
          --pre-bg: #f8fafc !important;
          --header-title-color: #1e1b4b !important;
          --title-gradient: linear-gradient(135deg, #1e1b4b 30%, #7c3aed 100%) !important;
          --expander-bg: #ffffff !important;
          --card-shadow: 0 4px 20px rgba(124, 58, 237, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        }
        .stApp {
          background: radial-gradient(circle at 50% 10%, #f3effe 0%, #faf8ff 60%, #f5f3ff 100%) !important;
          color: #1e1b4b !important;
        }
        /* Fixed bottom chat container in Light Mode - No more black bar! */
        div[data-testid="stBottom"],
        div[data-testid="stBottomBlockContainer"],
        div[data-testid="stBottom"] > div,
        .stBottom,
        div[data-testid="stAppViewContainer"] > div:has(div[data-testid="stChatInput"]),
        footer {
          background: #faf8ff !important;
          background-color: #faf8ff !important;
          border-top: 1px solid rgba(147, 51, 234, 0.15) !important;
        }
        div[data-testid="stChatInput"] {
          background: transparent !important;
          background-color: transparent !important;
        }
        div[data-testid="stChatInput"] > div {
          background: #ffffff !important;
          background-color: #ffffff !important;
          border: 1px solid rgba(147, 51, 234, 0.25) !important;
          box-shadow: 0 4px 20px rgba(124, 58, 237, 0.08) !important;
        }
        div[data-testid="stChatInput"] textarea {
          color: #1e1b4b !important;
          background: transparent !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
          color: #64748b !important;
        }
        div[data-testid="stChatInput"] button {
          color: #7c3aed !important;
          background: transparent !important;
          border-radius: 10px !important;
          transition: all 0.2s ease !important;
        }
        div[data-testid="stChatInput"] button:hover {
          background: rgba(124, 58, 237, 0.12) !important;
          color: #6d28d9 !important;
          transform: scale(1.08) !important;
        }
        div[data-testid="stChatInput"] button[data-testid="stChatInputMicButton"]:hover {
          background: rgba(124, 58, 237, 0.16) !important;
          box-shadow: 0 0 12px rgba(124, 58, 237, 0.25) !important;
        }
        div[data-testid="stChatInput"] button[data-testid="stChatInputStopButton"] {
          background: rgba(239, 68, 68, 0.12) !important;
          color: #dc2626 !important;
          border: 1px solid rgba(239, 68, 68, 0.35) !important;
          border-radius: 10px !important;
        }
        div[data-testid="stChatInput"] button[data-testid="stChatInputStopButton"]:hover {
          background: #dc2626 !important;
          color: #ffffff !important;
        }
        /* Voice Mode button in Light Mode */
        div[data-testid="stBottom"] div.stButton > button[kind="primary"],
        div[data-testid="stBottom"] div.stButton > button[data-testid="baseButton-primary"] {
          background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
          border: 1px solid rgba(124, 58, 237, 0.4) !important;
          color: #ffffff !important;
          box-shadow: 0 2px 10px rgba(124, 58, 237, 0.25) !important;
        }
        div[data-testid="stBottom"] div.stButton > button[kind="secondary"],
        div[data-testid="stBottom"] div.stButton > button[data-testid="baseButton-secondary"] {
          background: #ffffff !important;
          border: 1px solid rgba(147, 51, 234, 0.25) !important;
          color: #64748b !important;
          box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
        }
        div[data-testid="stBottom"] div.stButton > button[kind="secondary"]:hover,
        div[data-testid="stBottom"] div.stButton > button[data-testid="baseButton-secondary"]:hover {
          background: rgba(124, 58, 237, 0.08) !important;
          color: #7c3aed !important;
          border-color: #7c3aed !important;
        }
        /* Sidebar and Sidebar buttons in Light Mode */
        section[data-testid="stSidebar"] {
          background-color: #fbf9ff !important;
          border-right: 1px solid rgba(147, 51, 234, 0.15) !important;
        }
        section[data-testid="stSidebar"] button,
        section[data-testid="stSidebar"] .stButton > button,
        section[data-testid="stSidebar"] button[kind="secondary"],
        section[data-testid="stSidebar"] [data-testid="baseButton-secondary"],
        section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
          background: #ffffff !important;
          background-color: #ffffff !important;
          color: #1e1b4b !important;
          border: 1px solid rgba(147, 51, 234, 0.22) !important;
          border-radius: 12px !important;
          font-weight: 600 !important;
          box-shadow: 0 2px 8px rgba(124, 58, 237, 0.05) !important;
        }
        section[data-testid="stSidebar"] button:hover,
        section[data-testid="stSidebar"] .stButton > button:hover {
          background: rgba(124, 58, 237, 0.08) !important;
          border-color: #7c3aed !important;
          color: #7c3aed !important;
        }
        /* Chat Messages in Light Mode */
        div[data-testid="stChatMessage"] {
          background: #ffffff !important;
          border: 1px solid rgba(147, 51, 234, 0.18) !important;
          border-radius: 18px !important;
          box-shadow: 0 4px 16px rgba(124, 58, 237, 0.06) !important;
        }
        div[data-testid="stChatMessage"]:has([aria-label*="user"]),
        div[data-testid="stChatMessage"]:has([aria-label*="User"]) {
          background: rgba(124, 58, 237, 0.08) !important;
          border: 1px solid rgba(147, 51, 234, 0.2) !important;
        }
        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] span,
        div[data-testid="stChatMessage"] li {
          color: #1e1b4b !important;
        }
        /* Expander and Form in Light Mode */
        div[data-testid="stExpander"] {
          background: #ffffff !important;
          border: 1px solid rgba(147, 51, 234, 0.18) !important;
        }
        div[data-testid="stExpander"] summary {
          color: #1e1b4b !important;
        }
        div[data-testid="stForm"] {
          background: #ffffff !important;
          border: 1px solid rgba(147, 51, 234, 0.2) !important;
        }
        </style>
        """
        st.markdown(light_overrides, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_services():
    """Initializes and caches singleton service instances."""
    zep_service = ZepService()
    ai_service = get_ai_service()
    rag_service = RAGService()
    memory_service = MemoryService(
        zep_service=zep_service,
        ai_service=ai_service,
        rag_service=rag_service,
    )
    stt_service = ElevenLabsSTTService()
    tts_service = ElevenLabsTTSService()
    return zep_service, ai_service, memory_service, rag_service, stt_service, tts_service


def main():
    # 1. Initialize ephemeral session state
    init_session_state()

    # 2. Inject dark-purple AI aesthetic styling (or Light mode if chosen)
    load_custom_css()

    # 3. Initialize backend services
    zep_service, ai_service, memory_service, rag_service, stt_service, tts_service = get_services()

    # 4. Route views based on onboarding status
    if not st.session_state.initialized:
        render_session_screen(zep_service=zep_service)
    else:
        render_chat_interface(
            memory_service=memory_service,
            zep_service=zep_service,
            ai_service=ai_service,
            rag_service=rag_service,
            stt_service=stt_service,
            tts_service=tts_service,
        )


if __name__ == "__main__":
    main()
