"""
Session Initialization Screen Component.
Presents the initial onboarding card for learners before entering the chat room.
"""
import streamlit as st
from utils.helpers import sanitize_user_id, generate_thread_id
from services.zep_service import ZepService
from components.avatar import render_teacher_avatar


def render_session_screen(zep_service: ZepService):
    """
    Renders the centered session onboarding interface.
    """
    # Spacing for center alignment
    st.markdown("<div style='height: 4vh;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Centered Avatar, Badge, Title and Subtitle (no leading spaces on any line)
        avatar_html = render_teacher_avatar(state="IDLE", size=68)
        header_html = (
            f'<div style="text-align: center; margin-bottom: 1.5rem;">'
            f'{avatar_html}'
            f'<div style="margin-top: 1rem;">'
            f'<span class="session-badge">✦ AI Personalized Mentor ✦</span>'
            f'</div>'
            f'<h1 class="session-title">AI TEACHER</h1>'
            f'<p class="session-subtitle">Start Your Learning Session</p>'
            f'</div>'
        )
        st.markdown(header_html, unsafe_allow_html=True)

        with st.form("session_init_form", clear_on_submit=False):
            # Native Streamlit text input with empty placeholder
            name_input = st.text_input(
                label="Your Name",
                placeholder="",
                key="name_input_field",
            )

            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

            submit_button = st.form_submit_button("Initialize New Session", width="stretch")

            if submit_button:
                cleaned_name = name_input.strip()
                if not cleaned_name:
                    st.warning("Please enter your name to begin your learning session.")
                else:
                    with st.spinner("Initializing personalized memory graph..."):
                        # 1. Generate safe IDs
                        user_id = sanitize_user_id(cleaned_name)
                        thread_id = generate_thread_id()

                        # 2. Setup Zep User and Thread
                        zep_service.get_or_create_user(user_id=user_id, first_name=cleaned_name)
                        zep_service.create_thread(user_id=user_id, thread_id=thread_id)

                        # 3. Store state in st.session_state
                        st.session_state.user_name = cleaned_name
                        st.session_state.zep_user_id = user_id
                        st.session_state.zep_thread_id = thread_id
                        st.session_state.initialized = True

                        # 4. Prepare initial greeting
                        greeting = (
                            f"Hello {cleaned_name}!\n\n"
                            "What topic would you like to learn"
                        )

                        # Also ingest initial greeting into Zep thread so context includes it
                        zep_service.add_message(
                            thread_id=thread_id,
                            role="assistant",
                            content=greeting,
                            name="AI Teacher",
                        )

                        st.session_state.messages = [
                            {"role": "assistant", "content": greeting}
                        ]

                    st.rerun()

        # Footer feature highlights
        footer_html = (
            '<div style="text-align: center; margin-top: 2rem; color: var(--text-muted); font-size: 0.8rem;">'
            '<span>⚡ Gemini LLM</span> &nbsp;•&nbsp; '
            '<span>🧠 Zep Context Graph Memory</span> &nbsp;•&nbsp; '
            '<span>📚 Grounded RAG Knowledge Base</span>'
            '</div>'
        )
        st.markdown(footer_html, unsafe_allow_html=True)

    # Sidebar settings on onboarding screen
    with st.sidebar:
        st.markdown("### 🎨 Preferences")
        theme_val = st.radio(
            "Theme Mode",
            options=["🌙 Dark", "☀️ Light"],
            index=0 if st.session_state.get("theme_mode", "Dark") == "Dark" else 1,
            horizontal=True,
            key="theme_selector_radio",
        )
        current_theme = "Light" if "Light" in theme_val else "Dark"
        if st.session_state.get("theme_mode") != current_theme:
            st.session_state.theme_mode = current_theme
            st.rerun()
