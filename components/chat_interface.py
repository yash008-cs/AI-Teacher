import streamlit as st
from typing import Optional
from services.memory_service import MemoryService
from services.zep_service import ZepService
from services.base_provider import BaseAIService
from services.rag_service import RAGService
from utils.session_manager import (
    start_new_thread,
    reset_session,
    sync_current_thread_to_recents,
    switch_to_thread,
)
from components.avatar import render_teacher_avatar
from services.voice import (
    ElevenLabsSTTService,
    ElevenLabsTTSService,
    VoiceStreamPipe,
    synthesize_speech_neural,
)



def render_sidebar(
    zep_service: ZepService,
    ai_service: BaseAIService,
    rag_service: Optional[RAGService] = None,
    stt_service: Optional[ElevenLabsSTTService] = None,
    tts_service: Optional[ElevenLabsTTSService] = None,
):
    """
    Renders the minimalist sidebar with learner info, thread lifecycle, and memory diagnostics.
    """
    with st.sidebar:
        avatar = render_teacher_avatar(state="IDLE", size=42)
        sidebar_brand = (
            f'<div style="display: flex; align-items: center; gap: 0.85rem; margin-bottom: 1.2rem;">'
            f'{avatar}'
            f'<div>'
            f'<div style="font-weight: 700; font-size: 1.1rem; color: var(--header-title-color);">AI Teacher</div>'
            f'<div style="font-size: 0.78rem; color: var(--accent-purple);">Personal Tutor</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(sidebar_brand, unsafe_allow_html=True)

        st.markdown("---")

        # Learner Profile Summary
        learner_box = (
            f'<div style="background: var(--msg-user-bg); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 0.85rem; margin-bottom: 1.2rem;">'
            f'<div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted);">Current Learner</div>'
            f'<div style="font-size: 1.1rem; font-weight: 700; color: var(--text-primary); margin-top: 0.2rem;">{st.session_state.user_name}</div>'
            f'</div>'
        )
        st.markdown(learner_box, unsafe_allow_html=True)

        # New Chat button (Creates fresh thread while preserving recent chat history)
        if st.button("➕ New Chat", key="new_chat_sidebar_btn", width="stretch", help="Start a new conversation thread while keeping your learning memory"):
            # Create new thread under the same user
            start_new_thread()
            zep_service.create_thread(
                user_id=st.session_state.zep_user_id,
                thread_id=st.session_state.zep_thread_id
            )
            # Ingest greeting into new thread
            if st.session_state.messages:
                zep_service.add_message(
                    thread_id=st.session_state.zep_thread_id,
                    role="assistant",
                    content=st.session_state.messages[0]["content"],
                    name="AI Teacher"
                )
            st.rerun()

        # Recents Section (replacing AI model, Knowledge base, Zep memory, Voice input, Voice output)
        st.markdown('<div class="recents-header">Recents</div>', unsafe_allow_html=True)

        recent_chats = st.session_state.get("recent_chats", [])
        if not recent_chats:
            st.markdown(
                '<div style="color: var(--text-muted); font-size: 0.82rem; padding: 0.4rem 0.2rem; font-style: italic;">'
                'No recent chats yet'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            for chat in recent_chats:
                tid = chat.get("thread_id")
                title = chat.get("title", "Conversation")
                is_active = (tid == st.session_state.get("zep_thread_id"))

                display_title = title if len(title) <= 26 else title[:24].rstrip() + "..."
                btn_type = "primary" if is_active else "secondary"

                if st.button(
                    display_title,
                    key=f"rec_chat_{tid}",
                    type=btn_type,
                    width="stretch",
                    help=title,
                ):
                    if not is_active:
                        switch_to_thread(tid)
                        st.rerun()

        # Appearance Mode Selector
        st.markdown("---")
        theme_val = st.radio(
            "🎨 Appearance",
            options=["🌙 Dark", "☀️ Light"],
            index=0 if st.session_state.get("theme_mode", "Dark") == "Dark" else 1,
            horizontal=True,
            key="chat_theme_selector",
        )
        current_theme = "Light" if "Light" in theme_val else "Dark"
        if st.session_state.get("theme_mode") != current_theme:
            st.session_state.theme_mode = current_theme
            st.rerun()

        # Collapsible Memory & Diagnostic Details
        with st.expander("🔍 Session & Memory Details"):
            st.caption(f"**Learner ID:** `{st.session_state.zep_user_id}`")
            st.caption(f"**Thread ID:** `{st.session_state.zep_thread_id}`")
            if st.session_state.last_context:
                st.caption("**Last Retrieved Context Block:**")
                st.code(st.session_state.last_context, language="markdown")
            else:
                st.caption("_Context will appear as Zep ingests conversation turns._")

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # Switch Learner / Reset Session
        if st.button("🚪 Switch Learner", width="stretch", help="End current session and return to the onboarding screen"):
            reset_session()
            st.rerun()


def process_user_turn(
    user_query: str,
    memory_service: MemoryService,
    tts_service: Optional[ElevenLabsTTSService] = None,
    voice_output_enabled: bool = False,
):
    """
    Unified question processing pipeline for BOTH typed chat and microphone input:
    1. Records user message in UI session state.
    2. Calls memory_service.retrieve_context_and_stream_response(...) which handles:
       - Zep user message ingestion
       - Zep user context retrieval
       - RAG vector search & top-K chunk retrieval
       - Grounded unified prompt assembly
       - Gemini streaming generation
    3. Streams text live in UI via st.write_stream(stream).
    4. Displays course sources expandable badge.
    5. Records completed assistant answer into Zep memory.
    """
    clean_query = (user_query or "").strip()
    if not clean_query:
        return

    # De-duplication guard: prevent re-running on accidental Streamlit reruns
    if st.session_state.get("last_processed_query") == clean_query:
        return
    st.session_state.last_processed_query = clean_query

    # Immediately display and store user message
    st.session_state.messages.append({"role": "user", "content": clean_query})
    sync_current_thread_to_recents()
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(clean_query)

    # Render assistant response with fast live text streaming
    with st.chat_message("assistant", avatar="👩‍🏫"):
        with st.spinner("AI Teacher is thinking, consulting course notes & recalling your learning context..."):
            raw_context, stream, sources = memory_service.retrieve_context_and_stream_response(
                thread_id=st.session_state.zep_thread_id,
                user_message=clean_query,
                learner_name=st.session_state.user_name,
                conversation_history=st.session_state.messages,
                voice_mode=voice_output_enabled,
            )

            if raw_context:
                st.session_state.last_context = raw_context

        # TTS Voice Output integration
        voice_pipe = None
        display_stream = stream
        elevenlabs_quota_out = st.session_state.get("elevenlabs_quota_exhausted", True)
        use_elevenlabs = bool(voice_output_enabled and tts_service and tts_service.is_configured() and not elevenlabs_quota_out)

        if use_elevenlabs:
            voice_pipe = VoiceStreamPipe(tts_service=tts_service, enabled=True)
            display_stream = voice_pipe.pipe(stream)

        # Stream the generated response into the UI in real time
        full_response = st.write_stream(display_stream)

        # Retrieve and play synthesized speech if voice mode is enabled
        audio_bytes = None
        if voice_output_enabled:
            # 1. First check if ElevenLabs streaming pipe produced audio
            if voice_pipe:
                try:
                    audio_bytes = voice_pipe.get_audio(timeout=3.0)
                except Exception:
                    audio_bytes = None

            # 2. Resilient Neural Fallback: Ultra-fast Microsoft Edge Neural TTS
            if not audio_bytes and full_response:
                try:
                    audio_bytes = synthesize_speech_neural(full_response)
                except Exception as synth_err:
                    audio_bytes = None

            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            else:
                st.caption("🎙️ *Audio response temporarily unavailable.*")

        # Display source attribution expandable badge
        if sources:
            with st.expander(f"📚 Course Sources ({len(sources)})", expanded=False):
                for s in sources:
                    page_txt = f" | Page {s['page_number']}" if s.get("page_number") else ""
                    st.markdown(
                        f"**• {s.get('source_filename', 'Unknown')}** ({s.get('topic', 'General')}{page_txt}) "
                        f"— *Relevance: {s.get('score', 0.0):.2f}*"
                    )
                    if s.get("snippet"):
                        st.caption(f"_{s['snippet']}..._")

        # Record completed assistant answer into Zep memory
        if full_response and isinstance(full_response, str):
            memory_service.record_assistant_response(
                thread_id=st.session_state.zep_thread_id,
                content=full_response,
            )
            msg_payload = {
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            }
            if audio_bytes:
                msg_payload["audio_bytes"] = audio_bytes
            st.session_state.messages.append(msg_payload)
            sync_current_thread_to_recents()


def render_chat_interface(
    memory_service: MemoryService,
    zep_service: ZepService,
    ai_service: BaseAIService,
    rag_service: Optional[RAGService] = None,
    stt_service: Optional[ElevenLabsSTTService] = None,
    tts_service: Optional[ElevenLabsTTSService] = None,
):
    """
    Renders the chat header, conversation message feed, and search bar with microphone.
    """
    # 1. Render Sidebar
    render_sidebar(zep_service, ai_service, rag_service, stt_service, tts_service)

    # 2. Main Header Bar
    avatar_header = render_teacher_avatar(state="IDLE", size=36)
    chat_header_html = (
        f'<div class="chat-header">'
        f'<div class="chat-header-title">'
        f'{avatar_header}'
        f'<div>'
        f'<div>AI TEACHER</div>'
        f'<div class="chat-header-subtitle">Your Personal AI Tutor • Guiding <b>{st.session_state.user_name}</b></div>'
        f'</div>'
        f'</div>'
        f'<div>'
        f'<span class="status-pill status-active">Online</span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(chat_header_html, unsafe_allow_html=True)

    # 3. Clean and Render Historical Messages (no audio player)
    # If the user stopped the previous run before the assistant finished, remove the incomplete query
    if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
        st.session_state.messages.pop()
        st.session_state.last_processed_query = None
        st.toast("Generation stopped. Ready for your next question.", icon="🛑")

    # Auto-migrate initial greeting if running existing session with old text
    if st.session_state.messages and st.session_state.messages[0].get("role") == "assistant":
        first_content = st.session_state.messages[0].get("content", "")
        if "I'm here to guide you through" in first_content or "What topic would you like to explore" in first_content:
            name = st.session_state.get("user_name", "Yashraj")
            st.session_state.messages[0]["content"] = f"Hello {name}!\n\nWhat topic would you like to learn"

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])
        audio_bytes = msg.get("audio_bytes")

        with st.chat_message(role, avatar="🧑‍🎓" if role == "user" else "👩‍🏫"):
            st.markdown(content)
            if role == "assistant" and audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", autoplay=False)
            if role == "assistant" and sources:
                with st.expander(f"📚 Course Sources ({len(sources)})", expanded=False):
                    for s in sources:
                        page_txt = f" | Page {s['page_number']}" if s.get("page_number") else ""
                        st.markdown(
                            f"**• {s.get('source_filename', 'Unknown')}** ({s.get('topic', 'General')}{page_txt}) "
                            f"— *Relevance: {s.get('score', 0.0):.2f}*"
                        )
                        if s.get("snippet"):
                            st.caption(f"_{s['snippet']}..._")

    # 4. Handle Incoming User Input (integrated microphone in search bar + Voice mode toggle + stop button)
    accept_voice = bool(stt_service and stt_service.is_configured())

    with st.bottom:
        col_btn, _ = st.columns([3.2, 6.8])
        with col_btn:
            voice_mode_active = st.session_state.get("voice_mode_enabled", True)
            btn_label = "🔊 Voice mode: ON" if voice_mode_active else "🎙️ Voice mode: OFF"
            btn_type = "primary" if voice_mode_active else "secondary"
            btn_help = (
                "Voice mode is ON: AI Teacher will speak answers aloud (TTS). Click to switch to text-only mode."
                if voice_mode_active
                else "Voice mode is OFF (silent mode). Click to enable spoken answers (TTS)."
            )
            if st.button(
                btn_label,
                key="voice_mode_toggle_btn",
                type=btn_type,
                help=btn_help,
            ):
                st.session_state.voice_mode_enabled = not voice_mode_active
                st.session_state.voice_output_enabled = st.session_state.voice_mode_enabled
                st.rerun()

        chat_prompt = st.chat_input(
            "Ask your AI Teacher something...",
            accept_audio=accept_voice,
            submit_mode="stop",
        )

    if chat_prompt:
        user_query = ""

        # Check if student spoke their question via the search bar microphone button
        if hasattr(chat_prompt, "audio") and chat_prompt.audio:
            with st.spinner("Processing voice input..."):
                elevenlabs_out = st.session_state.get("elevenlabs_quota_exhausted", True)
                # 1. Try ElevenLabs STT if configured and quota not exhausted
                if not elevenlabs_out and stt_service and stt_service.is_configured():
                    try:
                        user_query = stt_service.transcribe_file(chat_prompt.audio)
                    except Exception as voice_err:
                        user_query = ""
                        st.session_state["elevenlabs_quota_exhausted"] = True

                # 2. Resilient Fallback: Ultra-fast direct Gemini Multimodal Audio transcription
                if not user_query and ai_service and hasattr(ai_service, "transcribe_audio"):
                    try:
                        user_query = ai_service.transcribe_audio(chat_prompt.audio)
                    except Exception as gem_err:
                        pass

                if not user_query and not (hasattr(chat_prompt, "text") and chat_prompt.text):
                    st.warning("⚠️ Could not recognize speech from the recording. Please try speaking again or typing your question.")

        # Check if student typed their question into the search bar
        if not user_query:
            if hasattr(chat_prompt, "text") and chat_prompt.text:
                user_query = chat_prompt.text.strip()
            elif isinstance(chat_prompt, str):
                user_query = chat_prompt.strip()

        if user_query:
            voice_output_on = bool(
                st.session_state.get("voice_mode_enabled", True)
            )
            process_user_turn(
                user_query=user_query,
                memory_service=memory_service,
                tts_service=tts_service,
                voice_output_enabled=voice_output_on,
            )


