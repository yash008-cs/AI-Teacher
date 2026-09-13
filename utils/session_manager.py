"""
Streamlit session state management.
Manages ephemeral client-side state, distinguishing it from persistent Zep memory.
Provides multi-thread chat history management modeled after ChatGPT's sidebar recents.
"""
import time
from typing import Optional
import streamlit as st
from .helpers import generate_thread_id


def init_session_state():
    """
    Initializes default keys in Streamlit's session state.
    """
    defaults = {
        "initialized": False,          # True once learner completes session onboarding
        "user_name": "",               # Learner's display name
        "zep_user_id": "",             # Persistent learner ID in Zep
        "zep_thread_id": "",           # Active conversation thread ID in Zep
        "messages": [],                # Ephemeral chat message history for UI rendering
        "current_topic": "General AI", # Current topic being explored
        "last_context": "",            # Last retrieved Zep memory context (for diagnostics)
        "is_thinking": False,          # State for avatar / indicator
        "theme_mode": "Dark",          # Active UI theme mode (Dark / Light)
        "last_processed_query": None,  # De-duplication guard for voice/text queries
        "voice_status": "IDLE",        # Ephemeral voice state indicator
        "voice_partial_transcript": "", # Live speech subtitle buffer
        "voice_final_transcript": "",   # Committed voice query buffer
        "voice_output_enabled": True,  # Whether teacher responds with voice audio
        "voice_mode_enabled": True,    # Dedicated Voice mode toggle near search bar
        "recent_chats": [],            # List of past conversation threads [{"thread_id", "title", "messages", "last_context", "timestamp"}]
        "current_chat_title": "New conversation", # Title of the active chat
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def sync_current_thread_to_recents():
    """
    Saves or updates the current active thread in st.session_state.recent_chats.
    Only creates/updates history if at least one user question has been asked.
    """
    thread_id = st.session_state.get("zep_thread_id")
    if not thread_id:
        return

    messages = st.session_state.get("messages", [])
    # Check if there is at least one user question
    first_user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), None)
    if not first_user_msg:
        return

    # Derive clean title if not already set or default
    current_title = st.session_state.get("current_chat_title")
    if not current_title or current_title == "New conversation":
        clean = first_user_msg.strip().split("\n")[0].replace("#", "").replace("*", "").strip()
        if len(clean) > 30:
            clean = clean[:28].rstrip() + "..."
        current_title = clean[0].upper() + clean[1:] if clean else "Conversation"
        st.session_state.current_chat_title = current_title

    recents = st.session_state.get("recent_chats", [])
    idx = next((i for i, c in enumerate(recents) if c.get("thread_id") == thread_id), None)
    entry = {
        "thread_id": thread_id,
        "title": current_title,
        "messages": list(messages),
        "last_context": st.session_state.get("last_context", ""),
        "timestamp": time.time(),
    }
    if idx is not None:
        recents[idx] = entry
    else:
        recents.insert(0, entry)
    st.session_state.recent_chats = recents


def start_new_thread():
    """
    Starts a fresh conversation thread while preserving previous threads in recents
    and long-term Zep memory.
    """
    # 1. Sync current active thread into recents before starting new one
    sync_current_thread_to_recents()

    # 2. Generate new thread
    new_thread_id = generate_thread_id()
    st.session_state.zep_thread_id = new_thread_id
    st.session_state.current_chat_title = "New conversation"
    st.session_state.last_context = ""
    st.session_state.last_processed_query = None

    # 3. Add initial greeting message to the new thread
    name = st.session_state.user_name or "there"
    greeting = f"Hello {name}!\n\nWhat topic would you like to learn"
    st.session_state.messages = [{
        "role": "assistant",
        "content": greeting
    }]


def switch_to_thread(target_thread_id: str):
    """
    Switches active conversation to a past thread from recent chats.
    """
    if target_thread_id == st.session_state.get("zep_thread_id"):
        return

    # 1. Sync current state before switching
    sync_current_thread_to_recents()

    # 2. Find target thread in recents
    recents = st.session_state.get("recent_chats", [])
    target = next((c for c in recents if c.get("thread_id") == target_thread_id), None)
    if target:
        st.session_state.zep_thread_id = target["thread_id"]
        st.session_state.messages = list(target["messages"])
        st.session_state.last_context = target.get("last_context", "")
        st.session_state.current_chat_title = target.get("title", "Conversation")
        st.session_state.last_processed_query = None


def reset_session():
    """
    Resets the entire application state back to the onboarding screen.
    """
    st.session_state.initialized = False
    st.session_state.user_name = ""
    st.session_state.zep_user_id = ""
    st.session_state.zep_thread_id = ""
    st.session_state.messages = []
    st.session_state.last_context = ""
    st.session_state.is_thinking = False
    st.session_state.last_processed_query = None
    st.session_state.voice_status = "IDLE"
    st.session_state.voice_partial_transcript = ""
    st.session_state.voice_final_transcript = ""
    st.session_state.voice_output_enabled = True
    st.session_state.voice_mode_enabled = True
    st.session_state.recent_chats = []
    st.session_state.current_chat_title = "New conversation"
