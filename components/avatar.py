"""
AI Teacher Avatar Component.
Provides lightweight visual states (IDLE, THINKING, TEACHING) with modern SVG and CSS animations.
"""
import streamlit as st


def render_teacher_avatar(state: str = "IDLE", size: int = 48) -> str:
    """
    Returns a clean HTML snippet of the animated teacher avatar for the current state.
    Guaranteed to have zero leading whitespace to prevent Markdown code-block interpretation.
    """
    is_thinking = state == "THINKING"
    glow_color = "#38bdf8" if is_thinking else "#c084fc"
    anim_class = "avatar-thinking" if is_thinking else "avatar-idle"

    svg_icon = (
        f'<svg width="{int(size * 0.58)}" height="{int(size * 0.58)}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 6C13.66 6 15 7.34 15 9C15 10.66 13.66 12 12 12C10.34 12 9 10.66 9 9C9 7.34 10.34 6 12 6ZM12 20.2C9.5 20.2 7.29 18.92 6 16.98C6.03 14.99 10 13.9 12 13.9C13.99 13.9 17.97 14.99 18 16.98C16.71 18.92 14.5 20.2 12 20.2Z" fill="#ffffff"/>'
        f'</svg>'
    )

    return (
        f'<div class="teacher-avatar-wrapper" style="display:inline-flex;align-items:center;justify-content:center;position:relative;">'
        f'<div class="teacher-avatar-circle {anim_class}" style="width:{size}px;height:{size}px;border-radius:50%;'
        f'background:linear-gradient(135deg, #4c1d95 0%, #7c3aed 50%, #c026d3 100%);'
        f'display:flex;align-items:center;justify-content:center;'
        f'box-shadow:0 0 20px {glow_color}66;border:2px solid rgba(255,255,255,0.25);">'
        f'{svg_icon}'
        f'</div>'
        f'</div>'
    )
