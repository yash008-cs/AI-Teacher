"""
Helper functions for data normalization, identifier generation, and sanitization.
"""
import re
import uuid
from datetime import datetime, timezone


def sanitize_user_id(name: str) -> str:
    """
    Creates a safe, deterministic learner user ID from their name.
    Example: 'Yashraj Patel' -> 'learner_yashraj_patel'
    """
    if not name:
        return f"learner_{uuid.uuid4().hex[:8]}"

    # Lowercase, replace non-alphanumeric characters with underscores
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    if not cleaned:
        cleaned = uuid.uuid4().hex[:8]

    return f"learner_{cleaned}"


def generate_thread_id() -> str:
    """
    Generates a unique Zep thread identifier.
    """
    return f"thread_{uuid.uuid4().hex}"


def get_current_utc_timestamp() -> str:
    """
    Returns current UTC timestamp in ISO 8601 (RFC3339) format.
    """
    return datetime.now(timezone.utc).isoformat()
