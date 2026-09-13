"""
Zep Cloud Memory Service.
Integrates with the official zep-cloud Python SDK.
Handles User, Thread, Message ingestion, and Context retrieval.
"""
import os
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger("ai_teacher.zep")

try:
    from zep_cloud.client import Zep
    from zep_cloud.types import Message
    from zep_cloud.errors import NotFoundError, ConflictError
    ZEP_SDK_AVAILABLE = True
except ImportError:
    ZEP_SDK_AVAILABLE = False
    NotFoundError = Exception  # type: ignore
    ConflictError = Exception  # type: ignore
    logger.warning("zep-cloud SDK not installed.")


class ZepService:
    """
    Manages long-term learner memory using Zep Cloud (zep-cloud 3.x).
    Coordinates User, Thread, Message ingestion, and Context retrieval.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ZEP_API_KEY", "").strip()
        self.client: Optional[Zep] = None
        self._initialize_client()

    def _initialize_client(self):
        if not ZEP_SDK_AVAILABLE:
            logger.warning("zep-cloud package is unavailable.")
            return

        if self.api_key and not self.api_key.startswith("your_"):
            try:
                self.client = Zep(api_key=self.api_key)
                logger.info("Zep Cloud client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Zep client: {e}")
                self.client = None
        else:
            logger.info("ZEP_API_KEY not provided or placeholder. Operating in memory-bypass mode.")

    def is_configured(self) -> bool:
        """Returns True if a valid Zep client is initialized."""
        return self.client is not None

    def get_or_create_user(self, user_id: str, first_name: str) -> Dict[str, Any]:
        """
        Retrieves an existing learner profile from Zep, or creates a new one if not found.
        """
        if not self.is_configured():
            logger.debug(f"[Mock Mode] get_or_create_user: {user_id} ({first_name})")
            return {"user_id": user_id, "first_name": first_name, "status": "mock"}

        # Attempt to get user first
        try:
            user = self.client.user.get(user_id=user_id)
            logger.info(f"Retrieved existing Zep user: {user_id}")
            return {"user_id": user_id, "first_name": first_name, "user": user, "status": "retrieved"}
        except NotFoundError:
            logger.info(f"Zep user '{user_id}' not found. Creating new learner user...")
        except Exception as e:
            logger.debug(f"User.get check for '{user_id}': {e}")

        try:
            user = self.client.user.add(
                user_id=user_id,
                first_name=first_name,
            )
            logger.info(f"Created new Zep user: {user_id}")
            return {"user_id": user_id, "first_name": first_name, "user": user, "status": "created"}
        except ConflictError:
            # Concurrently created, fetch it
            user = self.client.user.get(user_id=user_id)
            return {"user_id": user_id, "first_name": first_name, "user": user, "status": "retrieved"}
        except Exception as e:
            logger.error(f"Error creating/retrieving Zep user '{user_id}': {e}")
            return {"user_id": user_id, "first_name": first_name, "status": "error", "error": str(e)}

    def create_thread(self, user_id: str, thread_id: str) -> Dict[str, Any]:
        """
        Creates a new conversation thread under the specified learner ID.
        If thread exists, retrieves existing thread.
        """
        if not self.is_configured():
            logger.debug(f"[Mock Mode] create_thread: {thread_id} for {user_id}")
            return {"thread_id": thread_id, "user_id": user_id, "status": "mock"}

        try:
            thread = self.client.thread.create(
                thread_id=thread_id,
                user_id=user_id,
            )
            logger.info(f"Created Zep thread '{thread_id}' for user '{user_id}'")
            return {"thread_id": thread_id, "user_id": user_id, "thread": thread, "status": "created"}
        except Exception as e:
            # Check if thread already exists
            try:
                thread = self.client.thread.get(thread_id=thread_id)
                logger.info(f"Retrieved existing Zep thread: '{thread_id}'")
                return {"thread_id": thread_id, "user_id": user_id, "thread": thread, "status": "retrieved"}
            except Exception:
                logger.error(f"Error creating Zep thread '{thread_id}': {e}")
                return {"thread_id": thread_id, "user_id": user_id, "status": "error", "error": str(e)}

    def add_message(self, thread_id: str, role: str, content: str, name: Optional[str] = None) -> bool:
        """
        Adds a single user or assistant message to the active Zep thread.
        Uses RFC3339 timestamps for temporal awareness.
        """
        if not self.is_configured():
            logger.debug(f"[Mock Mode] add_message to {thread_id} ({role}): {content[:40]}...")
            return True

        # Map role safely to supported literal ('user', 'assistant', 'system')
        valid_role = role if role in ("user", "assistant", "system") else "user"

        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            safe_content = content if len(content) <= 4000 else content[:3990] + "..."
            msg = Message(
                created_at=timestamp,
                role=valid_role,
                content=safe_content,
                name=name or ("Learner" if valid_role == "user" else "AI Teacher")
            )
            self.client.thread.add_messages(
                thread_id=thread_id,
                messages=[msg]
            )
            logger.debug(f"Added {valid_role} message to Zep thread {thread_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding message to Zep thread '{thread_id}': {e}")
            return False

    def get_user_context(self, thread_id: str) -> Optional[str]:
        """
        Retrieves the assembled temporal Context Block (<USER_SUMMARY>, <FACTS>, etc.)
        from the learner's knowledge graph for the current thread.
        """
        if not self.is_configured():
            logger.debug(f"[Mock Mode] get_user_context for {thread_id}")
            return None

        try:
            user_context = self.client.thread.get_user_context(thread_id=thread_id)
            if user_context and hasattr(user_context, "context") and user_context.context:
                context_block = user_context.context.strip()
                logger.info(f"Retrieved Zep context for thread {thread_id} (Length: {len(context_block)})")
                return context_block
            return None
        except Exception as e:
            logger.warning(f"Error retrieving context for Zep thread '{thread_id}': {e}")
            return None
