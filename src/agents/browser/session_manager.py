"""
Browser session management.

Handles cookie/session persistence between browser tasks.
Sessions are stored as JSON files and can be loaded for subsequent tasks.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from playwright.async_api import BrowserContext

from src.core.config import settings

logger = structlog.get_logger()

# Session storage directory
SESSION_DIR = Path(settings.screenshot_dir).parent / "sessions"


class SessionManager:
    """
    Manages browser sessions (cookies, local storage) for reuse.
    
    Sessions are identified by a session_id (e.g., site domain or user-defined).
    Each session stores:
    - Cookies
    - Storage state (localStorage, sessionStorage)
    - Metadata (created, last used, expires)
    """

    def __init__(self, session_dir: Path | None = None) -> None:
        """
        Initialize session manager.
        
        Args:
            session_dir: Directory to store session files
        """
        self.session_dir = session_dir or SESSION_DIR
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug("session_manager_initialized", path=str(self.session_dir))

    def _session_path(self, session_id: str) -> Path:
        """Get path to session file."""
        # Sanitize session_id for filesystem
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id)
        return self.session_dir / f"{safe_id}.json"

    async def save_session(
        self,
        session_id: str,
        context: BrowserContext,
        expires_hours: int = 24,
    ) -> Path:
        """
        Save browser session state to file.
        
        Args:
            session_id: Unique identifier for this session
            context: Playwright browser context to save
            expires_hours: Hours until session expires
            
        Returns:
            Path to saved session file
        """
        session_path = self._session_path(session_id)
        
        # Get storage state from Playwright
        storage_state = await context.storage_state()
        
        # Add metadata
        session_data = {
            "storage_state": storage_state,
            "metadata": {
                "session_id": session_id,
                "created_at": datetime.utcnow().isoformat(),
                "last_used_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(hours=expires_hours)).isoformat(),
            },
        }
        
        # Save to file
        with open(session_path, "w") as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(
            "session_saved",
            session_id=session_id,
            cookies=len(storage_state.get("cookies", [])),
            path=str(session_path),
        )
        
        return session_path

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        """
        Load a saved session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Storage state dict for Playwright, or None if not found/expired
        """
        session_path = self._session_path(session_id)
        
        if not session_path.exists():
            logger.debug("session_not_found", session_id=session_id)
            return None
        
        try:
            with open(session_path, "r") as f:
                session_data = json.load(f)
            
            # Check expiration
            metadata = session_data.get("metadata", {})
            expires_at = metadata.get("expires_at")
            
            if expires_at:
                expires = datetime.fromisoformat(expires_at)
                if datetime.utcnow() > expires:
                    logger.info("session_expired", session_id=session_id)
                    self.delete_session(session_id)
                    return None
            
            # Update last used
            metadata["last_used_at"] = datetime.utcnow().isoformat()
            with open(session_path, "w") as f:
                json.dump(session_data, f, indent=2)
            
            logger.info(
                "session_loaded",
                session_id=session_id,
                cookies=len(session_data["storage_state"].get("cookies", [])),
            )
            
            return session_data["storage_state"]
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("session_corrupt", session_id=session_id, error=str(e))
            self.delete_session(session_id)
            return None

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a saved session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was deleted
        """
        session_path = self._session_path(session_id)
        
        if session_path.exists():
            session_path.unlink()
            logger.info("session_deleted", session_id=session_id)
            return True
        
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all saved sessions with metadata.
        
        Returns:
            List of session metadata dicts
        """
        sessions = []
        
        for session_file in self.session_dir.glob("*.json"):
            try:
                with open(session_file, "r") as f:
                    data = json.load(f)
                
                metadata = data.get("metadata", {})
                metadata["file"] = session_file.name
                metadata["size_bytes"] = session_file.stat().st_size
                sessions.append(metadata)
                
            except Exception:
                continue
        
        return sorted(sessions, key=lambda s: s.get("last_used_at", ""), reverse=True)

    def cleanup_expired(self) -> int:
        """
        Remove all expired sessions.
        
        Returns:
            Number of sessions removed
        """
        removed = 0
        
        for session_file in self.session_dir.glob("*.json"):
            try:
                with open(session_file, "r") as f:
                    data = json.load(f)
                
                expires_at = data.get("metadata", {}).get("expires_at")
                if expires_at:
                    expires = datetime.fromisoformat(expires_at)
                    if datetime.utcnow() > expires:
                        session_file.unlink()
                        removed += 1
                        
            except Exception:
                continue
        
        if removed:
            logger.info("sessions_cleaned", removed=removed)
        
        return removed


# Singleton instance
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the singleton session manager."""
    global _session_manager
    
    if _session_manager is None:
        _session_manager = SessionManager()
    
    return _session_manager


def session_id_from_url(url: str) -> str:
    """
    Extract a session ID from a URL.
    
    Uses the domain as the session ID, so all tasks on the same
    site share cookies/sessions.
    
    Args:
        url: Full URL
        
    Returns:
        Session ID (domain name)
    """
    from urllib.parse import urlparse
    
    parsed = urlparse(url)
    # Remove www. prefix for consistency
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    
    return domain
