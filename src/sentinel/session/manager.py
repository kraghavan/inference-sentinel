"""Session manager with state machine and TTL.

Implements the "one-way trapdoor" pattern:
- Sessions start as CLOUD_ELIGIBLE
- Once Tier 2/3 PII is detected, state flips to LOCAL_LOCKED
- LOCAL_LOCKED is permanent for that session (no going back)
- Sessions expire after TTL of inactivity
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Literal, Optional

import structlog

try:
    from cachetools import TTLCache
except ImportError:
    TTLCache = None  # Will raise if session feature enabled without cachetools

from sentinel.session.salt import generate_session_id

logger = structlog.get_logger()


class SessionState(str, Enum):
    """Session routing state."""
    
    CLOUD_ELIGIBLE = "cloud_eligible"  # Can route to cloud
    LOCAL_LOCKED = "local_locked"      # Must route to local (PII detected)


@dataclass
class SessionInfo:
    """Session metadata and state."""
    
    session_id: str
    state: SessionState = SessionState.CLOUD_ELIGIBLE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Backend stickiness
    cloud_backend: Optional[str] = None   # "anthropic" or "google"
    local_backend: Optional[str] = None   # "gemma" or "mistral"
    
    # Metrics
    request_count: int = 0
    local_locked_at: Optional[datetime] = None
    lock_trigger_tier: Optional[int] = None
    lock_trigger_entities: list = field(default_factory=list)
    
    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)
        self.request_count += 1
    
    def lock_to_local(self, tier: int, entities: list) -> None:
        """Lock session to local routing (one-way trapdoor)."""
        if self.state == SessionState.LOCAL_LOCKED:
            return  # Already locked
        
        self.state = SessionState.LOCAL_LOCKED
        self.local_locked_at = datetime.now(timezone.utc)
        self.lock_trigger_tier = tier
        self.lock_trigger_entities = entities.copy()
        
        logger.info(
            "Session locked to local",
            session_id=self.session_id[:16] + "...",
            trigger_tier=tier,
            trigger_entities=entities,
        )
    
    @property
    def is_local_locked(self) -> bool:
        """Check if session is locked to local routing."""
        return self.state == SessionState.LOCAL_LOCKED
    
    def set_cloud_backend(self, backend: str) -> None:
        """Set sticky cloud backend for this session."""
        if self.cloud_backend is None:
            self.cloud_backend = backend
            logger.debug(
                "Session cloud backend set",
                session_id=self.session_id[:16] + "...",
                backend=backend,
            )
    
    def set_local_backend(self, backend: str) -> None:
        """Set sticky local backend for this session."""
        if self.local_backend is None:
            self.local_backend = backend
            logger.debug(
                "Session local backend set",
                session_id=self.session_id[:16] + "...",
                backend=backend,
            )


class SessionManager:
    """Manages session state with TTL expiration.
    
    Async-safe session storage using TTLCache with asyncio.Lock.
    Sessions automatically expire after `ttl_seconds` of inactivity.
    
    Use async methods (get_or_create_session_async, etc.) from FastAPI routes.
    Use sync methods (get_or_create_session, etc.) from tests or sync code.
    """
    
    def __init__(
        self,
        ttl_seconds: int = 900,  # 15 minutes default
        max_sessions: int = 10000,
        lock_threshold_tier: int = 2,  # Tier 2+ triggers lock
        buffer_max_turns: int = 5,
        buffer_max_chars: int = 4000,
    ):
        """Initialize session manager.
        
        Args:
            ttl_seconds: Session TTL in seconds (default: 15 min)
            max_sessions: Maximum concurrent sessions
            lock_threshold_tier: Minimum tier to trigger LOCAL_LOCKED
            buffer_max_turns: Max turns per buffer
            buffer_max_chars: Max chars per buffer (~1000 tokens default)
        """
        if TTLCache is None:
            raise ImportError(
                "cachetools required for session management. "
                "Install with: pip install cachetools"
            )
        
        self._ttl_seconds = ttl_seconds
        self._lock_threshold_tier = lock_threshold_tier
        self._buffer_max_turns = buffer_max_turns
        self._buffer_max_chars = buffer_max_chars
        
        self._sessions: TTLCache = TTLCache(maxsize=max_sessions, ttl=ttl_seconds)
        self._buffers: TTLCache = TTLCache(maxsize=max_sessions, ttl=ttl_seconds)
        
        # Dual locks: asyncio for FastAPI, threading for sync code/tests
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        
        # Metrics
        self._total_sessions_created = 0
        self._total_sessions_locked = 0
        self._total_sessions_expired = 0
        
        logger.info(
            "Session manager initialized",
            ttl_seconds=ttl_seconds,
            max_sessions=max_sessions,
            lock_threshold_tier=lock_threshold_tier,
            buffer_max_turns=buffer_max_turns,
            buffer_max_chars=buffer_max_chars,
        )
    
    # ==================== ASYNC METHODS (for FastAPI) ====================
    
    async def get_or_create_session_async(self, client_ip: str) -> SessionInfo:
        """Get existing session or create new one (async).
        
        Args:
            client_ip: Client's IP address
            
        Returns:
            SessionInfo for this client
        """
        session_id = generate_session_id(client_ip)
        
        async with self._async_lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session
            
            # Create new session
            session = SessionInfo(session_id=session_id)
            self._sessions[session_id] = session
            self._total_sessions_created += 1
            
            logger.info(
                "Session created",
                session_id=session_id[:16] + "...",
            )
            
            return session
    
    async def get_session_async(self, client_ip: str) -> Optional[SessionInfo]:
        """Get existing session without creating (async).
        
        Args:
            client_ip: Client's IP address
            
        Returns:
            SessionInfo if exists, None otherwise
        """
        session_id = generate_session_id(client_ip)
        
        async with self._async_lock:
            return self._sessions.get(session_id)
    
    async def update_session_state_async(
        self,
        client_ip: str,
        tier: int,
        entities: list,
    ) -> SessionInfo:
        """Update session state based on classification result (async).
        
        Applies the one-way trapdoor: if tier >= threshold, locks to local.
        
        Args:
            client_ip: Client's IP address
            tier: Classification tier (0-3)
            entities: Detected entity types
            
        Returns:
            Updated SessionInfo
        """
        session = await self.get_or_create_session_async(client_ip)
        
        # Check if we should lock
        if tier >= self._lock_threshold_tier and not session.is_local_locked:
            async with self._async_lock:
                session.lock_to_local(tier, entities)
                self._total_sessions_locked += 1
        
        return session
    
    async def set_backend_async(
        self,
        client_ip: str,
        route: Literal["local", "cloud"],
        backend_name: str,
    ) -> None:
        """Set sticky backend for a session (async).
        
        Args:
            client_ip: Client's IP address
            route: "local" or "cloud"
            backend_name: Backend identifier
        """
        session = await self.get_session_async(client_ip)
        if session is None:
            return
        
        async with self._async_lock:
            if route == "cloud":
                session.set_cloud_backend(backend_name)
            else:
                session.set_local_backend(backend_name)
    
    async def get_sticky_backend_async(
        self,
        client_ip: str,
        route: Literal["local", "cloud"],
    ) -> Optional[str]:
        """Get sticky backend for a session (async).
        
        Args:
            client_ip: Client's IP address
            route: "local" or "cloud"
            
        Returns:
            Backend name if set, None for round-robin
        """
        session = await self.get_session_async(client_ip)
        if session is None:
            return None
        
        if route == "cloud":
            return session.cloud_backend
        else:
            return session.local_backend
    
    async def should_route_local_async(self, client_ip: str) -> bool:
        """Check if session is locked to local routing (async).
        
        Args:
            client_ip: Client's IP address
            
        Returns:
            True if session is LOCAL_LOCKED
        """
        session = await self.get_session_async(client_ip)
        if session is None:
            return False
        return session.is_local_locked
    
    # ==================== BUFFER METHODS ====================
    
    def _get_or_create_buffer(self, session_id: str):
        """Get or create buffer for a session (internal, requires lock held)."""
        # Import here to avoid circular import
        from sentinel.session.buffer import RollingBuffer
        
        if session_id not in self._buffers:
            self._buffers[session_id] = RollingBuffer(
                max_turns=self._buffer_max_turns,
                max_chars=self._buffer_max_chars,
            )
        return self._buffers[session_id]
    
    async def add_to_buffer_async(
        self,
        client_ip: str,
        role: str,
        content: str,
        tier: int = 0,
        scrubbed_content: str = None,
    ) -> None:
        """Add message to session buffer (async).
        
        Args:
            client_ip: Client's IP address
            role: "user" or "assistant"
            content: Message content
            tier: Classification tier
            scrubbed_content: Pre-scrubbed content (optional)
        """
        session_id = generate_session_id(client_ip)
        
        async with self._async_lock:
            buffer = self._get_or_create_buffer(session_id)
            buffer.add(role, content, tier, scrubbed_content)
    
    async def get_buffer_async(self, client_ip: str):
        """Get buffer for a session (async).
        
        Args:
            client_ip: Client's IP address
            
        Returns:
            RollingBuffer or None
        """
        session_id = generate_session_id(client_ip)
        
        async with self._async_lock:
            return self._buffers.get(session_id)
    
    async def get_handoff_context_async(self, client_ip: str) -> str:
        """Get formatted context for local model handoff (async).
        
        Args:
            client_ip: Client's IP address
            
        Returns:
            Formatted handoff string (empty if no buffer)
        """
        buffer = await self.get_buffer_async(client_ip)
        if buffer is None:
            return ""
        return buffer.format_for_handoff()
    
    def add_to_buffer(
        self,
        client_ip: str,
        role: str,
        content: str,
        tier: int = 0,
        scrubbed_content: str = None,
    ) -> None:
        """Add message to session buffer (sync).
        
        Args:
            client_ip: Client's IP address
            role: "user" or "assistant"
            content: Message content
            tier: Classification tier
            scrubbed_content: Pre-scrubbed content (optional)
        """
        session_id = generate_session_id(client_ip)
        
        with self._sync_lock:
            buffer = self._get_or_create_buffer(session_id)
            buffer.add(role, content, tier, scrubbed_content)
    
    def get_buffer(self, client_ip: str):
        """Get buffer for a session (sync).
        
        Args:
            client_ip: Client's IP address
            
        Returns:
            RollingBuffer or None
        """
        session_id = generate_session_id(client_ip)
        
        with self._sync_lock:
            return self._buffers.get(session_id)

    # ==================== SYNC METHODS (for tests) ====================
    
    def get_or_create_session(self, client_ip: str) -> SessionInfo:
        """Get existing session or create new one (sync).
        
        Args:
            client_ip: Client's IP address
            
        Returns:
            SessionInfo for this client
        """
        session_id = generate_session_id(client_ip)
        
        with self._sync_lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session
            
            # Create new session
            session = SessionInfo(session_id=session_id)
            self._sessions[session_id] = session
            self._total_sessions_created += 1
            
            logger.info(
                "Session created",
                session_id=session_id[:16] + "...",
            )
            
            return session
    
    def get_session(self, client_ip: str) -> Optional[SessionInfo]:
        """Get existing session without creating.
        
        Args:
            client_ip: Client's IP address
            
        Returns:
            SessionInfo if exists, None otherwise
        """
        session_id = generate_session_id(client_ip)
        
        with self._sync_lock:
            return self._sessions.get(session_id)
    
    def update_session_state(
        self,
        client_ip: str,
        tier: int,
        entities: list,
    ) -> SessionInfo:
        """Update session state based on classification result.
        
        Applies the one-way trapdoor: if tier >= threshold, locks to local.
        
        Args:
            client_ip: Client's IP address
            tier: Classification tier (0-3)
            entities: Detected entity types
            
        Returns:
            Updated SessionInfo
        """
        session = self.get_or_create_session(client_ip)
        
        # Check if we should lock
        if tier >= self._lock_threshold_tier and not session.is_local_locked:
            with self._sync_lock:
                session.lock_to_local(tier, entities)
                self._total_sessions_locked += 1
        
        return session
    
    def set_backend(
        self,
        client_ip: str,
        route: Literal["local", "cloud"],
        backend_name: str,
    ) -> None:
        """Set sticky backend for a session.
        
        Args:
            client_ip: Client's IP address
            route: "local" or "cloud"
            backend_name: Backend identifier
        """
        session = self.get_session(client_ip)
        if session is None:
            return
        
        with self._sync_lock:
            if route == "cloud":
                session.set_cloud_backend(backend_name)
            else:
                session.set_local_backend(backend_name)
    
    def get_sticky_backend(
        self,
        client_ip: str,
        route: Literal["local", "cloud"],
    ) -> Optional[str]:
        """Get sticky backend for a session.
        
        Args:
            client_ip: Client's IP address
            route: "local" or "cloud"
            
        Returns:
            Backend name if set, None for round-robin
        """
        session = self.get_session(client_ip)
        if session is None:
            return None
        
        if route == "cloud":
            return session.cloud_backend
        else:
            return session.local_backend
    
    def should_route_local(self, client_ip: str) -> bool:
        """Check if session is locked to local routing.
        
        Args:
            client_ip: Client's IP address
            
        Returns:
            True if session is LOCAL_LOCKED
        """
        session = self.get_session(client_ip)
        if session is None:
            return False
        return session.is_local_locked
    
    def purge_session(self, client_ip: str) -> bool:
        """Manually purge a session.
        
        Args:
            client_ip: Client's IP address
            
        Returns:
            True if session was purged
        """
        session_id = generate_session_id(client_ip)
        
        with self._sync_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(
                    "Session purged",
                    session_id=session_id[:16] + "...",
                )
                return True
        return False
    
    def get_metrics(self) -> dict:
        """Get session manager metrics."""
        with self._sync_lock:
            active_sessions = len(self._sessions)
            locked_sessions = sum(
                1 for s in self._sessions.values()
                if s.is_local_locked
            )
        
        return {
            "active_sessions": active_sessions,
            "locked_sessions": locked_sessions,
            "total_created": self._total_sessions_created,
            "total_locked": self._total_sessions_locked,
            "ttl_seconds": self._ttl_seconds,
            "lock_threshold_tier": self._lock_threshold_tier,
        }
    
    def clear_all(self) -> int:
        """Clear all sessions. For testing only.
        
        Returns:
            Number of sessions cleared
        """
        with self._sync_lock:
            count = len(self._sessions)
            self._sessions.clear()
            logger.warning("All sessions cleared", count=count)
            return count


# Module-level singleton
_session_manager: Optional[SessionManager] = None
_manager_lock = threading.Lock()


def get_session_manager() -> Optional[SessionManager]:
    """Get the session manager singleton.
    
    Returns None if session management is not enabled.
    """
    return _session_manager


def configure_session_manager(
    enabled: bool = False,
    ttl_seconds: int = 900,
    max_sessions: int = 10000,
    lock_threshold_tier: int = 2,
    buffer_max_turns: int = 5,
    buffer_max_chars: int = 4000,
) -> Optional[SessionManager]:
    """Configure and enable session management.
    
    Args:
        enabled: Whether to enable session management
        ttl_seconds: Session TTL
        max_sessions: Max concurrent sessions
        lock_threshold_tier: Tier threshold for LOCAL_LOCKED
        buffer_max_turns: Max turns in rolling buffer
        buffer_max_chars: Max chars in rolling buffer
        
    Returns:
        SessionManager if enabled, None otherwise
    """
    global _session_manager
    
    if not enabled:
        _session_manager = None
        logger.info("Session management disabled")
        return None
    
    with _manager_lock:
        _session_manager = SessionManager(
            ttl_seconds=ttl_seconds,
            max_sessions=max_sessions,
            lock_threshold_tier=lock_threshold_tier,
            buffer_max_turns=buffer_max_turns,
            buffer_max_chars=buffer_max_chars,
        )
    
    return _session_manager
