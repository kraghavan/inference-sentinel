"""Session management for inference-sentinel.

Implements the "one-way trapdoor" pattern for privacy-aware routing:
- Sessions start as CLOUD_ELIGIBLE
- Once Tier 2/3 PII is detected, state flips to LOCAL_LOCKED
- LOCAL_LOCKED is permanent for that session
- Sessions expire after TTL of inactivity

Usage:
    from sentinel.session import (
        get_session_manager,
        configure_session_manager,
        SessionState,
    )
    
    # Configure at startup
    manager = configure_session_manager(
        enabled=True,
        ttl_seconds=900,
        lock_threshold_tier=2,
    )
    
    # In request handler
    session = manager.get_or_create_session(client_ip)
    if session.is_local_locked:
        # Must route to local
        ...
"""

from sentinel.session.buffer import (
    BufferEntry,
    RollingBuffer,
    create_handoff_system_prompt,
    scrub_content_for_buffer,
)
from sentinel.session.manager import (
    SessionInfo,
    SessionManager,
    SessionState,
    configure_session_manager,
    get_session_manager,
)
from sentinel.session.salt import (
    DailySalt,
    generate_session_id,
    get_daily_salt,
)

__all__ = [
    # State enum
    "SessionState",
    # Session info
    "SessionInfo",
    # Manager
    "SessionManager",
    "get_session_manager",
    "configure_session_manager",
    # Salt
    "DailySalt",
    "get_daily_salt",
    "generate_session_id",
    # Buffer
    "BufferEntry",
    "RollingBuffer",
    "create_handoff_system_prompt",
    "scrub_content_for_buffer",
]
