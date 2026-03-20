"""Daily salt rotation for session ID hashing.

Rotates salt every 24 hours to prevent cross-day user tracking.
"""

import hashlib
import secrets
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import structlog

logger = structlog.get_logger()


class DailySalt:
    """Manages daily-rotating salt for session ID generation.
    
    The salt rotates at midnight UTC. Previous day's salt is retained
    briefly to handle requests that span the rotation boundary.
    """
    
    def __init__(self, rotation_hour_utc: int = 0):
        """Initialize salt manager.
        
        Args:
            rotation_hour_utc: Hour (0-23) when salt rotates. Default: midnight.
        """
        self._rotation_hour = rotation_hour_utc
        self._current_salt: Optional[str] = None
        self._previous_salt: Optional[str] = None
        self._current_date: Optional[str] = None
        self._lock = threading.Lock()
        
        # Initialize on first access
        self._ensure_current()
    
    def _generate_salt(self) -> str:
        """Generate a cryptographically secure random salt."""
        return secrets.token_hex(32)  # 256-bit salt
    
    def _get_rotation_date(self) -> str:
        """Get current rotation date string (YYYY-MM-DD)."""
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%d")
    
    def _ensure_current(self) -> None:
        """Ensure salt is current, rotating if necessary."""
        current_date = self._get_rotation_date()
        
        with self._lock:
            if self._current_date != current_date:
                # Rotate salt
                self._previous_salt = self._current_salt
                self._current_salt = self._generate_salt()
                old_date = self._current_date
                self._current_date = current_date
                
                logger.info(
                    "Salt rotated",
                    old_date=old_date,
                    new_date=current_date,
                )
    
    @property
    def current(self) -> str:
        """Get current salt, rotating if necessary."""
        self._ensure_current()
        return self._current_salt
    
    @property
    def previous(self) -> Optional[str]:
        """Get previous salt (for boundary handling)."""
        self._ensure_current()
        return self._previous_salt
    
    def hash_with_salt(self, value: str) -> str:
        """Hash a value with current salt.
        
        Args:
            value: Value to hash (e.g., IP address)
            
        Returns:
            SHA-256 hash of value + salt
        """
        self._ensure_current()
        combined = f"{value}:{self._current_salt}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def verify_hash(self, value: str, hash_to_check: str) -> bool:
        """Verify a hash against current or previous salt.
        
        Used for boundary handling when salt just rotated.
        
        Args:
            value: Original value
            hash_to_check: Hash to verify
            
        Returns:
            True if hash matches current or previous salt
        """
        self._ensure_current()
        
        # Check current salt
        current_hash = hashlib.sha256(f"{value}:{self._current_salt}".encode()).hexdigest()
        if current_hash == hash_to_check:
            return True
        
        # Check previous salt (boundary case)
        if self._previous_salt:
            previous_hash = hashlib.sha256(f"{value}:{self._previous_salt}".encode()).hexdigest()
            if previous_hash == hash_to_check:
                return True
        
        return False
    
    def force_rotate(self) -> None:
        """Force immediate salt rotation. For testing only."""
        with self._lock:
            self._previous_salt = self._current_salt
            self._current_salt = self._generate_salt()
            logger.warning("Salt force-rotated (testing)")


# Module-level singleton
_daily_salt: Optional[DailySalt] = None
_salt_lock = threading.Lock()


def get_daily_salt() -> DailySalt:
    """Get or create the daily salt singleton."""
    global _daily_salt
    
    if _daily_salt is None:
        with _salt_lock:
            if _daily_salt is None:
                _daily_salt = DailySalt()
    
    return _daily_salt


def generate_session_id(client_ip: str) -> str:
    """Generate a session ID from client IP.
    
    Args:
        client_ip: Client's IP address
        
    Returns:
        Hashed session ID (cannot be reversed to IP)
    """
    salt = get_daily_salt()
    return salt.hash_with_salt(client_ip)
