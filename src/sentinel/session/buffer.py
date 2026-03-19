"""Rolling buffer for session context with dual bounding.

Stores recent interactions for context injection during local handoff.
Bounded by BOTH turn count AND total character length to prevent
context window blowup on local models.
"""

import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import structlog

logger = structlog.get_logger()


# Approximate chars per token (conservative estimate)
CHARS_PER_TOKEN = 4


@dataclass
class BufferEntry:
    """Single interaction in the buffer."""
    
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tier: int = 0  # Classification tier when added
    scrubbed: bool = False  # Whether content was scrubbed before storage
    char_count: int = field(init=False)
    
    def __post_init__(self):
        self.char_count = len(self.content)
    
    def to_dict(self) -> dict:
        """Convert to message dict format."""
        return {
            "role": self.role,
            "content": self.content,
        }


class RollingBuffer:
    """Rolling buffer with dual bounding.
    
    Bounded by:
    1. Max turns (number of user+assistant pairs)
    2. Max total characters (prevents massive payloads)
    
    When either limit is exceeded, oldest entries are evicted.
    """
    
    def __init__(
        self,
        max_turns: int = 5,
        max_chars: int = 4000,  # ~1000 tokens
        scrub_before_store: bool = True,
    ):
        """Initialize buffer.
        
        Args:
            max_turns: Maximum number of turns to keep
            max_chars: Maximum total characters across all entries
            scrub_before_store: Whether to scrub content through classifier
        """
        self._max_turns = max_turns
        self._max_chars = max_chars
        self._scrub_before_store = scrub_before_store
        
        self._entries: List[BufferEntry] = []
        self._total_chars = 0
        self._lock = threading.Lock()
        
        # Metrics
        self._total_added = 0
        self._total_evicted_by_turns = 0
        self._total_evicted_by_chars = 0
        self._total_scrubbed = 0
    
    @property
    def turn_count(self) -> int:
        """Number of user messages (turns) in buffer."""
        return sum(1 for e in self._entries if e.role == "user")
    
    @property
    def total_chars(self) -> int:
        """Total characters in buffer."""
        return self._total_chars
    
    @property
    def approx_tokens(self) -> int:
        """Approximate token count."""
        return self._total_chars // CHARS_PER_TOKEN
    
    def add(
        self,
        role: str,
        content: str,
        tier: int = 0,
        scrubbed_content: Optional[str] = None,
    ) -> None:
        """Add an entry to the buffer.
        
        Args:
            role: "user" or "assistant"
            content: Message content
            tier: Classification tier (for logging)
            scrubbed_content: Pre-scrubbed content (if scrubbing was done externally)
        """
        # Use scrubbed content if provided
        final_content = scrubbed_content if scrubbed_content is not None else content
        was_scrubbed = scrubbed_content is not None
        
        entry = BufferEntry(
            role=role,
            content=final_content,
            tier=tier,
            scrubbed=was_scrubbed,
        )
        
        with self._lock:
            # Add entry
            self._entries.append(entry)
            self._total_chars += entry.char_count
            self._total_added += 1
            
            if was_scrubbed:
                self._total_scrubbed += 1
            
            # Enforce character limit (evict oldest until under limit)
            while self._total_chars > self._max_chars and len(self._entries) > 1:
                evicted = self._entries.pop(0)
                self._total_chars -= evicted.char_count
                self._total_evicted_by_chars += 1
                logger.debug(
                    "Buffer entry evicted (char limit)",
                    evicted_chars=evicted.char_count,
                    total_chars=self._total_chars,
                )
            
            # Enforce turn limit (evict oldest pairs until under limit)
            while self.turn_count > self._max_turns and len(self._entries) > 1:
                evicted = self._entries.pop(0)
                self._total_chars -= evicted.char_count
                self._total_evicted_by_turns += 1
                logger.debug(
                    "Buffer entry evicted (turn limit)",
                    turn_count=self.turn_count,
                )
    
    def get_entries(self) -> List[BufferEntry]:
        """Get all entries (copy)."""
        with self._lock:
            return list(self._entries)
    
    def get_messages(self) -> List[dict]:
        """Get entries as message dicts for LLM."""
        with self._lock:
            return [e.to_dict() for e in self._entries]
    
    def format_for_handoff(self) -> str:
        """Format buffer for injection into local model.
        
        Uses clear XML tags to delineate historical context
        from current request (prevents context confusion).
        
        Returns:
            Formatted string for system prompt injection
        """
        if not self._entries:
            return ""
        
        with self._lock:
            lines = []
            for entry in self._entries:
                role_tag = "user_message" if entry.role == "user" else "assistant_response"
                lines.append(f"<{role_tag}>{entry.content}</{role_tag}>")
            
            return "\n".join(lines)
    
    def clear(self) -> int:
        """Clear all entries.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._total_chars = 0
            return count
    
    def get_metrics(self) -> dict:
        """Get buffer metrics."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "turns": self.turn_count,
                "total_chars": self._total_chars,
                "approx_tokens": self.approx_tokens,
                "max_turns": self._max_turns,
                "max_chars": self._max_chars,
                "total_added": self._total_added,
                "evicted_by_turns": self._total_evicted_by_turns,
                "evicted_by_chars": self._total_evicted_by_chars,
                "total_scrubbed": self._total_scrubbed,
            }


def create_handoff_system_prompt(
    buffer: RollingBuffer,
    capability_guardrail: bool = True,
) -> str:
    """Create system prompt for local model handoff.
    
    Clearly delineates:
    1. Capability restrictions (what the model cannot do)
    2. Historical context from the session
    3. Instructions for handling the current request
    
    Args:
        buffer: Rolling buffer with session history
        capability_guardrail: Whether to inject capability restrictions
        
    Returns:
        Complete system prompt for handoff
    """
    parts = []
    
    # Capability guardrail
    if capability_guardrail:
        parts.append("""<capability_restrictions>
You are operating in a SECURE LOCAL environment with the following restrictions:
- You have NO access to the internet or web browsing
- You have NO access to external APIs or services
- You have NO access to databases or file systems
- You have NO ability to execute code or run tools
- You CANNOT make HTTP requests or fetch external data
- You MUST answer based solely on your training knowledge and the conversation context provided

If the user asks for anything requiring external access, politely explain that you cannot perform that action in this secure environment.
</capability_restrictions>
""")
    
    # Historical context
    history = buffer.format_for_handoff()
    if history:
        parts.append(f"""<historical_context>
The following is the conversation history from this session. This context is provided so you can maintain continuity. The user may reference previous messages.

{history}
</historical_context>
""")
    
    # Current request instructions
    parts.append("""<instructions>
Respond to the user's current message below. Maintain the conversational context from the history if provided. Be helpful, accurate, and concise.
</instructions>
""")
    
    return "\n".join(parts)


def scrub_content_for_buffer(
    content: str,
    detected_entities: List[dict],
) -> str:
    """Scrub sensitive entities from content before buffering.
    
    Replaces detected PII with redaction markers.
    
    Args:
        content: Original content
        detected_entities: List of detected entity dicts with 'value', 'type'
        
    Returns:
        Scrubbed content
    """
    if not detected_entities:
        return content
    
    scrubbed = content
    
    # Sort by length descending to handle overlapping matches
    sorted_entities = sorted(
        detected_entities,
        key=lambda e: len(e.get("value", "")),
        reverse=True,
    )
    
    for entity in sorted_entities:
        value = entity.get("value", "")
        entity_type = entity.get("type", "REDACTED")
        
        if value and value in scrubbed:
            # Create deterministic placeholder (hash-based for consistency)
            hash_suffix = hashlib.sha256(value.encode()).hexdigest()[:6]
            placeholder = f"[{entity_type.upper()}_{hash_suffix}]"
            scrubbed = scrubbed.replace(value, placeholder)
    
    return scrubbed
