"""Unit tests for session management."""

import time
from unittest.mock import patch

import pytest


class TestDailySalt:
    """Tests for daily salt rotation."""
    
    def test_salt_generation(self):
        """Salt should be generated on first access."""
        from sentinel.session.salt import DailySalt
        
        salt = DailySalt()
        assert salt.current is not None
        assert len(salt.current) == 64  # 256-bit hex = 64 chars
    
    def test_salt_consistency(self):
        """Same salt should be returned within same day."""
        from sentinel.session.salt import DailySalt
        
        salt = DailySalt()
        first = salt.current
        second = salt.current
        assert first == second
    
    def test_hash_with_salt(self):
        """Hash should be deterministic for same input."""
        from sentinel.session.salt import DailySalt
        
        salt = DailySalt()
        hash1 = salt.hash_with_salt("192.168.1.1")
        hash2 = salt.hash_with_salt("192.168.1.1")
        assert hash1 == hash2
    
    def test_different_inputs_different_hashes(self):
        """Different IPs should produce different hashes."""
        from sentinel.session.salt import DailySalt
        
        salt = DailySalt()
        hash1 = salt.hash_with_salt("192.168.1.1")
        hash2 = salt.hash_with_salt("192.168.1.2")
        assert hash1 != hash2
    
    def test_force_rotate(self):
        """Force rotation should change salt."""
        from sentinel.session.salt import DailySalt
        
        salt = DailySalt()
        original = salt.current
        salt.force_rotate()
        rotated = salt.current
        
        assert original != rotated
        assert salt.previous == original
    
    def test_verify_hash_current(self):
        """Should verify hash against current salt."""
        from sentinel.session.salt import DailySalt
        
        salt = DailySalt()
        ip = "10.0.0.1"
        hashed = salt.hash_with_salt(ip)
        
        assert salt.verify_hash(ip, hashed) is True
    
    def test_verify_hash_previous(self):
        """Should verify hash against previous salt after rotation."""
        from sentinel.session.salt import DailySalt
        
        salt = DailySalt()
        ip = "10.0.0.1"
        hashed = salt.hash_with_salt(ip)
        
        # Rotate
        salt.force_rotate()
        
        # Should still verify against previous
        assert salt.verify_hash(ip, hashed) is True
    
    def test_generate_session_id(self):
        """Module function should generate session ID."""
        from sentinel.session.salt import generate_session_id
        
        session_id = generate_session_id("192.168.1.100")
        assert session_id is not None
        assert len(session_id) == 64


class TestSessionState:
    """Tests for session state enum."""
    
    def test_states_exist(self):
        """Should have expected states."""
        from sentinel.session import SessionState
        
        assert SessionState.CLOUD_ELIGIBLE.value == "cloud_eligible"
        assert SessionState.LOCAL_LOCKED.value == "local_locked"


class TestSessionInfo:
    """Tests for SessionInfo dataclass."""
    
    def test_default_state(self):
        """New session should be CLOUD_ELIGIBLE."""
        from sentinel.session.manager import SessionInfo, SessionState
        
        session = SessionInfo(session_id="test123")
        assert session.state == SessionState.CLOUD_ELIGIBLE
        assert session.is_local_locked is False
    
    def test_lock_to_local(self):
        """Session should lock to local and not unlock."""
        from sentinel.session.manager import SessionInfo, SessionState
        
        session = SessionInfo(session_id="test123")
        session.lock_to_local(tier=3, entities=["SSN"])
        
        assert session.state == SessionState.LOCAL_LOCKED
        assert session.is_local_locked is True
        assert session.lock_trigger_tier == 3
        assert session.lock_trigger_entities == ["SSN"]
        assert session.local_locked_at is not None
    
    def test_lock_is_idempotent(self):
        """Locking again should not change original lock info."""
        from sentinel.session.manager import SessionInfo
        
        session = SessionInfo(session_id="test123")
        session.lock_to_local(tier=2, entities=["EMAIL"])
        original_time = session.local_locked_at
        
        # Try to lock again with different tier
        session.lock_to_local(tier=3, entities=["SSN"])
        
        # Should keep original lock info
        assert session.lock_trigger_tier == 2
        assert session.lock_trigger_entities == ["EMAIL"]
        assert session.local_locked_at == original_time
    
    def test_touch_updates_activity(self):
        """Touch should update last_activity and increment count."""
        from sentinel.session.manager import SessionInfo
        
        session = SessionInfo(session_id="test123")
        original_time = session.last_activity
        original_count = session.request_count
        
        time.sleep(0.01)
        session.touch()
        
        assert session.last_activity > original_time
        assert session.request_count == original_count + 1
    
    def test_backend_stickiness(self):
        """Should set and preserve backend stickiness."""
        from sentinel.session.manager import SessionInfo
        
        session = SessionInfo(session_id="test123")
        
        # First set should work
        session.set_cloud_backend("anthropic")
        assert session.cloud_backend == "anthropic"
        
        # Second set should be ignored
        session.set_cloud_backend("google")
        assert session.cloud_backend == "anthropic"
        
        # Same for local
        session.set_local_backend("gemma")
        assert session.local_backend == "gemma"
        
        session.set_local_backend("mistral")
        assert session.local_backend == "gemma"


class TestSessionManager:
    """Tests for SessionManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh session manager."""
        from sentinel.session.manager import SessionManager
        return SessionManager(ttl_seconds=60, max_sessions=100, lock_threshold_tier=2)
    
    def test_create_session(self, manager):
        """Should create new session."""
        session = manager.get_or_create_session("192.168.1.1")
        
        assert session is not None
        assert session.session_id is not None
        assert session.is_local_locked is False
    
    def test_get_existing_session(self, manager):
        """Should return same session for same IP."""
        session1 = manager.get_or_create_session("192.168.1.1")
        session2 = manager.get_or_create_session("192.168.1.1")
        
        assert session1.session_id == session2.session_id
    
    def test_different_ips_different_sessions(self, manager):
        """Different IPs should get different sessions."""
        session1 = manager.get_or_create_session("192.168.1.1")
        session2 = manager.get_or_create_session("192.168.1.2")
        
        assert session1.session_id != session2.session_id
    
    def test_update_session_state_locks_on_tier(self, manager):
        """Should lock session when tier >= threshold."""
        # Tier 1 should not lock (threshold is 2)
        session = manager.update_session_state("192.168.1.1", tier=1, entities=["internal_url"])
        assert session.is_local_locked is False
        
        # Tier 2 should lock
        session = manager.update_session_state("192.168.1.1", tier=2, entities=["EMAIL"])
        assert session.is_local_locked is True
    
    def test_should_route_local(self, manager):
        """Should report routing correctly."""
        # New session - should route cloud
        assert manager.should_route_local("192.168.1.1") is False
        
        # After locking
        manager.update_session_state("192.168.1.1", tier=3, entities=["SSN"])
        assert manager.should_route_local("192.168.1.1") is True
    
    def test_backend_stickiness(self, manager):
        """Should track sticky backends."""
        ip = "192.168.1.1"
        manager.get_or_create_session(ip)
        
        # Initially no sticky backend
        assert manager.get_sticky_backend(ip, "cloud") is None
        
        # Set and retrieve
        manager.set_backend(ip, "cloud", "anthropic")
        assert manager.get_sticky_backend(ip, "cloud") == "anthropic"
        
        # Local should still be None
        assert manager.get_sticky_backend(ip, "local") is None
    
    def test_purge_session(self, manager):
        """Should purge session."""
        ip = "192.168.1.1"
        manager.get_or_create_session(ip)
        
        assert manager.get_session(ip) is not None
        
        result = manager.purge_session(ip)
        assert result is True
        assert manager.get_session(ip) is None
    
    def test_get_metrics(self, manager):
        """Should return correct metrics."""
        manager.get_or_create_session("192.168.1.1")
        manager.get_or_create_session("192.168.1.2")
        manager.update_session_state("192.168.1.1", tier=3, entities=["SSN"])
        
        metrics = manager.get_metrics()
        
        assert metrics["active_sessions"] == 2
        assert metrics["locked_sessions"] == 1
        assert metrics["total_created"] == 2
        assert metrics["total_locked"] == 1
    
    def test_clear_all(self, manager):
        """Should clear all sessions."""
        manager.get_or_create_session("192.168.1.1")
        manager.get_or_create_session("192.168.1.2")
        
        count = manager.clear_all()
        
        assert count == 2
        assert manager.get_metrics()["active_sessions"] == 0


class TestSessionConfig:
    """Tests for session configuration."""
    
    def test_default_config(self):
        """Default config should have session disabled."""
        from sentinel.config import get_settings
        
        # Clear cache
        get_settings.cache_clear()
        
        settings = get_settings()
        assert settings.session.enabled is False
        assert settings.session.ttl_seconds == 900
        assert settings.session.lock_threshold_tier == 2
    
    def test_config_from_env(self, monkeypatch):
        """Should load config from environment."""
        from sentinel.config import get_settings
        
        monkeypatch.setenv("SENTINEL_SESSION__ENABLED", "true")
        monkeypatch.setenv("SENTINEL_SESSION__TTL_SECONDS", "1800")
        monkeypatch.setenv("SENTINEL_SESSION__LOCK_THRESHOLD_TIER", "3")
        
        # Clear cache
        get_settings.cache_clear()
        
        settings = get_settings()
        assert settings.session.enabled is True
        assert settings.session.ttl_seconds == 1800
        assert settings.session.lock_threshold_tier == 3
        
        # Clean up
        get_settings.cache_clear()


class TestConfigureSessionManager:
    """Tests for session manager configuration."""
    
    def test_configure_disabled(self):
        """Should return None when disabled."""
        from sentinel.session import configure_session_manager, get_session_manager
        
        result = configure_session_manager(enabled=False)
        assert result is None
        assert get_session_manager() is None
    
    def test_configure_enabled(self):
        """Should create manager when enabled."""
        from sentinel.session import configure_session_manager, get_session_manager
        
        result = configure_session_manager(
            enabled=True,
            ttl_seconds=600,
            max_sessions=500,
            lock_threshold_tier=3,
        )
        
        assert result is not None
        assert get_session_manager() is result
        
        metrics = result.get_metrics()
        assert metrics["ttl_seconds"] == 600
        assert metrics["lock_threshold_tier"] == 3
        
        # Clean up
        configure_session_manager(enabled=False)


class TestRollingBuffer:
    """Tests for rolling buffer with dual bounding."""
    
    def test_add_entry(self):
        """Should add entries to buffer."""
        from sentinel.session.buffer import RollingBuffer
        
        buffer = RollingBuffer(max_turns=5, max_chars=4000)
        buffer.add("user", "Hello world")
        buffer.add("assistant", "Hi there!")
        
        entries = buffer.get_entries()
        assert len(entries) == 2
        assert entries[0].role == "user"
        assert entries[1].role == "assistant"
    
    def test_turn_limit(self):
        """Should evict oldest entries when turn limit exceeded."""
        from sentinel.session.buffer import RollingBuffer
        
        buffer = RollingBuffer(max_turns=2, max_chars=10000)
        
        # Add 3 user turns
        buffer.add("user", "Turn 1")
        buffer.add("assistant", "Response 1")
        buffer.add("user", "Turn 2")
        buffer.add("assistant", "Response 2")
        buffer.add("user", "Turn 3")  # Should evict Turn 1
        
        entries = buffer.get_entries()
        # Should have evicted oldest to stay under 2 turns
        assert buffer.turn_count <= 2
    
    def test_char_limit(self):
        """Should evict oldest entries when char limit exceeded."""
        from sentinel.session.buffer import RollingBuffer
        
        buffer = RollingBuffer(max_turns=100, max_chars=50)
        
        # Add entries that exceed char limit
        buffer.add("user", "A" * 30)
        buffer.add("user", "B" * 30)  # Should evict first
        
        assert buffer.total_chars <= 50
        entries = buffer.get_entries()
        # First entry should have been evicted
        assert entries[0].content.startswith("B") or len(entries) == 1
    
    def test_format_for_handoff(self):
        """Should format buffer with XML tags for handoff."""
        from sentinel.session.buffer import RollingBuffer
        
        buffer = RollingBuffer()
        buffer.add("user", "What's 2+2?")
        buffer.add("assistant", "4")
        
        formatted = buffer.format_for_handoff()
        
        assert "<user_message>" in formatted
        assert "</user_message>" in formatted
        assert "<assistant_response>" in formatted
        assert "</assistant_response>" in formatted
        assert "What's 2+2?" in formatted
        assert "4" in formatted
    
    def test_get_messages(self):
        """Should return messages in LLM format."""
        from sentinel.session.buffer import RollingBuffer
        
        buffer = RollingBuffer()
        buffer.add("user", "Hello")
        buffer.add("assistant", "Hi")
        
        messages = buffer.get_messages()
        
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi"}
    
    def test_metrics(self):
        """Should track buffer metrics."""
        from sentinel.session.buffer import RollingBuffer
        
        buffer = RollingBuffer(max_turns=2, max_chars=100)
        buffer.add("user", "Test message")
        
        metrics = buffer.get_metrics()
        
        assert metrics["entries"] == 1
        assert metrics["turns"] == 1
        assert metrics["total_chars"] == len("Test message")
        assert metrics["total_added"] == 1


class TestScrubContent:
    """Tests for content scrubbing."""
    
    def test_scrub_entities(self):
        """Should replace detected entities with placeholders."""
        from sentinel.session.buffer import scrub_content_for_buffer
        
        content = "My email is john@example.com and SSN is 123-45-6789"
        entities = [
            {"value": "john@example.com", "type": "EMAIL"},
            {"value": "123-45-6789", "type": "SSN"},
        ]
        
        scrubbed = scrub_content_for_buffer(content, entities)
        
        assert "john@example.com" not in scrubbed
        assert "123-45-6789" not in scrubbed
        assert "[EMAIL_" in scrubbed
        assert "[SSN_" in scrubbed
    
    def test_scrub_empty_entities(self):
        """Should return original content when no entities."""
        from sentinel.session.buffer import scrub_content_for_buffer
        
        content = "Just a normal message"
        scrubbed = scrub_content_for_buffer(content, [])
        
        assert scrubbed == content


class TestHandoffSystemPrompt:
    """Tests for handoff system prompt generation."""
    
    def test_create_with_guardrail(self):
        """Should include capability restrictions."""
        from sentinel.session.buffer import RollingBuffer, create_handoff_system_prompt
        
        buffer = RollingBuffer()
        buffer.add("user", "Hello")
        
        prompt = create_handoff_system_prompt(buffer, capability_guardrail=True)
        
        assert "<capability_restrictions>" in prompt
        assert "NO access to the internet" in prompt
        assert "<historical_context>" in prompt
    
    def test_create_without_guardrail(self):
        """Should exclude capability restrictions when disabled."""
        from sentinel.session.buffer import RollingBuffer, create_handoff_system_prompt
        
        buffer = RollingBuffer()
        buffer.add("user", "Hello")
        
        prompt = create_handoff_system_prompt(buffer, capability_guardrail=False)
        
        assert "<capability_restrictions>" not in prompt
        assert "<historical_context>" in prompt
    
    def test_empty_buffer(self):
        """Should handle empty buffer gracefully."""
        from sentinel.session.buffer import RollingBuffer, create_handoff_system_prompt
        
        buffer = RollingBuffer()
        prompt = create_handoff_system_prompt(buffer, capability_guardrail=True)
        
        assert "<capability_restrictions>" in prompt
        assert "<instructions>" in prompt


class TestSessionManagerAsync:
    """Async tests for SessionManager (FastAPI compatibility)."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh session manager."""
        from sentinel.session.manager import SessionManager
        return SessionManager(ttl_seconds=60, max_sessions=100, lock_threshold_tier=2)
    
    @pytest.mark.asyncio
    async def test_async_create_session(self, manager):
        """Should create session asynchronously."""
        session = await manager.get_or_create_session_async("192.168.1.1")
        
        assert session is not None
        assert session.is_local_locked is False
    
    @pytest.mark.asyncio
    async def test_async_get_same_session(self, manager):
        """Should return same session for same IP."""
        session1 = await manager.get_or_create_session_async("192.168.1.1")
        session2 = await manager.get_or_create_session_async("192.168.1.1")
        
        assert session1.session_id == session2.session_id
    
    @pytest.mark.asyncio
    async def test_async_update_locks_session(self, manager):
        """Should lock session on tier >= threshold."""
        session = await manager.update_session_state_async(
            "192.168.1.1", tier=3, entities=["SSN"]
        )
        
        assert session.is_local_locked is True
        
        # Verify routing check
        should_local = await manager.should_route_local_async("192.168.1.1")
        assert should_local is True
    
    @pytest.mark.asyncio
    async def test_async_backend_stickiness(self, manager):
        """Should track sticky backends asynchronously."""
        ip = "192.168.1.1"
        await manager.get_or_create_session_async(ip)
        
        # Set and retrieve
        await manager.set_backend_async(ip, "cloud", "anthropic")
        backend = await manager.get_sticky_backend_async(ip, "cloud")
        
        assert backend == "anthropic"
    
    @pytest.mark.asyncio
    async def test_concurrent_session_creation(self, manager):
        """Multiple concurrent requests for same IP should get same session."""
        import asyncio
        
        ip = "192.168.1.100"
        
        # Simulate 10 concurrent requests for same IP
        tasks = [
            manager.get_or_create_session_async(ip)
            for _ in range(10)
        ]
        
        sessions = await asyncio.gather(*tasks)
        
        # All should have same session ID
        session_ids = [s.session_id for s in sessions]
        assert len(set(session_ids)) == 1
        
        # Request count should reflect all touches
        final_session = await manager.get_session_async(ip)
        # At least some concurrent touches should have been counted
        assert final_session.request_count >= 1
    
    @pytest.mark.asyncio
    async def test_concurrent_different_ips(self, manager):
        """Concurrent requests from different IPs should get different sessions."""
        import asyncio
        
        ips = [f"192.168.1.{i}" for i in range(20)]
        
        # Simulate concurrent requests from 20 different IPs
        tasks = [
            manager.get_or_create_session_async(ip)
            for ip in ips
        ]
        
        sessions = await asyncio.gather(*tasks)
        
        # All should have different session IDs
        session_ids = [s.session_id for s in sessions]
        assert len(set(session_ids)) == 20
    
    @pytest.mark.asyncio
    async def test_concurrent_lock_race(self, manager):
        """Concurrent tier updates should not corrupt session state."""
        import asyncio
        
        ip = "192.168.1.200"
        
        # First, create session with tier 1 (no lock)
        await manager.update_session_state_async(ip, tier=1, entities=["internal"])
        
        # Now simulate race: multiple tier 3 updates at once
        tasks = [
            manager.update_session_state_async(ip, tier=3, entities=[f"SSN_{i}"])
            for i in range(5)
        ]
        
        sessions = await asyncio.gather(*tasks)
        
        # All should report locked
        for s in sessions:
            assert s.is_local_locked is True
        
        # Original lock trigger should be preserved (idempotent)
        final = await manager.get_session_async(ip)
        assert final.lock_trigger_tier == 3
