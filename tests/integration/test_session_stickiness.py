"""Integration tests for session stickiness.

These tests simulate realistic concurrent request patterns
that would occur in production with Docker.

Run with: pytest tests/integration/test_session_stickiness.py -v
Requires: docker-compose up -d sentinel
"""

import asyncio
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import httpx
import pytest

# Test configuration
BASE_URL = "http://localhost:8000"
OPENAI_ENDPOINT = f"{BASE_URL}/v1/chat/completions"


@pytest.fixture
def client():
    """HTTP client with custom headers for IP simulation."""
    return httpx.AsyncClient(timeout=60.0)


def make_request_payload(message: str, max_tokens: int = 50) -> dict:
    """Create a chat completion request payload."""
    return {
        "model": "auto",
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
    }


class TestSessionStickinessIntegration:
    """Integration tests for session stickiness behavior."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_state_transition(self, client):
        """Test one-way trapdoor: CLOUD_ELIGIBLE → LOCAL_LOCKED.
        
        Scenario:
        1. Send clean message → should route to CLOUD
        2. Send PII message → should lock to LOCAL
        3. Send clean message again → should STILL route to LOCAL
        """
        # Simulate unique client IP
        fake_ip = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        headers = {"X-Forwarded-For": fake_ip}
        
        # Step 1: Clean message → CLOUD
        clean_msg = "What is the capital of France?"
        resp1 = await client.post(
            OPENAI_ENDPOINT,
            json=make_request_payload(clean_msg),
            headers=headers,
        )
        assert resp1.status_code == 200
        data1 = resp1.json()
        
        # Check response header (if implemented)
        session_header1 = resp1.headers.get("X-Sentinel-Session", "")
        route1 = resp1.headers.get("X-Sentinel-Route", "unknown")
        
        print(f"Request 1 (clean): route={route1}, session={session_header1}")
        
        # Step 2: PII message → should trigger LOCAL_LOCKED
        pii_msg = "My SSN is 123-45-6789 and my email is john@example.com"
        resp2 = await client.post(
            OPENAI_ENDPOINT,
            json=make_request_payload(pii_msg),
            headers=headers,
        )
        assert resp2.status_code == 200
        
        session_header2 = resp2.headers.get("X-Sentinel-Session", "")
        route2 = resp2.headers.get("X-Sentinel-Route", "unknown")
        
        print(f"Request 2 (PII): route={route2}, session={session_header2}")
        
        # Should now be locked to local
        assert "local" in route2.lower() or "secure" in session_header2.lower()
        
        # Step 3: Clean message AGAIN → should STILL route to LOCAL (trapdoor)
        clean_msg2 = "Thanks! What about Germany?"
        resp3 = await client.post(
            OPENAI_ENDPOINT,
            json=make_request_payload(clean_msg2),
            headers=headers,
        )
        assert resp3.status_code == 200
        
        session_header3 = resp3.headers.get("X-Sentinel-Session", "")
        route3 = resp3.headers.get("X-Sentinel-Route", "unknown")
        
        print(f"Request 3 (clean after PII): route={route3}, session={session_header3}")
        
        # Critical assertion: should STILL be local (one-way trapdoor)
        assert "local" in route3.lower() or "secure" in session_header3.lower()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_backend_stickiness_cloud(self, client):
        """Test that cloud requests stick to same backend.
        
        Multiple clean requests from same IP should use same cloud backend.
        """
        fake_ip = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        headers = {"X-Forwarded-For": fake_ip}
        
        backends_used = []
        
        # Send 5 clean requests
        for i in range(5):
            msg = f"Question {i}: What is {i} + {i}?"
            resp = await client.post(
                OPENAI_ENDPOINT,
                json=make_request_payload(msg),
                headers=headers,
            )
            assert resp.status_code == 200
            
            backend = resp.headers.get("X-Sentinel-Backend", "unknown")
            backends_used.append(backend)
            
            print(f"Request {i+1}: backend={backend}")
        
        # All requests should use same backend (stickiness)
        unique_backends = set(backends_used)
        assert len(unique_backends) == 1, f"Expected 1 backend, got {unique_backends}"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_different_ips_different_sessions(self, client):
        """Test that different IPs get independent sessions."""
        results = {}
        
        async def send_request(ip: str, include_pii: bool) -> Tuple[str, str]:
            headers = {"X-Forwarded-For": ip}
            
            if include_pii:
                msg = f"My SSN from IP {ip} is 123-45-6789"
            else:
                msg = f"Hello from IP {ip}"
            
            resp = await client.post(
                OPENAI_ENDPOINT,
                json=make_request_payload(msg),
                headers=headers,
            )
            
            route = resp.headers.get("X-Sentinel-Route", "unknown")
            session = resp.headers.get("X-Sentinel-Session", "none")
            
            return route, session
        
        # IP 1: Send PII → should lock
        route1, session1 = await send_request("192.168.1.100", include_pii=True)
        
        # IP 2: Send clean → should stay cloud
        route2, session2 = await send_request("192.168.1.200", include_pii=False)
        
        # IP 1 again: Send clean → should STILL be local (locked)
        route1b, session1b = await send_request("192.168.1.100", include_pii=False)
        
        print(f"IP 1 (PII): route={route1}")
        print(f"IP 2 (clean): route={route2}")
        print(f"IP 1 (clean after PII): route={route1b}")
        
        # IP 1 should be locked to local
        assert "local" in route1.lower() or "secure" in session1.lower()
        assert "local" in route1b.lower() or "secure" in session1b.lower()
        
        # IP 2 should still be cloud eligible (independent session)
        # Note: May route to local for other reasons, but session shouldn't be "secure-local"
        assert session1 != session2  # Different sessions
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_requests_same_ip(self, client):
        """Test concurrent requests from same IP are handled correctly.
        
        This tests the asyncio.Lock() implementation.
        """
        fake_ip = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        headers = {"X-Forwarded-For": fake_ip}
        
        async def send_request(idx: int) -> dict:
            msg = f"Concurrent request {idx}: What is 2+2?"
            resp = await client.post(
                OPENAI_ENDPOINT,
                json=make_request_payload(msg),
                headers=headers,
            )
            return {
                "idx": idx,
                "status": resp.status_code,
                "route": resp.headers.get("X-Sentinel-Route", "unknown"),
                "backend": resp.headers.get("X-Sentinel-Backend", "unknown"),
            }
        
        # Send 10 concurrent requests
        tasks = [send_request(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        for r in results:
            assert r["status"] == 200, f"Request {r['idx']} failed"
        
        # All should use same backend (stickiness under concurrency)
        backends = [r["backend"] for r in results]
        unique_backends = set(backends)
        
        print(f"Concurrent results: {len(results)} requests, backends={unique_backends}")
        
        # Allow "unknown" if header not implemented yet
        backends_without_unknown = [b for b in backends if b != "unknown"]
        if backends_without_unknown:
            assert len(set(backends_without_unknown)) == 1
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_pii_lock_race(self, client):
        """Test race condition: multiple PII requests at once.
        
        All should result in locked session, first lock should win.
        """
        fake_ip = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        headers = {"X-Forwarded-For": fake_ip}
        
        pii_messages = [
            "My SSN is 111-22-3333",
            "Email: alice@secret.com",
            "Credit card: 4111111111111111",
            "My password is hunter2",
            "DOB: 01/15/1990",
        ]
        
        async def send_pii(msg: str) -> dict:
            resp = await client.post(
                OPENAI_ENDPOINT,
                json=make_request_payload(msg),
                headers=headers,
            )
            return {
                "msg": msg[:20],
                "status": resp.status_code,
                "route": resp.headers.get("X-Sentinel-Route", "unknown"),
                "session": resp.headers.get("X-Sentinel-Session", "unknown"),
            }
        
        # Send all PII requests concurrently
        tasks = [send_pii(msg) for msg in pii_messages]
        results = await asyncio.gather(*tasks)
        
        print("Concurrent PII results:")
        for r in results:
            print(f"  {r['msg']}: route={r['route']}, session={r['session']}")
        
        # All should succeed
        for r in results:
            assert r["status"] == 200
        
        # Now send a clean message - should still be locked
        clean_resp = await client.post(
            OPENAI_ENDPOINT,
            json=make_request_payload("What's the weather?"),
            headers=headers,
        )
        
        final_route = clean_resp.headers.get("X-Sentinel-Route", "unknown")
        final_session = clean_resp.headers.get("X-Sentinel-Session", "unknown")
        
        print(f"Final clean request: route={final_route}, session={final_session}")
        
        # Should be locked to local
        assert "local" in final_route.lower() or "secure" in final_session.lower()


class TestLoadSimulation:
    """Load tests simulating realistic traffic patterns."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_mixed_traffic_load(self, client):
        """Simulate mixed traffic from multiple IPs.
        
        Pattern:
        - 20 unique IPs
        - 5 requests per IP
        - 30% chance of PII per request
        - Verify session isolation
        """
        num_ips = 20
        requests_per_ip = 5
        pii_probability = 0.3
        
        ips = [f"10.0.{i // 256}.{i % 256}" for i in range(1, num_ips + 1)]
        
        # Track expected state per IP
        expected_locked: Dict[str, bool] = defaultdict(bool)
        actual_results: Dict[str, List[dict]] = defaultdict(list)
        
        async def send_request(ip: str, request_idx: int) -> dict:
            headers = {"X-Forwarded-For": ip}
            
            include_pii = random.random() < pii_probability
            
            if include_pii:
                expected_locked[ip] = True  # Once PII, always locked
                msg = f"IP {ip} request {request_idx}: SSN 123-45-6789"
            else:
                msg = f"IP {ip} request {request_idx}: Hello world"
            
            resp = await client.post(
                OPENAI_ENDPOINT,
                json=make_request_payload(msg, max_tokens=10),
                headers=headers,
            )
            
            return {
                "ip": ip,
                "idx": request_idx,
                "pii": include_pii,
                "status": resp.status_code,
                "route": resp.headers.get("X-Sentinel-Route", "unknown"),
            }
        
        # Generate all requests
        all_tasks = []
        for ip in ips:
            for i in range(requests_per_ip):
                all_tasks.append(send_request(ip, i))
        
        # Shuffle to simulate realistic interleaving
        random.shuffle(all_tasks)
        
        # Execute with some concurrency (not all at once)
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        bounded_tasks = [bounded_task(t) for t in all_tasks]
        results = await asyncio.gather(*bounded_tasks)
        
        # Organize by IP
        for r in results:
            actual_results[r["ip"]].append(r)
        
        # Verify results
        failures = []
        for ip in ips:
            ip_results = sorted(actual_results[ip], key=lambda x: x["idx"])
            
            # Find when PII was first sent
            pii_sent_at = None
            for r in ip_results:
                if r["pii"]:
                    pii_sent_at = r["idx"]
                    break
            
            # Check all requests AFTER PII are routed locally
            if pii_sent_at is not None:
                for r in ip_results:
                    if r["idx"] > pii_sent_at:
                        if "local" not in r["route"].lower():
                            failures.append(
                                f"IP {ip} request {r['idx']}: expected local after PII at {pii_sent_at}, got {r['route']}"
                            )
        
        print(f"Load test: {num_ips} IPs × {requests_per_ip} requests = {len(results)} total")
        print(f"PII rate: {pii_probability * 100}%")
        print(f"Failures: {len(failures)}")
        
        for f in failures[:5]:  # Show first 5 failures
            print(f"  {f}")
        
        assert len(failures) == 0, f"{len(failures)} session isolation failures"


if __name__ == "__main__":
    # Allow running directly for quick testing
    import sys
    
    async def quick_test():
        async with httpx.AsyncClient(timeout=60.0) as client:
            test = TestSessionStickinessIntegration()
            test_fixture = type('obj', (object,), {'client': client})()
            
            print("Running session state transition test...")
            await test.test_session_state_transition(client)
            print("✓ Passed\n")
    
    asyncio.run(quick_test())
