"""Experiment 5: Session Stickiness & One-Way Trapdoor.

Tests the session management feature:
1. State transition: CLOUD_ELIGIBLE → LOCAL_LOCKED
2. Backend stickiness within sessions
3. Handoff overhead when switching to local
4. Buffer memory behavior under load
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from benchmarks.datasets.generator import LabeledPrompt


@dataclass
class SessionTransitionResult:
    """Result from a session transition test."""
    
    session_ip: str
    requests_sent: int
    
    # State tracking
    initial_state: str  # "cloud_eligible"
    final_state: str    # "cloud_eligible" or "local_locked"
    locked_at_request: int = -1  # Request index that triggered lock
    lock_trigger_tier: int = -1
    
    # Routing after lock
    post_lock_routes: list = field(default_factory=list)
    trapdoor_violations: int = 0  # Routes to cloud after lock
    
    # Backend stickiness
    backends_used: list = field(default_factory=list)
    backend_changes: int = 0  # Should be 0 for proper stickiness
    
    # Latencies
    avg_latency_ms: float = 0.0
    handoff_latency_ms: float = 0.0  # First request after lock
    
    success: bool = True
    errors: list = field(default_factory=list)


@dataclass
class SessionExperimentResults:
    """Aggregated results from session experiment."""
    
    experiment: str = "session_stickiness"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    endpoint: str = ""
    
    # Test configuration
    sessions_tested: int = 0
    requests_per_session: int = 0
    pii_probability: float = 0.0
    
    # State transition metrics
    sessions_locked: int = 0
    avg_requests_before_lock: float = 0.0
    
    # Trapdoor validation
    total_post_lock_requests: int = 0
    trapdoor_violations: int = 0
    trapdoor_success_rate: float = 0.0
    
    # Backend stickiness
    sessions_with_stickiness_violations: int = 0
    stickiness_success_rate: float = 0.0
    
    # Handoff overhead
    avg_handoff_latency_ms: float = 0.0
    avg_normal_latency_ms: float = 0.0
    handoff_overhead_ms: float = 0.0
    handoff_overhead_percent: float = 0.0
    
    # Individual session results
    session_results: list = field(default_factory=list)
    
    # Errors
    total_errors: int = 0


class SessionExperiment:
    """Experiment 5: Session Stickiness & One-Way Trapdoor."""
    
    # PII patterns for triggering locks
    PII_PATTERNS = [
        "My SSN is {ssn}",
        "My email is {email}",
        "Credit card: {cc}",
        "Password: {pwd}",
    ]
    
    def __init__(
        self,
        endpoint: str = "http://localhost:8000",
        timeout: float = 120.0,
        sessions: int = 20,
        requests_per_session: int = 10,
        pii_probability: float = 0.3,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.sessions = sessions
        self.requests_per_session = requests_per_session
        self.pii_probability = pii_probability
    
    def _generate_ip(self, session_index: int) -> str:
        """Generate unique IP for each session."""
        return f"10.0.{(session_index >> 8) & 255}.{session_index & 255}"
    
    def _generate_pii_message(self) -> str:
        """Generate a message with PII to trigger lock."""
        pattern = random.choice(self.PII_PATTERNS)
        return pattern.format(
            ssn=f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}",
            email=f"user{random.randint(1,9999)}@secret.com",
            cc=f"4{''.join(str(random.randint(0,9)) for _ in range(15))}",
            pwd=f"secret{random.randint(1000,9999)}",
        )
    
    def _generate_clean_message(self, index: int) -> str:
        """Generate a clean message (no PII)."""
        questions = [
            "What is the capital of France?",
            "How many planets are there?",
            "What is 2 + 2?",
            "Tell me about the weather.",
            "What's a good recipe for pasta?",
        ]
        return f"Question {index}: {random.choice(questions)}"
    
    async def test_session(
        self,
        client: httpx.AsyncClient,
        session_index: int,
    ) -> SessionTransitionResult:
        """Test a single session's behavior."""
        ip = self._generate_ip(session_index)
        headers = {"X-Forwarded-For": ip}
        
        result = SessionTransitionResult(
            session_ip=ip,
            requests_sent=0,
            initial_state="cloud_eligible",
            final_state="cloud_eligible",
        )
        
        is_locked = False
        latencies = []
        normal_latencies = []
        
        for i in range(self.requests_per_session):
            # Decide if this request includes PII
            include_pii = random.random() < self.pii_probability
            
            if include_pii:
                message = self._generate_pii_message()
            else:
                message = self._generate_clean_message(i)
            
            start = time.perf_counter()
            
            try:
                response = await client.post(
                    f"{self.endpoint}/v1/inference",
                    json={
                        "model": "auto",
                        "messages": [{"role": "user", "content": message}],
                        "max_tokens": 30,
                        "temperature": 0.7,
                    },
                    headers=headers,
                    timeout=self.timeout,
                )
                
                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)
                
                if response.status_code != 200:
                    result.errors.append(f"Request {i}: HTTP {response.status_code}")
                    continue
                
                data = response.json()
                sentinel = data.get("sentinel", {})
                
                route = sentinel.get("route", "unknown")
                backend = sentinel.get("backend", "unknown")
                session_state = sentinel.get("session_state", "unknown")
                session_locked_by_pii = sentinel.get("session_locked_by_pii", False)
                
                result.requests_sent += 1
                result.backends_used.append(backend)
                
                # Track lock transition
                if session_locked_by_pii and not is_locked:
                    is_locked = True
                    result.locked_at_request = i
                    result.lock_trigger_tier = sentinel.get("privacy_tier", 0)
                    result.handoff_latency_ms = latency_ms
                
                # Track post-lock routing
                if is_locked:
                    result.post_lock_routes.append(route)
                    if route == "cloud":
                        result.trapdoor_violations += 1
                else:
                    normal_latencies.append(latency_ms)
                
            except Exception as e:
                result.errors.append(f"Request {i}: {str(e)}")
        
        # Finalize results
        result.final_state = "local_locked" if is_locked else "cloud_eligible"
        
        if latencies:
            result.avg_latency_ms = sum(latencies) / len(latencies)
        
        # Check backend stickiness
        if result.backends_used:
            unique_backends = set(result.backends_used)
            result.backend_changes = len(unique_backends) - 1
        
        result.success = len(result.errors) == 0
        
        return result
    
    async def run_async(self, dataset: list[LabeledPrompt] = None) -> SessionExperimentResults:
        """Run the session experiment.
        
        Note: dataset parameter is ignored - this experiment generates its own traffic.
        """
        results = SessionExperimentResults(
            endpoint=self.endpoint,
            sessions_tested=self.sessions,
            requests_per_session=self.requests_per_session,
            pii_probability=self.pii_probability,
        )
        
        print(f"Running session experiment...")
        print(f"  Sessions: {self.sessions}")
        print(f"  Requests per session: {self.requests_per_session}")
        print(f"  PII probability: {self.pii_probability * 100}%")
        print(f"  Endpoint: {self.endpoint}")
        
        session_results: list[SessionTransitionResult] = []
        
        async with httpx.AsyncClient() as client:
            for i in range(self.sessions):
                if (i + 1) % 5 == 0:
                    print(f"  Progress: {i + 1}/{self.sessions} sessions")
                
                session_result = await self.test_session(client, i)
                session_results.append(session_result)
        
        # Aggregate results
        results.session_results = [
            {
                "ip": r.session_ip,
                "final_state": r.final_state,
                "locked_at": r.locked_at_request,
                "trapdoor_violations": r.trapdoor_violations,
                "backend_changes": r.backend_changes,
            }
            for r in session_results
        ]
        
        # Count locked sessions
        locked_sessions = [r for r in session_results if r.final_state == "local_locked"]
        results.sessions_locked = len(locked_sessions)
        
        if locked_sessions:
            results.avg_requests_before_lock = sum(
                r.locked_at_request for r in locked_sessions
            ) / len(locked_sessions)
        
        # Trapdoor validation
        results.total_post_lock_requests = sum(
            len(r.post_lock_routes) for r in session_results
        )
        results.trapdoor_violations = sum(
            r.trapdoor_violations for r in session_results
        )
        
        if results.total_post_lock_requests > 0:
            results.trapdoor_success_rate = 1.0 - (
                results.trapdoor_violations / results.total_post_lock_requests
            )
        else:
            results.trapdoor_success_rate = 1.0
        
        # Backend stickiness
        sessions_with_changes = sum(
            1 for r in session_results if r.backend_changes > 0
        )
        results.sessions_with_stickiness_violations = sessions_with_changes
        results.stickiness_success_rate = 1.0 - (
            sessions_with_changes / self.sessions
        )
        
        # Handoff overhead
        handoff_latencies = [
            r.handoff_latency_ms for r in locked_sessions
            if r.handoff_latency_ms > 0
        ]
        normal_latencies = [
            r.avg_latency_ms for r in session_results
            if r.final_state == "cloud_eligible" and r.avg_latency_ms > 0
        ]
        
        if handoff_latencies:
            results.avg_handoff_latency_ms = sum(handoff_latencies) / len(handoff_latencies)
        if normal_latencies:
            results.avg_normal_latency_ms = sum(normal_latencies) / len(normal_latencies)
        
        if results.avg_normal_latency_ms > 0:
            results.handoff_overhead_ms = (
                results.avg_handoff_latency_ms - results.avg_normal_latency_ms
            )
            results.handoff_overhead_percent = (
                results.handoff_overhead_ms / results.avg_normal_latency_ms * 100
            )
        
        # Errors
        results.total_errors = sum(len(r.errors) for r in session_results)
        
        return results
    
    def run(self, dataset: list[LabeledPrompt] = None) -> SessionExperimentResults:
        """Synchronous wrapper for run_async."""
        return asyncio.run(self.run_async(dataset))
    
    def save_results(
        self,
        results: SessionExperimentResults,
        output_dir: Path,
    ) -> Path:
        """Save results to JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"session_{results.timestamp.replace(':', '-')}.json"
        
        with open(output_file, "w") as f:
            json.dump(results.__dict__, f, indent=2, default=str)
        
        return output_file


def print_session_summary(results: SessionExperimentResults) -> None:
    """Print a summary of session experiment results."""
    print("\n" + "=" * 60)
    print("SESSION STICKINESS EXPERIMENT RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Configuration:")
    print(f"   Sessions tested: {results.sessions_tested}")
    print(f"   Requests per session: {results.requests_per_session}")
    print(f"   PII probability: {results.pii_probability * 100:.0f}%")
    
    print(f"\n🔒 State Transitions:")
    print(f"   Sessions locked: {results.sessions_locked}/{results.sessions_tested}")
    print(f"   Avg requests before lock: {results.avg_requests_before_lock:.1f}")
    
    print(f"\n🚪 One-Way Trapdoor:")
    print(f"   Post-lock requests: {results.total_post_lock_requests}")
    print(f"   Trapdoor violations: {results.trapdoor_violations}")
    print(f"   Success rate: {results.trapdoor_success_rate * 100:.1f}%")
    
    if results.trapdoor_violations > 0:
        print(f"   ⚠️  WARNING: {results.trapdoor_violations} requests routed to cloud after lock!")
    else:
        print(f"   ✓ All post-lock requests correctly routed to local")
    
    print(f"\n🔗 Backend Stickiness:")
    print(f"   Sessions with violations: {results.sessions_with_stickiness_violations}")
    print(f"   Success rate: {results.stickiness_success_rate * 100:.1f}%")
    
    print(f"\n⏱️  Handoff Overhead:")
    print(f"   Normal latency: {results.avg_normal_latency_ms:.1f} ms")
    print(f"   Handoff latency: {results.avg_handoff_latency_ms:.1f} ms")
    print(f"   Overhead: {results.handoff_overhead_ms:.1f} ms ({results.handoff_overhead_percent:.1f}%)")
    
    if results.total_errors > 0:
        print(f"\n❌ Errors: {results.total_errors}")
    
    print("\n" + "=" * 60)
    
    # Overall verdict
    if (results.trapdoor_violations == 0 and 
        results.sessions_with_stickiness_violations == 0 and
        results.total_errors == 0):
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ ISSUES DETECTED")
    
    print("=" * 60)
