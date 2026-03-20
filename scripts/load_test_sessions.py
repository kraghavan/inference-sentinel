#!/usr/bin/env python3
"""Load test script for session stickiness.

Simulates realistic traffic patterns to stress test:
1. Concurrent requests from same IP (race conditions)
2. Session isolation between different IPs
3. One-way trapdoor behavior under load

Usage:
    python scripts/load_test_sessions.py --ips 50 --requests 10 --concurrency 20

Requires: httpx, docker-compose up -d sentinel
"""

import argparse
import asyncio
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)


@dataclass
class TestResult:
    """Results from a single request."""
    ip: str
    request_idx: int
    included_pii: bool
    status_code: int
    route: str
    backend: str
    session_header: str
    latency_ms: float
    error: str = ""


@dataclass
class IPStats:
    """Statistics for a single IP."""
    total_requests: int = 0
    pii_requests: int = 0
    first_pii_idx: int = -1
    routes_after_pii: List[str] = field(default_factory=list)
    backends: List[str] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    errors: int = 0


class SessionLoadTester:
    """Load tester for session stickiness."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        num_ips: int = 50,
        requests_per_ip: int = 10,
        concurrency: int = 20,
        pii_probability: float = 0.3,
    ):
        self.base_url = base_url
        self.endpoint = f"{base_url}/v1/chat/completions"
        self.num_ips = num_ips
        self.requests_per_ip = requests_per_ip
        self.concurrency = concurrency
        self.pii_probability = pii_probability
        
        self.results: List[TestResult] = []
        self.ip_stats: Dict[str, IPStats] = defaultdict(IPStats)
        
        # PII patterns for variety
        self.pii_patterns = [
            "My SSN is {ssn}",
            "Email me at {email}",
            "Credit card: {cc}",
            "My password is {pwd}",
            "DOB: {dob}",
            "Phone: {phone}",
        ]
    
    def _generate_pii_message(self) -> str:
        """Generate a random PII message."""
        pattern = random.choice(self.pii_patterns)
        return pattern.format(
            ssn=f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}",
            email=f"user{random.randint(1,9999)}@secret.com",
            cc=f"4{''.join(str(random.randint(0,9)) for _ in range(15))}",
            pwd=f"secret{random.randint(1000,9999)}",
            dob=f"{random.randint(1,12):02d}/{random.randint(1,28):02d}/{random.randint(1950,2000)}",
            phone=f"{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
        )
    
    def _generate_clean_message(self, ip: str, idx: int) -> str:
        """Generate a clean message."""
        questions = [
            "What is the capital of France?",
            "How many planets are in the solar system?",
            "What is 2 + 2?",
            "Tell me a joke",
            "What's the weather like?",
        ]
        return f"Request {idx}: {random.choice(questions)}"
    
    async def _send_request(
        self,
        client: httpx.AsyncClient,
        ip: str,
        request_idx: int,
        include_pii: bool,
    ) -> TestResult:
        """Send a single request."""
        headers = {"X-Forwarded-For": ip}
        
        if include_pii:
            message = self._generate_pii_message()
        else:
            message = self._generate_clean_message(ip, request_idx)
        
        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 10,  # Keep responses short for load test
        }
        
        start = time.perf_counter()
        
        try:
            resp = await client.post(self.endpoint, json=payload, headers=headers)
            latency_ms = (time.perf_counter() - start) * 1000
            
            return TestResult(
                ip=ip,
                request_idx=request_idx,
                included_pii=include_pii,
                status_code=resp.status_code,
                route=resp.headers.get("X-Sentinel-Route", "unknown"),
                backend=resp.headers.get("X-Sentinel-Backend", "unknown"),
                session_header=resp.headers.get("X-Sentinel-Session", "unknown"),
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return TestResult(
                ip=ip,
                request_idx=request_idx,
                included_pii=include_pii,
                status_code=0,
                route="error",
                backend="error",
                session_header="error",
                latency_ms=latency_ms,
                error=str(e),
            )
    
    async def run(self) -> None:
        """Run the load test."""
        print(f"\n{'='*60}")
        print(f"Session Stickiness Load Test")
        print(f"{'='*60}")
        print(f"IPs: {self.num_ips}")
        print(f"Requests per IP: {self.requests_per_ip}")
        print(f"Total requests: {self.num_ips * self.requests_per_ip}")
        print(f"Concurrency: {self.concurrency}")
        print(f"PII probability: {self.pii_probability * 100}%")
        print(f"Endpoint: {self.endpoint}")
        print(f"{'='*60}\n")
        
        # Generate IPs
        ips = [f"10.0.{i // 256}.{i % 256 + 1}" for i in range(self.num_ips)]
        
        # Track which IPs will get PII and when
        ip_pii_schedule: Dict[str, List[bool]] = {}
        for ip in ips:
            schedule = [random.random() < self.pii_probability for _ in range(self.requests_per_ip)]
            ip_pii_schedule[ip] = schedule
        
        # Build task list
        tasks = []
        for ip in ips:
            for idx in range(self.requests_per_ip):
                include_pii = ip_pii_schedule[ip][idx]
                tasks.append((ip, idx, include_pii))
        
        # Shuffle for realistic interleaving
        random.shuffle(tasks)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def bounded_request(client, ip, idx, pii):
            async with semaphore:
                return await self._send_request(client, ip, idx, pii)
        
        # Run requests
        start_time = time.perf_counter()
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Create all tasks
            coros = [bounded_request(client, ip, idx, pii) for ip, idx, pii in tasks]
            
            # Execute with progress
            completed = 0
            total = len(coros)
            
            for coro in asyncio.as_completed(coros):
                result = await coro
                self.results.append(result)
                completed += 1
                
                if completed % 50 == 0 or completed == total:
                    pct = completed / total * 100
                    print(f"Progress: {completed}/{total} ({pct:.1f}%)")
        
        elapsed = time.perf_counter() - start_time
        
        # Analyze results
        self._analyze_results(ip_pii_schedule)
        self._print_report(elapsed)
    
    def _analyze_results(self, ip_pii_schedule: Dict[str, List[bool]]) -> None:
        """Analyze test results."""
        # Organize by IP
        by_ip: Dict[str, List[TestResult]] = defaultdict(list)
        for r in self.results:
            by_ip[r.ip].append(r)
        
        # Calculate stats per IP
        for ip, results in by_ip.items():
            stats = self.ip_stats[ip]
            results_sorted = sorted(results, key=lambda x: x.request_idx)
            
            stats.total_requests = len(results_sorted)
            
            for r in results_sorted:
                if r.error:
                    stats.errors += 1
                    continue
                
                stats.latencies.append(r.latency_ms)
                stats.backends.append(r.backend)
                
                if r.included_pii:
                    stats.pii_requests += 1
                    if stats.first_pii_idx == -1:
                        stats.first_pii_idx = r.request_idx
                
                # Track routes after PII
                if stats.first_pii_idx != -1 and r.request_idx > stats.first_pii_idx:
                    stats.routes_after_pii.append(r.route)
    
    def _print_report(self, elapsed: float) -> None:
        """Print test report."""
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        
        total_requests = len(self.results)
        errors = sum(1 for r in self.results if r.error)
        success_rate = (total_requests - errors) / total_requests * 100
        
        print(f"\nOverall:")
        print(f"  Total requests: {total_requests}")
        print(f"  Errors: {errors}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {total_requests / elapsed:.1f} req/s")
        
        # Latency stats
        all_latencies = [r.latency_ms for r in self.results if not r.error]
        if all_latencies:
            print(f"\nLatency (ms):")
            print(f"  Min: {min(all_latencies):.1f}")
            print(f"  Max: {max(all_latencies):.1f}")
            print(f"  Avg: {sum(all_latencies) / len(all_latencies):.1f}")
            sorted_lat = sorted(all_latencies)
            p50 = sorted_lat[len(sorted_lat) // 2]
            p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
            p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
            print(f"  P50: {p50:.1f}")
            print(f"  P95: {p95:.1f}")
            print(f"  P99: {p99:.1f}")
        
        # Session isolation check
        print(f"\nSession Isolation (One-Way Trapdoor):")
        
        violations = 0
        ips_with_pii = 0
        ips_checked = 0
        
        for ip, stats in self.ip_stats.items():
            if stats.first_pii_idx != -1:
                ips_with_pii += 1
                
                # Check routes after PII
                for route in stats.routes_after_pii:
                    ips_checked += 1
                    if "local" not in route.lower() and route != "unknown":
                        violations += 1
        
        print(f"  IPs that sent PII: {ips_with_pii}")
        print(f"  Post-PII requests checked: {ips_checked}")
        print(f"  Trapdoor violations: {violations}")
        
        if violations > 0:
            print(f"  ⚠️  WARNING: {violations} requests routed to cloud after PII!")
        else:
            print(f"  ✓ All post-PII requests correctly routed to local")
        
        # Backend stickiness check
        print(f"\nBackend Stickiness:")
        
        sticky_ok = 0
        sticky_fail = 0
        
        for ip, stats in self.ip_stats.items():
            backends = [b for b in stats.backends if b != "unknown"]
            if backends:
                unique = set(backends)
                if len(unique) == 1:
                    sticky_ok += 1
                else:
                    sticky_fail += 1
        
        print(f"  IPs with consistent backend: {sticky_ok}")
        print(f"  IPs with backend changes: {sticky_fail}")
        
        if sticky_fail > 0:
            print(f"  ⚠️  WARNING: {sticky_fail} IPs changed backends mid-session!")
        else:
            print(f"  ✓ All sessions maintained backend stickiness")
        
        print(f"\n{'='*60}")
        
        # Overall verdict
        if violations == 0 and sticky_fail == 0 and errors == 0:
            print("✓ ALL TESTS PASSED")
        else:
            print("✗ ISSUES DETECTED")
        
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Load test session stickiness")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--ips", type=int, default=50, help="Number of unique IPs")
    parser.add_argument("--requests", type=int, default=10, help="Requests per IP")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent requests")
    parser.add_argument("--pii-rate", type=float, default=0.3, help="PII probability (0-1)")
    
    args = parser.parse_args()
    
    tester = SessionLoadTester(
        base_url=args.url,
        num_ips=args.ips,
        requests_per_ip=args.requests,
        concurrency=args.concurrency,
        pii_probability=args.pii_rate,
    )
    
    asyncio.run(tester.run())


if __name__ == "__main__":
    main()
