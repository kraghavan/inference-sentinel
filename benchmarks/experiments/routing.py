"""Experiment 2: Routing Performance.

Measures end-to-end latency and throughput through the Sentinel API.
"""

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from benchmarks.datasets.generator import LabeledPrompt


@dataclass
class RoutingResult:
    """Result from a single routed request."""
    
    prompt_id: str
    expected_tier: int
    actual_tier: int
    route: str  # "local" or "cloud"
    backend: str
    model: str
    
    # Latencies (ms)
    total_latency_ms: float
    classification_latency_ms: float
    routing_latency_ms: float
    inference_latency_ms: float
    
    # Cost
    cost_usd: float
    cost_savings_usd: float
    
    # Tokens
    prompt_tokens: int
    completion_tokens: int
    
    # Status
    success: bool
    error: str | None = None


@dataclass
class RoutingExperimentResults:
    """Aggregated results from routing experiment."""
    
    experiment: str = "routing_performance"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_path: str = ""
    endpoint: str = ""
    
    # Summary
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Routing breakdown
    routed_local: int = 0
    routed_cloud: int = 0
    routing_by_tier: dict = field(default_factory=dict)
    
    # Latency stats (ms)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_mean_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    
    # Component latencies
    classification_latency_mean_ms: float = 0.0
    routing_latency_mean_ms: float = 0.0
    inference_latency_mean_ms: float = 0.0
    
    # Throughput
    total_duration_seconds: float = 0.0
    requests_per_second: float = 0.0
    
    # Cost
    total_cost_usd: float = 0.0
    total_savings_usd: float = 0.0
    
    # Tokens
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    
    # Per-tier breakdown
    tier_stats: dict = field(default_factory=dict)
    
    # Individual results
    results: list = field(default_factory=list)
    errors: list = field(default_factory=list)


class RoutingExperiment:
    """Experiment 2: Routing Performance."""
    
    def __init__(
        self,
        endpoint: str = "http://localhost:8000",
        timeout: float = 120.0,
        concurrency: int = 1,  # Sequential by default for accurate latency
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.concurrency = concurrency
    
    async def send_request(
        self,
        client: httpx.AsyncClient,
        prompt: LabeledPrompt,
    ) -> RoutingResult:
        """Send a single inference request."""
        start = time.perf_counter()
        
        try:
            response = await client.post(
                f"{self.endpoint}/v1/inference",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": prompt.text}],
                    "max_tokens": 50,  # Short response for benchmarking
                    "temperature": 0.7,
                },
                timeout=self.timeout,
            )
            
            total_latency = (time.perf_counter() - start) * 1000
            
            if response.status_code != 200:
                return RoutingResult(
                    prompt_id=prompt.id,
                    expected_tier=prompt.expected_tier,
                    actual_tier=-1,
                    route="error",
                    backend="",
                    model="",
                    total_latency_ms=total_latency,
                    classification_latency_ms=0,
                    routing_latency_ms=0,
                    inference_latency_ms=0,
                    cost_usd=0,
                    cost_savings_usd=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                )
            
            data = response.json()
            sentinel = data.get("sentinel", {})
            usage = data.get("usage", {})
            
            return RoutingResult(
                prompt_id=prompt.id,
                expected_tier=prompt.expected_tier,
                actual_tier=sentinel.get("privacy_tier", 0),
                route=sentinel.get("route", "unknown"),
                backend=sentinel.get("backend", ""),
                model=sentinel.get("model", ""),
                total_latency_ms=total_latency,
                classification_latency_ms=sentinel.get("classification_latency_ms", 0),
                routing_latency_ms=sentinel.get("routing_latency_ms", 0),
                inference_latency_ms=sentinel.get("inference_latency_ms", 0),
                cost_usd=sentinel.get("cost_usd", 0),
                cost_savings_usd=sentinel.get("cost_savings_usd", 0),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                success=True,
            )
            
        except Exception as e:
            total_latency = (time.perf_counter() - start) * 1000
            return RoutingResult(
                prompt_id=prompt.id,
                expected_tier=prompt.expected_tier,
                actual_tier=-1,
                route="error",
                backend="",
                model="",
                total_latency_ms=total_latency,
                classification_latency_ms=0,
                routing_latency_ms=0,
                inference_latency_ms=0,
                cost_usd=0,
                cost_savings_usd=0,
                prompt_tokens=0,
                completion_tokens=0,
                success=False,
                error=str(e),
            )
    
    async def run_async(self, dataset: list[LabeledPrompt]) -> RoutingExperimentResults:
        """Run the experiment asynchronously."""
        results = RoutingExperimentResults(endpoint=self.endpoint)
        individual_results: list[RoutingResult] = []
        
        print(f"Running {len(dataset)} requests against {self.endpoint}...")
        
        start_time = time.perf_counter()
        
        async with httpx.AsyncClient() as client:
            # Run sequentially for accurate latency measurement
            for i, prompt in enumerate(dataset):
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{len(dataset)}")
                
                result = await self.send_request(client, prompt)
                individual_results.append(result)
        
        total_duration = time.perf_counter() - start_time
        
        # Aggregate results
        results.total_requests = len(individual_results)
        results.total_duration_seconds = total_duration
        
        successful = [r for r in individual_results if r.success]
        failed = [r for r in individual_results if not r.success]
        
        results.successful_requests = len(successful)
        results.failed_requests = len(failed)
        results.errors = [{"prompt_id": r.prompt_id, "error": r.error} for r in failed]
        
        if successful:
            latencies = [r.total_latency_ms for r in successful]
            latencies.sort()
            
            results.latency_mean_ms = statistics.mean(latencies)
            results.latency_min_ms = min(latencies)
            results.latency_max_ms = max(latencies)
            results.latency_p50_ms = latencies[int(len(latencies) * 0.50)]
            results.latency_p95_ms = latencies[int(len(latencies) * 0.95)]
            results.latency_p99_ms = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
            
            # Component latencies
            results.classification_latency_mean_ms = statistics.mean(
                [r.classification_latency_ms for r in successful]
            )
            results.routing_latency_mean_ms = statistics.mean(
                [r.routing_latency_ms for r in successful]
            )
            results.inference_latency_mean_ms = statistics.mean(
                [r.inference_latency_ms for r in successful]
            )
            
            # Routing breakdown
            results.routed_local = sum(1 for r in successful if r.route == "local")
            results.routed_cloud = sum(1 for r in successful if r.route == "cloud")
            
            # By tier
            for tier in range(4):
                tier_results = [r for r in successful if r.actual_tier == tier]
                if tier_results:
                    results.routing_by_tier[tier] = {
                        "count": len(tier_results),
                        "local": sum(1 for r in tier_results if r.route == "local"),
                        "cloud": sum(1 for r in tier_results if r.route == "cloud"),
                        "latency_mean_ms": statistics.mean([r.total_latency_ms for r in tier_results]),
                    }
            
            # Per-tier detailed stats
            for tier in range(4):
                tier_results = [r for r in successful if r.expected_tier == tier]
                if tier_results:
                    tier_latencies = [r.total_latency_ms for r in tier_results]
                    tier_latencies.sort()
                    results.tier_stats[tier] = {
                        "count": len(tier_results),
                        "latency_p50_ms": tier_latencies[int(len(tier_latencies) * 0.50)],
                        "latency_p95_ms": tier_latencies[min(int(len(tier_latencies) * 0.95), len(tier_latencies) - 1)],
                        "latency_mean_ms": statistics.mean(tier_latencies),
                        "routed_local": sum(1 for r in tier_results if r.route == "local"),
                        "routed_cloud": sum(1 for r in tier_results if r.route == "cloud"),
                    }
            
            # Throughput
            results.requests_per_second = len(successful) / total_duration
            
            # Cost
            results.total_cost_usd = sum(r.cost_usd for r in successful)
            results.total_savings_usd = sum(r.cost_savings_usd for r in successful)
            
            # Tokens
            results.total_prompt_tokens = sum(r.prompt_tokens for r in successful)
            results.total_completion_tokens = sum(r.completion_tokens for r in successful)
        
        # Store individual results (without full objects for JSON serialization)
        results.results = [
            {
                "prompt_id": r.prompt_id,
                "expected_tier": r.expected_tier,
                "actual_tier": r.actual_tier,
                "route": r.route,
                "backend": r.backend,
                "total_latency_ms": r.total_latency_ms,
                "success": r.success,
            }
            for r in individual_results
        ]
        
        return results
    
    def run(self, dataset: list[LabeledPrompt]) -> RoutingExperimentResults:
        """Run the experiment (sync wrapper)."""
        return asyncio.run(self.run_async(dataset))
    
    def print_summary(self, results: RoutingExperimentResults) -> None:
        """Print a summary of results."""
        print(f"\n{'─'*50}")
        print("ROUTING PERFORMANCE SUMMARY")
        print(f"{'─'*50}")
        
        print(f"\n📊 Requests:")
        print(f"   Total: {results.total_requests}")
        print(f"   Successful: {results.successful_requests}")
        print(f"   Failed: {results.failed_requests}")
        
        print(f"\n⏱️  Latency (end-to-end):")
        print(f"   p50: {results.latency_p50_ms:.1f} ms")
        print(f"   p95: {results.latency_p95_ms:.1f} ms")
        print(f"   p99: {results.latency_p99_ms:.1f} ms")
        print(f"   Mean: {results.latency_mean_ms:.1f} ms")
        
        print(f"\n🔧 Component Latencies (mean):")
        print(f"   Classification: {results.classification_latency_mean_ms:.2f} ms")
        print(f"   Routing: {results.routing_latency_mean_ms:.2f} ms")
        print(f"   Inference: {results.inference_latency_mean_ms:.1f} ms")
        
        print(f"\n🔀 Routing Decisions:")
        print(f"   Local: {results.routed_local} ({100*results.routed_local/max(1, results.successful_requests):.1f}%)")
        print(f"   Cloud: {results.routed_cloud} ({100*results.routed_cloud/max(1, results.successful_requests):.1f}%)")
        
        print(f"\n📈 Throughput:")
        print(f"   Duration: {results.total_duration_seconds:.1f}s")
        print(f"   Requests/sec: {results.requests_per_second:.2f}")
        
        print(f"\n💰 Cost:")
        print(f"   Total: ${results.total_cost_usd:.4f}")
        print(f"   Savings: ${results.total_savings_usd:.4f}")
        
        if results.tier_stats:
            print(f"\n📊 Per-Tier Latency:")
            for tier, stats in sorted(results.tier_stats.items()):
                print(f"   Tier {tier}: p50={stats['latency_p50_ms']:.0f}ms, p95={stats['latency_p95_ms']:.0f}ms, n={stats['count']}")
    
    def save_results(self, results: RoutingExperimentResults, output_path: Path) -> None:
        """Save results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "experiment": results.experiment,
            "timestamp": results.timestamp,
            "endpoint": results.endpoint,
            "dataset_path": results.dataset_path,
            "summary": {
                "total_requests": results.total_requests,
                "successful_requests": results.successful_requests,
                "failed_requests": results.failed_requests,
                "total_duration_seconds": results.total_duration_seconds,
                "requests_per_second": results.requests_per_second,
            },
            "latency": {
                "p50_ms": results.latency_p50_ms,
                "p95_ms": results.latency_p95_ms,
                "p99_ms": results.latency_p99_ms,
                "mean_ms": results.latency_mean_ms,
                "min_ms": results.latency_min_ms,
                "max_ms": results.latency_max_ms,
            },
            "component_latency": {
                "classification_mean_ms": results.classification_latency_mean_ms,
                "routing_mean_ms": results.routing_latency_mean_ms,
                "inference_mean_ms": results.inference_latency_mean_ms,
            },
            "routing": {
                "local": results.routed_local,
                "cloud": results.routed_cloud,
                "by_tier": results.routing_by_tier,
            },
            "cost": {
                "total_usd": results.total_cost_usd,
                "savings_usd": results.total_savings_usd,
            },
            "tokens": {
                "prompt": results.total_prompt_tokens,
                "completion": results.total_completion_tokens,
            },
            "tier_stats": results.tier_stats,
            "errors": results.errors,
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
