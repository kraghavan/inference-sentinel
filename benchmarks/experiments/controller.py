"""Experiment 4: Closed-Loop Controller Effectiveness.

Measures the controller's ability to:
1. Detect routing patterns and drift
2. Generate actionable recommendations
3. Track cost savings from routing decisions
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from benchmarks.datasets.generator import LabeledPrompt


@dataclass 
class ControllerMetrics:
    """Metrics from controller status."""
    
    enabled: bool = False
    mode: str = "observe"
    running: bool = False
    total_evaluations: int = 0
    recommendations: list = field(default_factory=list)
    tier_metrics: dict = field(default_factory=dict)


@dataclass
class ControllerExperimentResults:
    """Results from controller effectiveness experiment."""
    
    experiment: str = "controller_effectiveness"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_path: str = ""
    endpoint: str = ""
    
    # Traffic generated
    total_requests_sent: int = 0
    successful_requests: int = 0
    
    # Controller state before/after
    controller_before: dict = field(default_factory=dict)
    controller_after: dict = field(default_factory=dict)
    
    # Recommendations generated
    recommendations_count: int = 0
    recommendations: list = field(default_factory=list)
    
    # Routing distribution
    routing_distribution: dict = field(default_factory=dict)
    tier_distribution: dict = field(default_factory=dict)
    
    # Cost analysis
    total_cost_usd: float = 0.0
    total_savings_usd: float = 0.0
    potential_additional_savings_usd: float = 0.0
    
    # Controller metrics per tier
    tier_metrics: dict = field(default_factory=dict)
    
    # Drift detection
    drift_detected: bool = False
    drift_details: list = field(default_factory=list)
    
    # Shadow mode analysis (if enabled)
    shadow_metrics: dict = field(default_factory=dict)


class ControllerExperiment:
    """Experiment 4: Closed-Loop Controller Effectiveness."""
    
    def __init__(
        self,
        endpoint: str = "http://localhost:8000",
        timeout: float = 120.0,
        warmup_requests: int = 20,  # Requests before checking controller
        isolate_sessions: bool = True,  # Use unique IP per request
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.warmup_requests = warmup_requests
        self.isolate_sessions = isolate_sessions
    
    def _generate_unique_ip(self, index: int) -> str:
        """Generate unique IP for session isolation."""
        return f"10.{(index >> 16) & 255}.{(index >> 8) & 255}.{index & 255}"
    
    async def get_controller_status(self, client: httpx.AsyncClient) -> dict:
        """Get current controller status."""
        try:
            response = await client.get(
                f"{self.endpoint}/admin/controller/status",
                timeout=10.0,
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"  ⚠️  Failed to get controller status: {e}")
        return {}
    
    async def get_shadow_metrics(self, client: httpx.AsyncClient) -> dict:
        """Get shadow mode metrics."""
        try:
            response = await client.get(
                f"{self.endpoint}/admin/shadow/metrics",
                timeout=10.0,
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {}
    
    async def send_inference_request(
        self,
        client: httpx.AsyncClient,
        prompt: LabeledPrompt,
        request_index: int = 0,
    ) -> dict:
        """Send a single inference request."""
        # Build headers with unique IP for session isolation
        headers = {}
        if self.isolate_sessions:
            headers["X-Forwarded-For"] = self._generate_unique_ip(request_index)
        
        try:
            response = await client.post(
                f"{self.endpoint}/v1/inference",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": prompt.text}],
                    "max_tokens": 50,
                    "temperature": 0.7,
                },
                headers=headers,
                timeout=self.timeout,
            )
            
            if response.status_code == 200:
                data = response.json()
                sentinel = data.get("sentinel", {})
                return {
                    "success": True,
                    "tier": sentinel.get("privacy_tier", 0),
                    "route": sentinel.get("route", "unknown"),
                    "cost_usd": sentinel.get("cost_usd", 0),
                    "cost_savings_usd": sentinel.get("cost_savings_usd", 0),
                }
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def wait_for_controller_evaluation(
        self,
        client: httpx.AsyncClient,
        initial_evaluations: int,
        max_wait_seconds: int = 120,
    ) -> bool:
        """Wait for controller to run at least one evaluation."""
        print(f"  Waiting for controller evaluation (max {max_wait_seconds}s)...")
        
        start = time.time()
        while time.time() - start < max_wait_seconds:
            status = await self.get_controller_status(client)
            current_evals = status.get("total_evaluations", 0)
            
            if current_evals > initial_evaluations:
                print(f"  ✓ Controller evaluated ({current_evals} total evaluations)")
                return True
            
            await asyncio.sleep(5)
        
        print(f"  ⚠️  Timeout waiting for controller evaluation")
        return False
    
    async def run_async(self, dataset: list[LabeledPrompt]) -> ControllerExperimentResults:
        """Run the experiment."""
        results = ControllerExperimentResults(endpoint=self.endpoint)
        
        print(f"Running controller experiment with {len(dataset)} prompts...")
        
        async with httpx.AsyncClient() as client:
            # 1. Get initial controller state
            print("\n📊 Getting initial controller state...")
            results.controller_before = await self.get_controller_status(client)
            initial_evaluations = results.controller_before.get("total_evaluations", 0)
            
            if not results.controller_before.get("enabled", False):
                print("  ⚠️  Controller is not enabled!")
            else:
                print(f"  Mode: {results.controller_before.get('mode', 'unknown')}")
                print(f"  Running: {results.controller_before.get('running', False)}")
                print(f"  Evaluations so far: {initial_evaluations}")
            
            # 2. Send traffic
            print(f"\n🚀 Sending {len(dataset)} inference requests...")
            
            routing_counts = {"local": 0, "cloud": 0}
            tier_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            total_cost = 0.0
            total_savings = 0.0
            
            for i, prompt in enumerate(dataset):
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i + 1}/{len(dataset)}")
                
                result = await self.send_inference_request(client, prompt, request_index=i)
                
                if result.get("success"):
                    results.successful_requests += 1
                    route = result.get("route", "unknown")
                    tier = result.get("tier", 0)
                    
                    if route in routing_counts:
                        routing_counts[route] += 1
                    if tier in tier_counts:
                        tier_counts[tier] += 1
                    
                    total_cost += result.get("cost_usd", 0)
                    total_savings += result.get("cost_savings_usd", 0)
                
                results.total_requests_sent += 1
            
            results.routing_distribution = routing_counts
            results.tier_distribution = tier_counts
            results.total_cost_usd = total_cost
            results.total_savings_usd = total_savings
            
            # 3. Wait for controller to evaluate
            print("\n⏳ Waiting for controller to process metrics...")
            await self.wait_for_controller_evaluation(client, initial_evaluations)
            
            # 4. Get final controller state
            print("\n📊 Getting final controller state...")
            results.controller_after = await self.get_controller_status(client)
            
            # Extract recommendations
            recommendations = results.controller_after.get("recommendations", [])
            results.recommendations_count = len(recommendations)
            results.recommendations = recommendations
            
            # Extract tier metrics
            results.tier_metrics = results.controller_after.get("tier_metrics", {})
            
            # Check for drift
            for rec in recommendations:
                if rec.get("type") == "drift_detected":
                    results.drift_detected = True
                    results.drift_details.append(rec)
            
            # 5. Get shadow metrics if available
            print("\n🔍 Getting shadow metrics...")
            results.shadow_metrics = await self.get_shadow_metrics(client)
            
            # 6. Calculate potential savings
            # If any tier 0/1 went to cloud that could go local
            cloud_tier_0_1 = sum(
                1 for r in [routing_counts]  # Simplified
                if r.get("cloud", 0) > 0
            )
            # Estimate: local inference is ~$0.0001 per request vs cloud ~$0.001
            # This is a rough estimate
            results.potential_additional_savings_usd = (
                routing_counts.get("cloud", 0) * 0.0009  # Potential savings per cloud request
            )
        
        return results
    
    def run(self, dataset: list[LabeledPrompt]) -> ControllerExperimentResults:
        """Run the experiment (sync wrapper)."""
        return asyncio.run(self.run_async(dataset))
    
    def print_summary(self, results: ControllerExperimentResults) -> None:
        """Print experiment summary."""
        print(f"\n{'─'*50}")
        print("CONTROLLER EFFECTIVENESS SUMMARY")
        print(f"{'─'*50}")
        
        print(f"\n📊 Traffic Generated:")
        print(f"   Total requests: {results.total_requests_sent}")
        print(f"   Successful: {results.successful_requests}")
        
        print(f"\n🔀 Routing Distribution:")
        total = sum(results.routing_distribution.values())
        for route, count in results.routing_distribution.items():
            pct = 100 * count / max(1, total)
            print(f"   {route.capitalize()}: {count} ({pct:.1f}%)")
        
        print(f"\n📊 Tier Distribution:")
        tier_names = {0: "PUBLIC", 1: "INTERNAL", 2: "CONFIDENTIAL", 3: "RESTRICTED"}
        for tier, count in sorted(results.tier_distribution.items()):
            pct = 100 * count / max(1, total)
            print(f"   Tier {tier} ({tier_names.get(tier, '?')}): {count} ({pct:.1f}%)")
        
        print(f"\n🎛️  Controller State:")
        before = results.controller_before
        after = results.controller_after
        print(f"   Enabled: {after.get('enabled', False)}")
        print(f"   Mode: {after.get('mode', 'unknown')}")
        print(f"   Evaluations: {before.get('total_evaluations', 0)} → {after.get('total_evaluations', 0)}")
        
        print(f"\n💡 Recommendations ({results.recommendations_count}):")
        if results.recommendations:
            for i, rec in enumerate(results.recommendations[:5]):  # Show top 5
                rec_type = rec.get("type", "unknown")
                tier = rec.get("tier", "?")
                reason = rec.get("reason", "")
                print(f"   {i+1}. [{rec_type}] Tier {tier}: {reason[:60]}...")
        else:
            print("   No recommendations generated")
        
        print(f"\n⚠️  Drift Detection:")
        if results.drift_detected:
            print(f"   DRIFT DETECTED!")
            for detail in results.drift_details:
                print(f"   - {detail.get('reason', 'Unknown')}")
        else:
            print("   No drift detected")
        
        print(f"\n💰 Cost Analysis:")
        print(f"   Total cost: ${results.total_cost_usd:.4f}")
        print(f"   Savings (local routing): ${results.total_savings_usd:.4f}")
        print(f"   Potential additional savings: ${results.potential_additional_savings_usd:.4f}")
        
        if results.shadow_metrics:
            print(f"\n🔍 Shadow Mode Metrics:")
            sm = results.shadow_metrics
            print(f"   Samples: {sm.get('total_samples', 0)}")
            print(f"   Agreement rate: {sm.get('agreement_rate', 0)*100:.1f}%")
    
    def save_results(self, results: ControllerExperimentResults, output_path: Path) -> None:
        """Save results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "experiment": results.experiment,
            "timestamp": results.timestamp,
            "endpoint": results.endpoint,
            "dataset_path": results.dataset_path,
            "traffic": {
                "total_requests": results.total_requests_sent,
                "successful": results.successful_requests,
            },
            "routing_distribution": results.routing_distribution,
            "tier_distribution": {str(k): v for k, v in results.tier_distribution.items()},
            "controller_before": results.controller_before,
            "controller_after": results.controller_after,
            "recommendations": {
                "count": results.recommendations_count,
                "items": results.recommendations,
            },
            "drift": {
                "detected": results.drift_detected,
                "details": results.drift_details,
            },
            "cost": {
                "total_usd": results.total_cost_usd,
                "savings_usd": results.total_savings_usd,
                "potential_additional_savings_usd": results.potential_additional_savings_usd,
            },
            "tier_metrics": {str(k): v for k, v in results.tier_metrics.items()},
            "shadow_metrics": results.shadow_metrics,
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
