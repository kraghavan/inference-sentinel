"""Experiment 3: Cost Attribution.

Compares actual costs (with privacy routing) vs hypothetical all-cloud costs.
Tracks cost by backend, tier, and calculates savings from local routing.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from benchmarks.datasets.generator import LabeledPrompt


# Pricing per 1K tokens (as of 2024)
PRICING = {
    "anthropic": {
        "claude-3-5-haiku-20241022": {"input": 0.001, "output": 0.005},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "default": {"input": 0.003, "output": 0.015},
    },
    "google": {
        "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
        "gemini-2.0-flash-lite": {"input": 0.000075, "output": 0.0003},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "default": {"input": 0.0001, "output": 0.0004},
    },
    "local": {
        "default": {"input": 0.0, "output": 0.0},  # Free (compute cost negligible)
    },
}


@dataclass
class RequestCost:
    """Cost data for a single request."""
    
    prompt_id: str
    tier: int
    route: str  # "local" or "cloud"
    backend: str  # "anthropic", "google", "ollama"
    model: str
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    actual_cost_usd: float = 0.0
    hypothetical_cloud_cost_usd: float = 0.0  # If sent to Claude
    savings_usd: float = 0.0
    
    success: bool = True
    error: str | None = None


@dataclass
class BackendCostSummary:
    """Cost summary for a single backend."""
    
    backend: str
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    cost_per_request_usd: float = 0.0
    cost_per_1k_tokens_usd: float = 0.0


@dataclass
class TierCostSummary:
    """Cost summary for a single tier."""
    
    tier: int
    tier_name: str
    request_count: int = 0
    routed_local: int = 0
    routed_cloud: int = 0
    total_cost_usd: float = 0.0
    savings_usd: float = 0.0
    avg_cost_per_request_usd: float = 0.0


@dataclass
class CostExperimentResults:
    """Aggregated results from cost attribution experiment."""
    
    experiment: str = "cost_attribution"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_path: str = ""
    endpoint: str = ""
    
    # Request counts
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Token totals
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    
    # Cost totals
    actual_total_cost_usd: float = 0.0
    hypothetical_cloud_cost_usd: float = 0.0
    total_savings_usd: float = 0.0
    savings_percentage: float = 0.0
    
    # Per-backend breakdown
    backend_costs: dict = field(default_factory=dict)  # backend -> BackendCostSummary
    
    # Per-tier breakdown
    tier_costs: dict = field(default_factory=dict)  # tier -> TierCostSummary
    
    # Routing efficiency
    local_requests: int = 0
    cloud_requests: int = 0
    local_percentage: float = 0.0
    
    # Cost efficiency metrics
    avg_cost_per_request_usd: float = 0.0
    avg_cost_per_1k_tokens_usd: float = 0.0
    
    # Individual results
    request_costs: list = field(default_factory=list)
    errors: list = field(default_factory=list)


class CostExperiment:
    """Experiment 3: Cost Attribution."""
    
    TIER_NAMES = {
        0: "PUBLIC",
        1: "INTERNAL",
        2: "CONFIDENTIAL",
        3: "RESTRICTED",
    }
    
    # Reference model for hypothetical cloud cost
    REFERENCE_CLOUD_MODEL = "claude-3-5-sonnet-20241022"
    REFERENCE_CLOUD_PRICING = PRICING["anthropic"]["default"]
    
    def __init__(
        self,
        endpoint: str = "http://localhost:8000",
        timeout: float = 120.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
    
    def calculate_cost(
        self,
        backend: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate cost for a request."""
        backend_pricing = PRICING.get(backend, PRICING["local"])
        model_pricing = backend_pricing.get(model, backend_pricing["default"])
        
        input_cost = (prompt_tokens / 1000) * model_pricing["input"]
        output_cost = (completion_tokens / 1000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    def calculate_hypothetical_cloud_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate what this request would cost if sent to Claude."""
        input_cost = (prompt_tokens / 1000) * self.REFERENCE_CLOUD_PRICING["input"]
        output_cost = (completion_tokens / 1000) * self.REFERENCE_CLOUD_PRICING["output"]
        return input_cost + output_cost
    
    async def send_request(
        self,
        client: httpx.AsyncClient,
        prompt: LabeledPrompt,
    ) -> RequestCost:
        """Send a single inference request and track costs."""
        try:
            response = await client.post(
                f"{self.endpoint}/v1/inference",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": prompt.text}],
                    "max_tokens": 100,  # Slightly longer for cost accuracy
                    "temperature": 0.7,
                },
                timeout=self.timeout,
            )
            
            if response.status_code != 200:
                return RequestCost(
                    prompt_id=prompt.id,
                    tier=prompt.expected_tier,
                    route="error",
                    backend="",
                    model="",
                    success=False,
                    error=f"HTTP {response.status_code}",
                )
            
            data = response.json()
            sentinel = data.get("sentinel", {})
            usage = data.get("usage", {})
            
            route = sentinel.get("route", "cloud")
            backend = sentinel.get("backend", "anthropic")
            model = sentinel.get("model", "")
            
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            # Normalize backend name
            if "ollama" in backend.lower() or route == "local":
                backend = "local"
            elif "anthropic" in backend.lower() or "claude" in model.lower():
                backend = "anthropic"
            elif "google" in backend.lower() or "gemini" in model.lower():
                backend = "google"
            
            # Calculate costs
            actual_cost = self.calculate_cost(backend, model, prompt_tokens, completion_tokens)
            hypothetical_cost = self.calculate_hypothetical_cloud_cost(prompt_tokens, completion_tokens)
            savings = hypothetical_cost - actual_cost
            
            return RequestCost(
                prompt_id=prompt.id,
                tier=sentinel.get("privacy_tier", prompt.expected_tier),
                route=route,
                backend=backend,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                actual_cost_usd=actual_cost,
                hypothetical_cloud_cost_usd=hypothetical_cost,
                savings_usd=savings,
                success=True,
            )
            
        except Exception as e:
            return RequestCost(
                prompt_id=prompt.id,
                tier=prompt.expected_tier,
                route="error",
                backend="",
                model="",
                success=False,
                error=str(e),
            )
    
    async def run_async(self, dataset: list[LabeledPrompt]) -> CostExperimentResults:
        """Run the cost attribution experiment."""
        results = CostExperimentResults(endpoint=self.endpoint)
        request_costs: list[RequestCost] = []
        
        print(f"Running cost attribution with {len(dataset)} prompts...")
        print(f"Endpoint: {self.endpoint}")
        
        async with httpx.AsyncClient() as client:
            for i, prompt in enumerate(dataset):
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i + 1}/{len(dataset)}")
                
                cost = await self.send_request(client, prompt)
                request_costs.append(cost)
        
        # Aggregate results
        results.total_requests = len(request_costs)
        
        successful = [c for c in request_costs if c.success]
        failed = [c for c in request_costs if not c.success]
        
        results.successful_requests = len(successful)
        results.failed_requests = len(failed)
        results.errors = [{"prompt_id": c.prompt_id, "error": c.error} for c in failed]
        
        if not successful:
            return results
        
        # Token totals
        results.total_prompt_tokens = sum(c.prompt_tokens for c in successful)
        results.total_completion_tokens = sum(c.completion_tokens for c in successful)
        results.total_tokens = results.total_prompt_tokens + results.total_completion_tokens
        
        # Cost totals
        results.actual_total_cost_usd = sum(c.actual_cost_usd for c in successful)
        results.hypothetical_cloud_cost_usd = sum(c.hypothetical_cloud_cost_usd for c in successful)
        results.total_savings_usd = sum(c.savings_usd for c in successful)
        
        if results.hypothetical_cloud_cost_usd > 0:
            results.savings_percentage = 100 * results.total_savings_usd / results.hypothetical_cloud_cost_usd
        
        # Routing breakdown
        results.local_requests = sum(1 for c in successful if c.route == "local")
        results.cloud_requests = sum(1 for c in successful if c.route == "cloud")
        results.local_percentage = 100 * results.local_requests / len(successful)
        
        # Cost efficiency
        results.avg_cost_per_request_usd = results.actual_total_cost_usd / len(successful)
        if results.total_tokens > 0:
            results.avg_cost_per_1k_tokens_usd = (results.actual_total_cost_usd / results.total_tokens) * 1000
        
        # Per-backend breakdown
        backends = set(c.backend for c in successful)
        for backend in backends:
            backend_requests = [c for c in successful if c.backend == backend]
            summary = BackendCostSummary(backend=backend)
            summary.request_count = len(backend_requests)
            summary.prompt_tokens = sum(c.prompt_tokens for c in backend_requests)
            summary.completion_tokens = sum(c.completion_tokens for c in backend_requests)
            summary.total_cost_usd = sum(c.actual_cost_usd for c in backend_requests)
            
            if summary.request_count > 0:
                summary.cost_per_request_usd = summary.total_cost_usd / summary.request_count
            
            total_tokens = summary.prompt_tokens + summary.completion_tokens
            if total_tokens > 0:
                summary.cost_per_1k_tokens_usd = (summary.total_cost_usd / total_tokens) * 1000
            
            results.backend_costs[backend] = {
                "request_count": summary.request_count,
                "prompt_tokens": summary.prompt_tokens,
                "completion_tokens": summary.completion_tokens,
                "total_cost_usd": summary.total_cost_usd,
                "cost_per_request_usd": summary.cost_per_request_usd,
                "cost_per_1k_tokens_usd": summary.cost_per_1k_tokens_usd,
            }
        
        # Per-tier breakdown
        for tier in range(4):
            tier_requests = [c for c in successful if c.tier == tier]
            if not tier_requests:
                continue
            
            summary = TierCostSummary(tier=tier, tier_name=self.TIER_NAMES.get(tier, "?"))
            summary.request_count = len(tier_requests)
            summary.routed_local = sum(1 for c in tier_requests if c.route == "local")
            summary.routed_cloud = sum(1 for c in tier_requests if c.route == "cloud")
            summary.total_cost_usd = sum(c.actual_cost_usd for c in tier_requests)
            summary.savings_usd = sum(c.savings_usd for c in tier_requests)
            summary.avg_cost_per_request_usd = summary.total_cost_usd / summary.request_count
            
            results.tier_costs[tier] = {
                "tier_name": summary.tier_name,
                "request_count": summary.request_count,
                "routed_local": summary.routed_local,
                "routed_cloud": summary.routed_cloud,
                "total_cost_usd": summary.total_cost_usd,
                "savings_usd": summary.savings_usd,
                "avg_cost_per_request_usd": summary.avg_cost_per_request_usd,
            }
        
        # Store individual results
        results.request_costs = [
            {
                "prompt_id": c.prompt_id,
                "tier": c.tier,
                "route": c.route,
                "backend": c.backend,
                "model": c.model,
                "tokens": c.total_tokens,
                "actual_cost_usd": c.actual_cost_usd,
                "savings_usd": c.savings_usd,
            }
            for c in successful
        ]
        
        return results
    
    def run(self, dataset: list[LabeledPrompt]) -> CostExperimentResults:
        """Run the experiment (sync wrapper)."""
        return asyncio.run(self.run_async(dataset))
    
    def print_summary(self, results: CostExperimentResults) -> None:
        """Print experiment summary."""
        print(f"\n{'─'*50}")
        print("COST ATTRIBUTION SUMMARY")
        print(f"{'─'*50}")
        
        print(f"\n📊 Requests:")
        print(f"   Total: {results.total_requests}")
        print(f"   Successful: {results.successful_requests}")
        print(f"   Failed: {results.failed_requests}")
        
        print(f"\n📝 Tokens:")
        print(f"   Prompt: {results.total_prompt_tokens:,}")
        print(f"   Completion: {results.total_completion_tokens:,}")
        print(f"   Total: {results.total_tokens:,}")
        
        print(f"\n💰 Cost Summary:")
        print(f"   Actual cost: ${results.actual_total_cost_usd:.6f}")
        print(f"   If all cloud: ${results.hypothetical_cloud_cost_usd:.6f}")
        print(f"   Savings: ${results.total_savings_usd:.6f} ({results.savings_percentage:.1f}%)")
        
        print(f"\n🔀 Routing:")
        print(f"   Local: {results.local_requests} ({results.local_percentage:.1f}%)")
        print(f"   Cloud: {results.cloud_requests} ({100 - results.local_percentage:.1f}%)")
        
        print(f"\n📈 Cost Efficiency:")
        print(f"   Avg cost/request: ${results.avg_cost_per_request_usd:.6f}")
        print(f"   Avg cost/1K tokens: ${results.avg_cost_per_1k_tokens_usd:.6f}")
        
        if results.backend_costs:
            print(f"\n💵 Cost by Backend:")
            for backend, data in sorted(results.backend_costs.items()):
                count = data["request_count"]
                cost = data["total_cost_usd"]
                per_req = data["cost_per_request_usd"]
                print(f"   {backend}: {count} requests, ${cost:.6f} total (${per_req:.6f}/req)")
        
        if results.tier_costs:
            print(f"\n📊 Cost by Tier:")
            for tier, data in sorted(results.tier_costs.items()):
                name = data["tier_name"]
                count = data["request_count"]
                cost = data["total_cost_usd"]
                savings = data["savings_usd"]
                local_pct = 100 * data["routed_local"] / count if count > 0 else 0
                print(f"   Tier {tier} ({name}): {count} req, ${cost:.6f}, saved ${savings:.6f}, {local_pct:.0f}% local")
    
    def save_results(self, results: CostExperimentResults, output_path: Path) -> None:
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
            },
            "tokens": {
                "prompt": results.total_prompt_tokens,
                "completion": results.total_completion_tokens,
                "total": results.total_tokens,
            },
            "cost": {
                "actual_total_usd": results.actual_total_cost_usd,
                "hypothetical_cloud_usd": results.hypothetical_cloud_cost_usd,
                "savings_usd": results.total_savings_usd,
                "savings_percentage": results.savings_percentage,
            },
            "routing": {
                "local_requests": results.local_requests,
                "cloud_requests": results.cloud_requests,
                "local_percentage": results.local_percentage,
            },
            "efficiency": {
                "avg_cost_per_request_usd": results.avg_cost_per_request_usd,
                "avg_cost_per_1k_tokens_usd": results.avg_cost_per_1k_tokens_usd,
            },
            "backend_costs": results.backend_costs,
            "tier_costs": {str(k): v for k, v in results.tier_costs.items()},
            "errors": results.errors,
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
