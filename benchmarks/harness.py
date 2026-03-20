#!/usr/bin/env python3
"""Benchmark harness for inference-sentinel.

Main entry point for running all benchmark experiments.

Usage:
    # Generate dataset + run all experiments
    python -m benchmarks.harness --full
    
    # Generate dataset only
    python -m benchmarks.harness --generate --count 200
    
    # Run specific experiment
    python -m benchmarks.harness --experiment classification
    
    # Run with larger dataset for final results
    python -m benchmarks.harness --generate --count 1000 --experiment classification
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime


def generate_dataset(count: int, output: Path, seed: int = 42) -> None:
    """Generate synthetic benchmark dataset."""
    print(f"\n{'='*60}")
    print("GENERATING SYNTHETIC DATASET")
    print(f"{'='*60}")
    
    from benchmarks.datasets.generator import generate_dataset as gen, save_dataset
    
    import random
    from faker import Faker
    Faker.seed(seed)
    random.seed(seed)
    
    dataset = gen(count=count)
    save_dataset(dataset, output)


def run_classification_experiment(dataset_path: Path, output_dir: Path, ner_enabled: bool = False) -> dict:
    """Run Experiment 1: Classification Accuracy."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: CLASSIFICATION ACCURACY")
    print(f"{'='*60}")
    
    from benchmarks.datasets.generator import load_dataset
    from benchmarks.experiments.classification import ClassificationExperiment
    
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} prompts")
    print(f"NER enabled: {ner_enabled}")
    
    experiment = ClassificationExperiment(ner_enabled=ner_enabled)
    
    # Initialize NER if enabled (async)
    if ner_enabled:
        import asyncio
        asyncio.run(experiment.classifier.initialize())
    
    results = experiment.run(dataset)
    results.dataset_path = str(dataset_path)
    
    experiment.print_summary(results)
    
    output_path = output_dir / "classification_results.json"
    experiment.save_results(results, output_path)
    
    return {
        "accuracy": results.overall_accuracy,
        "avg_time_ms": results.avg_classification_time_ms,
        "misclassifications": len(results.misclassifications),
    }


def run_routing_experiment(dataset_path: Path, output_dir: Path, endpoint: str) -> dict:
    """Run Experiment 2: Routing Performance (latency/throughput)."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: ROUTING PERFORMANCE")
    print(f"{'='*60}")
    
    from benchmarks.datasets.generator import load_dataset
    from benchmarks.experiments.routing import RoutingExperiment
    
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} prompts")
    print(f"Endpoint: {endpoint}")
    
    experiment = RoutingExperiment(endpoint=endpoint)
    results = experiment.run(dataset)
    results.dataset_path = str(dataset_path)
    
    experiment.print_summary(results)
    
    output_path = output_dir / "routing_results.json"
    experiment.save_results(results, output_path)
    
    return {
        "successful_requests": results.successful_requests,
        "latency_p50_ms": results.latency_p50_ms,
        "latency_p95_ms": results.latency_p95_ms,
        "requests_per_second": results.requests_per_second,
        "routed_local_pct": 100 * results.routed_local / max(1, results.successful_requests),
    }


def run_cost_experiment(dataset_path: Path, output_dir: Path, endpoint: str) -> dict:
    """Run Experiment 3: Cost Attribution."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 3: COST ATTRIBUTION")
    print(f"{'='*60}")
    
    from benchmarks.datasets.generator import load_dataset
    from benchmarks.experiments.cost import CostExperiment
    
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} prompts")
    print(f"Endpoint: {endpoint}")
    
    experiment = CostExperiment(endpoint=endpoint)
    results = experiment.run(dataset)
    results.dataset_path = str(dataset_path)
    
    experiment.print_summary(results)
    
    output_path = output_dir / "cost_results.json"
    experiment.save_results(results, output_path)
    
    return {
        "actual_cost_usd": results.actual_total_cost_usd,
        "hypothetical_cloud_cost_usd": results.hypothetical_cloud_cost_usd,
        "savings_usd": results.total_savings_usd,
        "savings_percentage": results.savings_percentage,
        "local_percentage": results.local_percentage,
    }


def run_controller_experiment(dataset_path: Path, output_dir: Path, endpoint: str) -> dict:
    """Run Experiment 4: Closed-Loop Effectiveness."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 4: CLOSED-LOOP CONTROLLER")
    print(f"{'='*60}")
    
    from benchmarks.datasets.generator import load_dataset
    from benchmarks.experiments.controller import ControllerExperiment
    
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} prompts")
    print(f"Endpoint: {endpoint}")
    
    experiment = ControllerExperiment(endpoint=endpoint)
    results = experiment.run(dataset)
    results.dataset_path = str(dataset_path)
    
    experiment.print_summary(results)
    
    output_path = output_dir / "controller_results.json"
    experiment.save_results(results, output_path)
    
    return {
        "total_requests": results.total_requests_sent,
        "recommendations_count": results.recommendations_count,
        "drift_detected": results.drift_detected,
        "total_cost_usd": results.total_cost_usd,
        "total_savings_usd": results.total_savings_usd,
    }


def run_session_experiment(output_dir: Path, endpoint: str, sessions: int = 20) -> dict:
    """Run Experiment 5: Session Stickiness."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 5: SESSION STICKINESS")
    print(f"{'='*60}")
    
    from benchmarks.experiments.session import SessionExperiment, print_session_summary
    
    experiment = SessionExperiment(
        endpoint=endpoint,
        sessions=sessions,
        requests_per_session=10,
        pii_probability=0.3,
    )
    results = experiment.run()
    
    print_session_summary(results)
    
    output_path = experiment.save_results(results, output_dir)
    print(f"\nResults saved to: {output_path}")
    
    return {
        "sessions_tested": results.sessions_tested,
        "sessions_locked": results.sessions_locked,
        "trapdoor_violations": results.trapdoor_violations,
        "trapdoor_success_rate": results.trapdoor_success_rate,
        "stickiness_success_rate": results.stickiness_success_rate,
        "handoff_overhead_ms": results.handoff_overhead_ms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark harness for inference-sentinel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 200 prompts
  python -m benchmarks.harness --generate --count 200 --experiment classification
  
  # Full benchmark suite with 1000 prompts
  python -m benchmarks.harness --full --count 1000
  
  # Just generate dataset
  python -m benchmarks.harness --generate --count 500
        """,
    )
    
    parser.add_argument(
        "--generate", 
        action="store_true",
        help="Generate synthetic dataset before running experiments",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Number of prompts to generate (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmarks/datasets/privacy_prompts.json",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["classification", "routing", "cost", "controller", "session", "all"],
        default=None,
        help="Which experiment to run",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark suite (generate + all experiments)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000",
        help="Sentinel API endpoint for live experiments",
    )
    parser.add_argument(
        "--ner",
        action="store_true",
        help="Enable NER classifier (requires transformers)",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=20,
        help="Number of sessions for session experiment",
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print("# INFERENCE-SENTINEL BENCHMARK SUITE")
    print(f"# {datetime.now().isoformat()}")
    print(f"{'#'*60}")
    
    # Generate dataset if requested
    if args.generate or args.full:
        generate_dataset(args.count, dataset_path, args.seed)
    
    # Check dataset exists
    if not dataset_path.exists() and args.experiment:
        print(f"\n❌ Dataset not found: {dataset_path}")
        print("   Run with --generate first")
        sys.exit(1)
    
    # Run experiments
    results = {}
    experiments_to_run = []
    
    if args.full:
        experiments_to_run = ["classification", "routing", "cost", "controller", "session"]
    elif args.experiment:
        experiments_to_run = [args.experiment] if args.experiment != "all" else ["classification", "routing", "cost", "controller", "session"]
    
    for exp in experiments_to_run:
        if exp == "classification":
            results["classification"] = run_classification_experiment(dataset_path, output_dir, ner_enabled=args.ner)
        elif exp == "routing":
            results["routing"] = run_routing_experiment(dataset_path, output_dir, args.endpoint)
        elif exp == "cost":
            results["cost"] = run_cost_experiment(dataset_path, output_dir, args.endpoint)
        elif exp == "controller":
            results["controller"] = run_controller_experiment(dataset_path, output_dir, args.endpoint)
        elif exp == "session":
            results["session"] = run_session_experiment(output_dir, args.endpoint, args.sessions)
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        for exp, data in results.items():
            print(f"\n{exp.upper()}:")
            for key, value in data.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        print(f"\nResults saved to: {output_dir}/")
    
    print(f"\n{'#'*60}")
    print("# BENCHMARK COMPLETE")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
