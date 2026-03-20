"""Report generator for benchmark results.

Generates Markdown reports with embedded PNG charts.

Usage:
    python -m benchmarks.report --results-dir benchmarks/results
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def load_classification_results(results_dir: Path) -> dict | None:
    """Load classification experiment results."""
    path = results_dir / "classification_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_routing_results(results_dir: Path) -> dict | None:
    """Load routing experiment results."""
    path = results_dir / "routing_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_controller_results(results_dir: Path) -> dict | None:
    """Load controller experiment results."""
    path = results_dir / "controller_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_cost_results(results_dir: Path) -> dict | None:
    """Load cost experiment results."""
    path = results_dir / "cost_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def generate_confusion_matrix_chart(confusion_matrix: dict, output_path: Path) -> None:
    """Generate confusion matrix heatmap."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠️  matplotlib not installed, skipping chart generation")
        return
    
    matrix = np.array(confusion_matrix["matrix"])
    labels = ["Tier 0\n(PUBLIC)", "Tier 1\n(INTERNAL)", "Tier 2\n(CONF.)", "Tier 3\n(RESTR.)"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    
    # Labels
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Classification Confusion Matrix", fontsize=14, fontweight="bold")
    
    # Annotate cells
    for i in range(4):
        for j in range(4):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=14)
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {output_path}")


def generate_tier_metrics_chart(tier_metrics: dict, output_path: Path) -> None:
    """Generate bar chart of precision/recall/F1 per tier."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠️  matplotlib not installed, skipping chart generation")
        return
    
    tiers = sorted(tier_metrics.keys(), key=int)
    precision = [tier_metrics[t]["precision"] for t in tiers]
    recall = [tier_metrics[t]["recall"] for t in tiers]
    f1 = [tier_metrics[t]["f1"] for t in tiers]
    
    x = np.arange(4)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, precision, width, label="Precision", color="#2ecc71")
    bars2 = ax.bar(x, recall, width, label="Recall", color="#3498db")
    bars3 = ax.bar(x + width, f1, width, label="F1 Score", color="#9b59b6")
    
    ax.set_xlabel("Privacy Tier", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classification Metrics by Tier", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "Tier 0\n(PUBLIC)",
        "Tier 1\n(INTERNAL)", 
        "Tier 2\n(CONFIDENTIAL)",
        "Tier 3\n(RESTRICTED)",
    ])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="95% target")
    
    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.0%}",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {output_path}")


def generate_latency_chart(routing_results: dict, output_path: Path) -> None:
    """Generate latency percentiles bar chart."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠️  matplotlib not installed, skipping chart generation")
        return
    
    latency = routing_results["latency"]
    
    percentiles = ["p50", "p95", "p99", "Mean"]
    values = [
        latency["p50_ms"],
        latency["p95_ms"],
        latency["p99_ms"],
        latency["mean_ms"],
    ]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(percentiles, values, color=["#3498db", "#e67e22", "#e74c3c", "#9b59b6"])
    
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("End-to-End Latency Distribution", fontsize=14, fontweight="bold")
    
    # Value labels
    for bar, val in zip(bars, values):
        ax.annotate(f"{val:.0f}ms",
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {output_path}")


def generate_routing_distribution_chart(routing_results: dict, output_path: Path) -> None:
    """Generate pie chart of routing distribution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not installed, skipping chart generation")
        return
    
    routing = routing_results["routing"]
    
    labels = ["Local", "Cloud"]
    sizes = [routing["local"], routing["cloud"]]
    colors = ["#2ecc71", "#3498db"]
    explode = (0.05, 0)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 12})
    ax.set_title("Routing Distribution", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {output_path}")


def generate_cost_comparison_chart(cost_results: dict, output_path: Path) -> None:
    """Generate cost comparison bar chart (actual vs hypothetical)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠️  matplotlib not installed, skipping chart generation")
        return
    
    cost = cost_results["cost"]
    
    categories = ["Actual Cost", "If All Cloud", "Savings"]
    values = [
        cost["actual_total_usd"],
        cost["hypothetical_cloud_usd"],
        cost["savings_usd"],
    ]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(categories, values, color=colors)
    
    ax.set_ylabel("Cost (USD)", fontsize=12)
    ax.set_title("Cost Attribution: Actual vs Hypothetical", fontsize=14, fontweight="bold")
    
    # Value labels
    for bar, val in zip(bars, values):
        ax.annotate(f"${val:.4f}",
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    # Add savings percentage annotation
    if cost["hypothetical_cloud_usd"] > 0:
        savings_pct = cost["savings_percentage"]
        ax.annotate(f"({savings_pct:.1f}% saved)",
                   xy=(2, values[2]),
                   xytext=(0, 20),
                   textcoords="offset points",
                   ha="center", va="bottom", fontsize=10, color="#3498db")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {output_path}")


def generate_cost_by_backend_chart(cost_results: dict, output_path: Path) -> None:
    """Generate cost by backend pie chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not installed, skipping chart generation")
        return
    
    backend_costs = cost_results.get("backend_costs", {})
    if not backend_costs:
        return
    
    labels = list(backend_costs.keys())
    sizes = [data["total_cost_usd"] for data in backend_costs.values()]
    
    # Skip if all zeros
    if sum(sizes) == 0:
        return
    
    colors = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"][:len(labels)]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 11})
    ax.set_title("Cost Distribution by Backend", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {output_path}")


def generate_markdown_report(results_dir: Path, output_path: Path) -> None:
    """Generate full Markdown benchmark report."""
    
    classification = load_classification_results(results_dir)
    routing = load_routing_results(results_dir)
    controller = load_controller_results(results_dir)
    cost = load_cost_results(results_dir)
    
    # Generate charts
    images_dir = results_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    if classification:
        generate_confusion_matrix_chart(
            classification["confusion_matrix"],
            images_dir / "confusion_matrix.png"
        )
        generate_tier_metrics_chart(
            classification["tier_metrics"],
            images_dir / "tier_metrics.png"
        )
    
    if routing:
        generate_latency_chart(routing, images_dir / "latency_distribution.png")
        generate_routing_distribution_chart(routing, images_dir / "routing_distribution.png")
    
    if cost:
        generate_cost_comparison_chart(cost, images_dir / "cost_comparison.png")
        generate_cost_by_backend_chart(cost, images_dir / "cost_by_backend.png")
    
    # Build report
    lines = [
        "# Inference-Sentinel Benchmark Results",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]
    
    # Classification section
    if classification:
        summary = classification["summary"]
        tier_metrics = classification["tier_metrics"]
        
        lines.extend([
            "## Experiment 1: Classification Accuracy",
            "",
            "Measures the precision, recall, and F1 score of the privacy classifier",
            "against a synthetic dataset with ground-truth labels.",
            "",
            "### Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Total Prompts** | {summary['total_prompts']} |",
            f"| **Overall Accuracy** | {summary['overall_accuracy']:.1%} |",
            f"| **Avg Classification Time** | {summary['avg_classification_time_ms']:.2f} ms |",
            "",
            "### Per-Tier Metrics",
            "",
            "| Tier | Precision | Recall | F1 Score |",
            "|------|-----------|--------|----------|",
        ])
        
        for tier in sorted(tier_metrics.keys(), key=int):
            tm = tier_metrics[tier]
            lines.append(
                f"| Tier {tier} ({tm['tier_name']}) | "
                f"{tm['precision']:.1%} | {tm['recall']:.1%} | {tm['f1']:.1%} |"
            )
        
        lines.extend([
            "",
            "### Visualizations",
            "",
            "#### Metrics by Tier",
            "",
            "![Tier Metrics](images/tier_metrics.png)",
            "",
            "#### Confusion Matrix",
            "",
            "![Confusion Matrix](images/confusion_matrix.png)",
            "",
        ])
        
        # Misclassifications
        if classification.get("misclassifications"):
            lines.extend([
                "### Sample Misclassifications",
                "",
                "| Prompt ID | Expected | Predicted | Text Preview |",
                "|-----------|----------|-----------|--------------|",
            ])
            for mc in classification["misclassifications"][:10]:
                text_preview = mc["text_preview"][:50].replace("|", "\\|")
                lines.append(
                    f"| `{mc['prompt_id']}` | {mc['expected']} | {mc['predicted']} | {text_preview}... |"
                )
            lines.append("")
    
    # Placeholder for other experiments
    lines.extend([
        "---",
        "",
    ])
    
    # Routing section
    routing = load_routing_results(results_dir)
    if routing:
        summary = routing["summary"]
        latency = routing["latency"]
        routing_data = routing["routing"]
        
        lines.extend([
            "## Experiment 2: Routing Performance",
            "",
            "Measures end-to-end latency and throughput through the Sentinel API.",
            "",
            "### Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Total Requests** | {summary['total_requests']} |",
            f"| **Successful** | {summary['successful_requests']} |",
            f"| **Failed** | {summary['failed_requests']} |",
            f"| **Duration** | {summary['total_duration_seconds']:.1f}s |",
            f"| **Throughput** | {summary['requests_per_second']:.2f} req/s |",
            "",
            "### Latency (End-to-End)",
            "",
            "| Percentile | Latency |",
            "|------------|---------|",
            f"| **p50** | {latency['p50_ms']:.0f} ms |",
            f"| **p95** | {latency['p95_ms']:.0f} ms |",
            f"| **p99** | {latency['p99_ms']:.0f} ms |",
            f"| **Mean** | {latency['mean_ms']:.0f} ms |",
            "",
            "### Component Latencies",
            "",
            "| Component | Mean Latency |",
            "|-----------|--------------|",
            f"| Classification | {routing['component_latency']['classification_mean_ms']:.2f} ms |",
            f"| Routing | {routing['component_latency']['routing_mean_ms']:.2f} ms |",
            f"| Inference | {routing['component_latency']['inference_mean_ms']:.0f} ms |",
            "",
            "### Routing Distribution",
            "",
            f"| Route | Count | Percentage |",
            f"|-------|-------|------------|",
        ])
        
        total = routing_data["local"] + routing_data["cloud"]
        local_pct = 100 * routing_data["local"] / max(1, total)
        cloud_pct = 100 * routing_data["cloud"] / max(1, total)
        lines.append(f"| Local | {routing_data['local']} | {local_pct:.1f}% |")
        lines.append(f"| Cloud | {routing_data['cloud']} | {cloud_pct:.1f}% |")
        
        lines.extend([
            "",
            "### Cost",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Total Cost** | ${routing['cost']['total_usd']:.4f} |",
            f"| **Savings (Local)** | ${routing['cost']['savings_usd']:.4f} |",
            "",
            "### Visualizations",
            "",
            "#### Latency Distribution",
            "",
            "![Latency Distribution](images/latency_distribution.png)",
            "",
            "#### Routing Distribution",
            "",
            "![Routing Distribution](images/routing_distribution.png)",
            "",
        ])
    else:
        lines.extend([
            "## Experiment 2: Routing Performance",
            "",
            "*Not yet run. Use: `python -m benchmarks.harness --experiment routing`*",
            "",
        ])
    
    # Cost section
    cost_data = load_cost_results(results_dir)
    if cost_data:
        cost = cost_data["cost"]
        tokens = cost_data["tokens"]
        routing = cost_data["routing"]
        efficiency = cost_data["efficiency"]
        backend_costs = cost_data.get("backend_costs", {})
        tier_costs = cost_data.get("tier_costs", {})
        
        lines.extend([
            "## Experiment 3: Cost Attribution",
            "",
            "Compares actual costs (with privacy routing) vs hypothetical all-cloud costs.",
            "",
            "### Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Total Requests** | {cost_data['summary']['successful_requests']} |",
            f"| **Total Tokens** | {tokens['total']:,} |",
            f"| **Actual Cost** | ${cost['actual_total_usd']:.6f} |",
            f"| **If All Cloud** | ${cost['hypothetical_cloud_usd']:.6f} |",
            f"| **Savings** | ${cost['savings_usd']:.6f} ({cost['savings_percentage']:.1f}%) |",
            "",
            "### Routing Distribution",
            "",
            "| Route | Count | Percentage |",
            "|-------|-------|------------|",
            f"| Local | {routing['local_requests']} | {routing['local_percentage']:.1f}% |",
            f"| Cloud | {routing['cloud_requests']} | {100 - routing['local_percentage']:.1f}% |",
            "",
            "### Cost Efficiency",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Avg Cost/Request** | ${efficiency['avg_cost_per_request_usd']:.6f} |",
            f"| **Avg Cost/1K Tokens** | ${efficiency['avg_cost_per_1k_tokens_usd']:.6f} |",
            "",
        ])
        
        if backend_costs:
            lines.extend([
                "### Cost by Backend",
                "",
                "| Backend | Requests | Total Cost | Cost/Request |",
                "|---------|----------|------------|--------------|",
            ])
            for backend, data in sorted(backend_costs.items()):
                lines.append(
                    f"| {backend} | {data['request_count']} | "
                    f"${data['total_cost_usd']:.6f} | ${data['cost_per_request_usd']:.6f} |"
                )
            lines.append("")
        
        if tier_costs:
            lines.extend([
                "### Cost by Tier",
                "",
                "| Tier | Requests | Local % | Total Cost | Savings |",
                "|------|----------|---------|------------|---------|",
            ])
            for tier, data in sorted(tier_costs.items(), key=lambda x: int(x[0])):
                local_pct = 100 * data['routed_local'] / max(1, data['request_count'])
                lines.append(
                    f"| Tier {tier} ({data['tier_name']}) | {data['request_count']} | "
                    f"{local_pct:.0f}% | ${data['total_cost_usd']:.6f} | ${data['savings_usd']:.6f} |"
                )
            lines.append("")
        
        lines.extend([
            "### Visualizations",
            "",
            "#### Cost Comparison",
            "",
            "![Cost Comparison](images/cost_comparison.png)",
            "",
            "#### Cost by Backend",
            "",
            "![Cost by Backend](images/cost_by_backend.png)",
            "",
        ])
    else:
        lines.extend([
            "## Experiment 3: Cost Attribution",
            "",
            "*Not yet run. Use: `python -m benchmarks.harness --experiment cost`*",
            "",
        ])
    
    # Controller section
    controller = load_controller_results(results_dir)
    if controller:
        traffic = controller["traffic"]
        routing_dist = controller["routing_distribution"]
        cost = controller["cost"]
        recommendations = controller["recommendations"]
        drift = controller["drift"]
        
        lines.extend([
            "## Experiment 4: Closed-Loop Controller",
            "",
            "Measures the controller's ability to detect patterns, generate recommendations,",
            "and track cost savings from routing decisions.",
            "",
            "### Traffic Generated",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Total Requests** | {traffic['total_requests']} |",
            f"| **Successful** | {traffic['successful']} |",
            "",
            "### Routing Distribution",
            "",
            "| Route | Count |",
            "|-------|-------|",
        ])
        
        for route, count in routing_dist.items():
            lines.append(f"| {route.capitalize()} | {count} |")
        
        lines.extend([
            "",
            "### Controller Recommendations",
            "",
            f"**Total Recommendations:** {recommendations['count']}",
            "",
        ])
        
        if recommendations["items"]:
            lines.extend([
                "| Type | Tier | Reason |",
                "|------|------|--------|",
            ])
            for rec in recommendations["items"][:10]:
                rec_type = rec.get("type", "unknown")
                tier = rec.get("tier", "?")
                reason = rec.get("reason", "")[:60]
                lines.append(f"| {rec_type} | {tier} | {reason}... |")
            lines.append("")
        
        lines.extend([
            "### Drift Detection",
            "",
            f"**Drift Detected:** {'Yes ⚠️' if drift['detected'] else 'No ✓'}",
            "",
        ])
        
        if drift["detected"] and drift["details"]:
            lines.append("**Details:**")
            for detail in drift["details"]:
                lines.append(f"- {detail.get('reason', 'Unknown')}")
            lines.append("")
        
        lines.extend([
            "### Cost Analysis",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Total Cost** | ${cost['total_usd']:.4f} |",
            f"| **Savings (Local Routing)** | ${cost['savings_usd']:.4f} |",
            f"| **Potential Additional Savings** | ${cost['potential_additional_savings_usd']:.4f} |",
            "",
        ])
    else:
        lines.extend([
            "## Experiment 4: Closed-Loop Controller",
            "",
            "*Not yet run. Use: `python -m benchmarks.harness --experiment controller`*",
            "",
        ])
    
    lines.extend([
        "---",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "# Generate dataset",
        "python -m benchmarks.harness --generate --count 200",
        "",
        "# Run all experiments",
        "python -m benchmarks.harness --experiment classification --ner",
        "python -m benchmarks.harness --experiment routing",
        "python -m benchmarks.harness --experiment cost",
        "python -m benchmarks.harness --experiment controller",
        "",
        "# Generate report",
        "python -m benchmarks.report",
        "```",
        "",
    ])
    
    # Write report
    report_content = "\n".join(lines)
    output_path.write_text(report_content)
    print(f"\n📄 Report generated → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/BENCHMARK_RESULTS.md",
        help="Output path for Markdown report",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("   Run experiments first: python -m benchmarks.harness --experiment classification")
        return
    
    print("\n" + "=" * 60)
    print("GENERATING BENCHMARK REPORT")
    print("=" * 60)
    
    generate_markdown_report(results_dir, output_path)


if __name__ == "__main__":
    main()
