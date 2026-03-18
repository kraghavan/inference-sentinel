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


def generate_markdown_report(results_dir: Path, output_path: Path) -> None:
    """Generate full Markdown benchmark report."""
    
    classification = load_classification_results(results_dir)
    
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
        "## Experiment 2: Routing Performance",
        "",
        "*Coming soon: Latency and throughput measurements*",
        "",
        "## Experiment 3: Cost Attribution",
        "",
        "*Coming soon: Cloud vs local cost tracking*",
        "",
        "## Experiment 4: Closed-Loop Controller",
        "",
        "*Coming soon: Controller recommendation effectiveness*",
        "",
        "---",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "# Generate dataset",
        "python -m benchmarks.harness --generate --count 200",
        "",
        "# Run classification experiment",
        "python -m benchmarks.harness --experiment classification",
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
