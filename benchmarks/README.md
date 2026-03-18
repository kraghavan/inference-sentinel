# Benchmark Suite

Reproducible benchmark methodology for inference-sentinel.

## Quick Start

```bash
# Install additional dependencies
pip install faker matplotlib seaborn

# Generate dataset + run classification experiment
python -m benchmarks.harness --generate --count 200 --experiment classification

# Generate report with charts
python -m benchmarks.report
```

## Experiments

### Experiment 1: Classification Accuracy
**Status:** ✅ Implemented

Measures precision, recall, and F1 score of the privacy classifier against a synthetic dataset with ground-truth labels.

```bash
python -m benchmarks.harness --generate --count 200 --experiment classification
```

### Experiment 2: Routing Performance
**Status:** 🔜 Planned

Measures latency percentiles (p50, p95, p99) and throughput for local vs cloud routing.

### Experiment 3: Cost Attribution
**Status:** 🔜 Planned

Tracks actual vs estimated costs from shadow mode comparisons.

### Experiment 4: Closed-Loop Effectiveness
**Status:** 🔜 Planned

Measures controller recommendation accuracy and drift detection.

## Dataset Generation

The synthetic dataset uses [Faker](https://faker.readthedocs.io/) to generate prompts with known PII:

| Tier | Name | Example PII |
|------|------|-------------|
| 0 | PUBLIC | None |
| 1 | INTERNAL | Employee IDs, project codes |
| 2 | CONFIDENTIAL | Emails, phone numbers, addresses |
| 3 | RESTRICTED | SSN, credit cards, medical records |

### Generate Custom Dataset

```bash
# Small dataset for development
python -m benchmarks.datasets.generator --count 200

# Large dataset for final benchmarks
python -m benchmarks.datasets.generator --count 1000 --seed 42
```

## Report Generation

```bash
# Generate Markdown report with PNG charts
python -m benchmarks.report

# View report
cat benchmarks/results/BENCHMARK_RESULTS.md
```

## Directory Structure

```
benchmarks/
├── __init__.py
├── harness.py              # Main experiment runner
├── report.py               # Report generator
├── datasets/
│   ├── __init__.py
│   ├── generator.py        # Faker-based synthetic data
│   └── privacy_prompts.json  # Generated dataset
├── experiments/
│   ├── __init__.py
│   ├── classification.py   # Exp 1: Accuracy
│   ├── routing.py          # Exp 2: Latency (TODO)
│   ├── cost.py             # Exp 3: Cost (TODO)
│   └── controller.py       # Exp 4: Controller (TODO)
└── results/
    ├── classification_results.json
    ├── BENCHMARK_RESULTS.md
    └── images/
        ├── confusion_matrix.png
        └── tier_metrics.png
```

## Reproducibility

All experiments use fixed random seeds (default: 42) for reproducibility.

```bash
# Reproduce exact results
python -m benchmarks.harness --generate --count 200 --seed 42 --experiment classification
```

## Cost Optimization

For large-scale benchmarks hitting live APIs, use cheaper models:

```bash
# Set environment variables
export SENTINEL_CLOUD__ANTHROPIC_MODEL=claude-3-5-haiku-20241022
export SENTINEL_CLOUD__GOOGLE_MODEL=gemini-2.0-flash-lite
```
