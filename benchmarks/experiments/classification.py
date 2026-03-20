"""Experiment 1: Classification Accuracy.

Measures precision, recall, and F1 score of the privacy classifier
against a labeled synthetic dataset.

Usage:
    python -m benchmarks.experiments.classification --dataset benchmarks/datasets/privacy_prompts.json
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

from benchmarks.datasets.generator import LabeledPrompt, load_dataset


@dataclass
class TierMetrics:
    """Metrics for a single privacy tier."""
    tier: int
    tier_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    
    @property
    def precision(self) -> float:
        """TP / (TP + FP)"""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """TP / (TP + FN)"""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def f1(self) -> float:
        """2 * (P * R) / (P + R)"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        """(TP + TN) / total"""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0


@dataclass
class ConfusionMatrix:
    """4x4 confusion matrix for tier classification."""
    matrix: list[list[int]] = field(default_factory=lambda: [[0]*4 for _ in range(4)])
    
    def add(self, expected: int, predicted: int) -> None:
        """Add a prediction to the matrix."""
        self.matrix[expected][predicted] += 1
    
    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "labels": ["Tier 0 (PUBLIC)", "Tier 1 (INTERNAL)", "Tier 2 (CONFIDENTIAL)", "Tier 3 (RESTRICTED)"],
            "matrix": self.matrix,
        }


@dataclass 
class ClassificationResult:
    """Result of classifying a single prompt."""
    prompt_id: str
    expected_tier: int
    predicted_tier: int
    correct: bool
    classification_time_ms: float
    detected_entities: list[str] = field(default_factory=list)


@dataclass
class ExperimentResults:
    """Full results of the classification experiment."""
    timestamp: str
    dataset_path: str
    total_prompts: int
    overall_accuracy: float
    tier_metrics: dict[int, TierMetrics]
    confusion_matrix: ConfusionMatrix
    individual_results: list[ClassificationResult]
    avg_classification_time_ms: float
    
    # Errors
    misclassifications: list[dict] = field(default_factory=list)


class ClassificationExperiment:
    """Experiment 1: Classification Accuracy."""
    
    TIER_NAMES = {
        0: "PUBLIC",
        1: "INTERNAL", 
        2: "CONFIDENTIAL",
        3: "RESTRICTED",
    }
    
    def __init__(self, classifier=None, ner_enabled: bool = False):
        """Initialize experiment.
        
        Args:
            classifier: Optional classifier instance. If None, creates default.
            ner_enabled: Whether to enable NER classifier (requires transformers).
        """
        self.classifier = classifier
        self.ner_enabled = ner_enabled
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize the classifier if not provided."""
        if self.classifier is None:
            # Import here to avoid circular imports
            from sentinel.classification import HybridClassifier
            
            self.classifier = HybridClassifier(ner_enabled=self.ner_enabled)
    
    def classify_prompt(self, text: str) -> tuple[int, list[str], float]:
        """Classify a single prompt.
        
        Returns:
            Tuple of (predicted_tier, detected_entities, time_ms)
        """
        start = time.perf_counter()
        result = self.classifier.classify_sync(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Extract detected entity types from HybridResult
        entities = result.entity_types if result.entity_types else []
        
        return result.tier, entities, elapsed_ms
    
    def run(self, dataset: list[LabeledPrompt]) -> ExperimentResults:
        """Run the classification experiment.
        
        Args:
            dataset: List of labeled prompts with ground truth.
        
        Returns:
            ExperimentResults with full metrics.
        """
        from datetime import datetime
        
        confusion = ConfusionMatrix()
        tier_metrics = {
            tier: TierMetrics(tier=tier, tier_name=name)
            for tier, name in self.TIER_NAMES.items()
        }
        results = []
        misclassifications = []
        total_time_ms = 0.0
        
        for prompt in dataset:
            predicted_tier, entities, time_ms = self.classify_prompt(prompt.text)
            total_time_ms += time_ms
            
            correct = predicted_tier == prompt.expected_tier
            
            result = ClassificationResult(
                prompt_id=prompt.id,
                expected_tier=prompt.expected_tier,
                predicted_tier=predicted_tier,
                correct=correct,
                classification_time_ms=time_ms,
                detected_entities=entities,
            )
            results.append(result)
            
            # Update confusion matrix
            confusion.add(prompt.expected_tier, predicted_tier)
            
            # Track misclassifications for analysis
            if not correct:
                misclassifications.append({
                    "prompt_id": prompt.id,
                    "text_preview": prompt.text[:100] + "..." if len(prompt.text) > 100 else prompt.text,
                    "expected": f"Tier {prompt.expected_tier} ({prompt.tier_name})",
                    "predicted": f"Tier {predicted_tier} ({self.TIER_NAMES[predicted_tier]})",
                    "expected_entities": [e.type for e in prompt.entities],
                    "detected_entities": entities,
                })
        
        # Calculate per-tier metrics from confusion matrix
        for tier in range(4):
            tm = tier_metrics[tier]
            # TP: predicted == expected == tier
            tm.true_positives = confusion.matrix[tier][tier]
            # FP: predicted == tier but expected != tier
            tm.false_positives = sum(confusion.matrix[other][tier] for other in range(4) if other != tier)
            # FN: expected == tier but predicted != tier
            tm.false_negatives = sum(confusion.matrix[tier][other] for other in range(4) if other != tier)
            # TN: expected != tier and predicted != tier
            tm.true_negatives = sum(
                confusion.matrix[exp][pred]
                for exp in range(4) for pred in range(4)
                if exp != tier and pred != tier
            )
        
        # Overall accuracy
        correct_count = sum(1 for r in results if r.correct)
        overall_accuracy = correct_count / len(results) if results else 0.0
        
        return ExperimentResults(
            timestamp=datetime.utcnow().isoformat(),
            dataset_path="",  # Set by caller
            total_prompts=len(dataset),
            overall_accuracy=overall_accuracy,
            tier_metrics=tier_metrics,
            confusion_matrix=confusion,
            individual_results=results,
            avg_classification_time_ms=total_time_ms / len(results) if results else 0.0,
            misclassifications=misclassifications,
        )
    
    def save_results(self, results: ExperimentResults, output_path: Path) -> None:
        """Save experiment results to JSON."""
        data = {
            "experiment": "classification_accuracy",
            "timestamp": results.timestamp,
            "dataset_path": results.dataset_path,
            "summary": {
                "total_prompts": results.total_prompts,
                "overall_accuracy": round(results.overall_accuracy, 4),
                "avg_classification_time_ms": round(results.avg_classification_time_ms, 3),
            },
            "tier_metrics": {
                tier: {
                    "tier": tm.tier,
                    "tier_name": tm.tier_name,
                    "precision": round(tm.precision, 4),
                    "recall": round(tm.recall, 4),
                    "f1": round(tm.f1, 4),
                    "true_positives": tm.true_positives,
                    "false_positives": tm.false_positives,
                    "false_negatives": tm.false_negatives,
                }
                for tier, tm in results.tier_metrics.items()
            },
            "confusion_matrix": results.confusion_matrix.to_dict(),
            "misclassifications": results.misclassifications[:20],  # First 20 for analysis
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved → {output_path}")
    
    def print_summary(self, results: ExperimentResults) -> None:
        """Print a summary of results to console."""
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: CLASSIFICATION ACCURACY")
        print("=" * 60)
        
        print(f"\nDataset: {results.total_prompts} prompts")
        print(f"Overall Accuracy: {results.overall_accuracy:.1%}")
        print(f"Avg Classification Time: {results.avg_classification_time_ms:.2f}ms")
        
        print("\n" + "-" * 60)
        print("PER-TIER METRICS")
        print("-" * 60)
        print(f"{'Tier':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 60)
        
        for tier in sorted(results.tier_metrics.keys()):
            tm = results.tier_metrics[tier]
            print(f"Tier {tier} ({tm.tier_name:<15}) {tm.precision:>10.1%} {tm.recall:>10.1%} {tm.f1:>10.1%}")
        
        print("\n" + "-" * 60)
        print("CONFUSION MATRIX")
        print("-" * 60)
        print(f"{'':>12} | {'Pred 0':>8} {'Pred 1':>8} {'Pred 2':>8} {'Pred 3':>8}")
        print("-" * 60)
        
        for i, row in enumerate(results.confusion_matrix.matrix):
            label = f"Actual {i}"
            print(f"{label:>12} | {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8}")
        
        if results.misclassifications:
            print("\n" + "-" * 60)
            print(f"SAMPLE MISCLASSIFICATIONS ({len(results.misclassifications)} total)")
            print("-" * 60)
            for mc in results.misclassifications[:5]:
                print(f"\n  [{mc['prompt_id']}]")
                print(f"  Text: {mc['text_preview']}")
                print(f"  Expected: {mc['expected']} | Predicted: {mc['predicted']}")
        
        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run classification accuracy experiment")
    parser.add_argument("--dataset", type=str, default="benchmarks/datasets/privacy_prompts.json")
    parser.add_argument("--output", type=str, default="benchmarks/results/classification_results.json")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Generate it first: python -m benchmarks.datasets.generator")
        return
    
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} prompts from {dataset_path}")
    
    # Run experiment
    experiment = ClassificationExperiment()
    results = experiment.run(dataset)
    results.dataset_path = str(dataset_path)
    
    # Print summary
    experiment.print_summary(results)
    
    # Save results
    experiment.save_results(results, Path(args.output))


if __name__ == "__main__":
    main()
