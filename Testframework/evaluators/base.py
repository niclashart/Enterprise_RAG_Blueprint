"""Base evaluator interface for all RAG component evaluators."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationResult:
    """Container for evaluation results from any evaluator."""

    evaluator_name: str
    component_name: str
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    details: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of results."""
        lines = [
            f"=== {self.evaluator_name}: {self.component_name} ===",
            f"Config: {self.config}",
            f"Duration: {self.duration_seconds:.2f}s",
            "Metrics:",
        ]
        for name, value in self.metrics.items():
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)


@dataclass
class ComparisonResult:
    """Container for comparing two or more component configurations."""

    evaluator_name: str
    results: List[EvaluationResult] = field(default_factory=list)
    winner: Optional[str] = None
    ranking_metric: Optional[str] = None

    def add_result(self, result: EvaluationResult) -> None:
        self.results.append(result)

    def rank_by(self, metric: str, higher_is_better: bool = True) -> List[EvaluationResult]:
        """Rank results by a specific metric."""
        self.ranking_metric = metric
        ranked = sorted(
            self.results,
            key=lambda r: r.metrics.get(metric, 0.0),
            reverse=higher_is_better,
        )
        if ranked:
            self.winner = ranked[0].component_name
        return ranked

    def summary(self) -> str:
        """Human-readable comparison summary."""
        lines = [f"=== Comparison: {self.evaluator_name} ==="]
        if self.ranking_metric:
            lines.append(f"Ranked by: {self.ranking_metric}")
        if self.winner:
            lines.append(f"Winner: {self.winner}")
        lines.append("")
        for i, result in enumerate(self.results, 1):
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in result.metrics.items())
            lines.append(f"  #{i} {result.component_name}: {metric_str}")
        return "\n".join(lines)


class BaseEvaluator(ABC):
    """Abstract base class for all RAG component evaluators.

    Usage:
        from Testframework.evaluators import RetrievalEvaluator

        evaluator = RetrievalEvaluator()
        result = evaluator.evaluate(retriever, dataset)
        comparison = evaluator.compare([retriever_a, retriever_b], dataset)
    """

    name: str = "BaseEvaluator"

    @abstractmethod
    def evaluate(self, component: Any, dataset: Any, **kwargs) -> EvaluationResult:
        """Evaluate a single component against a dataset.

        Args:
            component: The RAG component to evaluate.
            dataset: Test data to evaluate against.
            **kwargs: Additional configuration.

        Returns:
            EvaluationResult with metrics and details.
        """
        ...

    def compare(
        self, components: List[Dict[str, Any]], dataset: Any, rank_metric: str = None, **kwargs
    ) -> ComparisonResult:
        """Compare multiple component configurations.

        Args:
            components: List of dicts with 'name' and 'component' keys.
            dataset: Test data to evaluate against.
            rank_metric: Metric name to rank by. If None, uses first metric.
            **kwargs: Additional configuration.

        Returns:
            ComparisonResult with ranked results.
        """
        comparison = ComparisonResult(evaluator_name=self.name)
        for comp_info in components:
            name = comp_info["name"]
            component = comp_info["component"]
            result = self.evaluate(component, dataset, component_name=name, **kwargs)
            comparison.add_result(result)

        if rank_metric and comparison.results:
            comparison.rank_by(rank_metric)
        elif comparison.results and comparison.results[0].metrics:
            first_metric = list(comparison.results[0].metrics.keys())[0]
            comparison.rank_by(first_metric)

        return comparison

    def _timed_evaluate(self, func, *args, **kwargs) -> tuple:
        """Run a function and return (result, duration_seconds)."""
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        return result, duration
