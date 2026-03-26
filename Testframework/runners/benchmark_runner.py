"""Runner for executing evaluation benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..evaluators.base import BaseEvaluator, EvaluationResult


@dataclass
class BenchmarkSuite:
    """A collection of evaluations to run as a benchmark."""

    name: str
    evaluations: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, evaluator: BaseEvaluator, component: Any, dataset: Any, component_name: str = "", **kwargs):
        """Add an evaluation to the suite.

        Args:
            evaluator: The evaluator instance to use.
            component: The component to evaluate.
            dataset: The dataset to evaluate against.
            component_name: Name for this evaluation.
            **kwargs: Additional kwargs passed to evaluator.evaluate().
        """
        self.evaluations.append({
            "evaluator": evaluator,
            "component": component,
            "dataset": dataset,
            "component_name": component_name or f"eval_{len(self.evaluations)}",
            "kwargs": kwargs,
        })


class BenchmarkRunner:
    """Execute benchmark suites and collect results.

    Example:
        from Testframework.runners import BenchmarkRunner
        from Testframework.evaluators import RetrievalEvaluator, GenerationEvaluator

        suite = BenchmarkSuite("my_benchmark")
        suite.add(RetrievalEvaluator(k=5), retriever, retrieval_dataset, "retriever_v1")
        suite.add(GenerationEvaluator(), generator, gen_dataset, "gpt4_generator")

        runner = BenchmarkRunner()
        results = runner.run(suite)

        for result in results:
            print(result.summary())
    """

    def run(self, suite: BenchmarkSuite, verbose: bool = True) -> List[EvaluationResult]:
        """Run all evaluations in a benchmark suite.

        Args:
            suite: The benchmark suite to execute.
            verbose: If True, print progress to stdout.

        Returns:
            List of EvaluationResult objects.
        """
        results = []
        total = len(suite.evaluations)

        if verbose:
            print(f"Running benchmark: {suite.name} ({total} evaluations)")
            print("=" * 60)

        for i, evaluation in enumerate(suite.evaluations, 1):
            evaluator = evaluation["evaluator"]
            component = evaluation["component"]
            dataset = evaluation["dataset"]
            name = evaluation["component_name"]
            kwargs = evaluation["kwargs"]

            if verbose:
                print(f"[{i}/{total}] {evaluator.name}: {name}...", end=" ", flush=True)

            try:
                result = evaluator.evaluate(
                    component, dataset, component_name=name, **kwargs
                )
                results.append(result)

                if verbose:
                    metric_str = ", ".join(f"{k}={v:.4f}" for k, v in list(result.metrics.items())[:3])
                    print(f"OK ({result.duration_seconds:.2f}s) [{metric_str}]")
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")
                results.append(EvaluationResult(
                    evaluator_name=evaluator.name,
                    component_name=name,
                    metrics={},
                    metadata={"error": str(e)},
                ))

        if verbose:
            print("=" * 60)
            print(f"Benchmark complete: {len(results)}/{total} evaluations finished")

        return results

    def run_multiple(self, suites: List[BenchmarkSuite], verbose: bool = True) -> Dict[str, List[EvaluationResult]]:
        """Run multiple benchmark suites.

        Args:
            suites: List of benchmark suites.
            verbose: If True, print progress.

        Returns:
            Dict mapping suite names to their results.
        """
        all_results = {}
        for suite in suites:
            all_results[suite.name] = self.run(suite, verbose=verbose)
        return all_results
