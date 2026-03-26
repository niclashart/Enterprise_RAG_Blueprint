"""Runner for A/B comparisons between RAG components."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..evaluators.base import BaseEvaluator, ComparisonResult


class ComparisonRunner:
    """Run A/B comparisons between multiple RAG component configurations.

    Example:
        from Testframework.runners import ComparisonRunner
        from Testframework.evaluators import RetrievalEvaluator

        runner = ComparisonRunner()
        result = runner.compare(
            evaluator=RetrievalEvaluator(k=5),
            components=[
                {"name": "bm25_retriever", "component": bm25},
                {"name": "semantic_retriever", "component": semantic},
                {"name": "hybrid_retriever", "component": hybrid},
            ],
            dataset=retrieval_dataset,
            rank_metric="recall@5",
        )
        print(result.summary())
    """

    def compare(
        self,
        evaluator: BaseEvaluator,
        components: List[Dict[str, Any]],
        dataset: Any,
        rank_metric: str = None,
        higher_is_better: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> ComparisonResult:
        """Run an A/B comparison.

        Args:
            evaluator: Evaluator to use for all components.
            components: List of {"name": str, "component": Any}.
            dataset: Dataset to evaluate against.
            rank_metric: Metric to rank by. If None, uses evaluator default.
            higher_is_better: Whether higher values of rank_metric are better.
            verbose: Print progress.
            **kwargs: Extra kwargs for evaluator.evaluate().

        Returns:
            ComparisonResult with ranked results.
        """
        comparison = ComparisonResult(evaluator_name=evaluator.name)

        if verbose:
            print(f"Comparing {len(components)} configurations with {evaluator.name}")
            print("-" * 50)

        for comp_info in components:
            name = comp_info["name"]
            component = comp_info["component"]

            if verbose:
                print(f"  Evaluating: {name}...", end=" ", flush=True)

            try:
                result = evaluator.evaluate(component, dataset, component_name=name, **kwargs)
                comparison.add_result(result)
                if verbose:
                    metric_str = ", ".join(f"{k}={v:.4f}" for k, v in list(result.metrics.items())[:3])
                    print(f"OK [{metric_str}]")
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")

        if rank_metric:
            comparison.rank_by(rank_metric, higher_is_better)

        if verbose:
            print("-" * 50)
            if comparison.winner:
                print(f"Winner: {comparison.winner}")

        return comparison

    def multi_compare(
        self,
        evaluations: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> List[ComparisonResult]:
        """Run multiple comparisons across different evaluator types.

        Args:
            evaluations: List of dicts with keys:
                - "evaluator": BaseEvaluator
                - "components": List[{"name": str, "component": Any}]
                - "dataset": evaluation dataset
                - "rank_metric": str (optional)
            verbose: Print progress.

        Returns:
            List of ComparisonResult objects.
        """
        results = []
        for eval_config in evaluations:
            result = self.compare(
                evaluator=eval_config["evaluator"],
                components=eval_config["components"],
                dataset=eval_config["dataset"],
                rank_metric=eval_config.get("rank_metric"),
                verbose=verbose,
            )
            results.append(result)
        return results
