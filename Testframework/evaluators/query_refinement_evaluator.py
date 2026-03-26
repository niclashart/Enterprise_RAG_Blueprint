"""Evaluator for query refinement components."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from .base import BaseEvaluator, EvaluationResult
from ..metrics.text_metrics import f1_token_overlap
from ..metrics.retrieval_metrics import recall_at_k, precision_at_k


class QueryRefinementEvaluator(BaseEvaluator):
    """Evaluate query refinement quality.

    Measures whether refined queries lead to better retrieval results
    compared to the original queries.

    Dataset format:
        List of dicts, each with:
            - "original_query": str
            - "relevant_docs": List[str]  (ground-truth relevant document IDs)
        Optionally:
            - "expected_refined": str (ideal refined query for text comparison)

    Component:
        A callable(query: str) -> str that returns the refined query.
        Compatible with chain.query_refinement.refine_query.

    Example:
        from Testframework.evaluators import QueryRefinementEvaluator
        from chain.query_refinement import refine_query

        evaluator = QueryRefinementEvaluator()
        result = evaluator.evaluate(
            refine_query, dataset, retriever=my_retriever
        )
        print(result.summary())
    """

    name = "QueryRefinementEvaluator"

    def __init__(self, k: int = 5, doc_id_fn: Optional[Callable] = None):
        self.k = k
        self.doc_id_fn = doc_id_fn or (lambda doc: doc.metadata.get("source", str(doc)))

    def evaluate(
        self, component: Any, dataset: List[Dict], component_name: str = "query_refinement", **kwargs
    ) -> EvaluationResult:
        """Evaluate query refinement by comparing retrieval with original vs refined queries.

        Args:
            component: Callable(query: str) -> str.
            dataset: List of {"original_query": str, "relevant_docs": List[str]}.
            component_name: Name for this refinement config.
            **kwargs:
                retriever: Required. A retriever with .invoke(query) to test retrieval improvement.

        Returns:
            EvaluationResult with retrieval improvement metrics.
        """
        retriever = kwargs.get("retriever")
        start = time.time()
        k = kwargs.get("k", self.k)

        original_recalls = []
        refined_recalls = []
        original_precisions = []
        refined_precisions = []
        details = []

        for item in dataset:
            original_query = item["original_query"]
            relevant = item["relevant_docs"]
            expected_refined = item.get("expected_refined")

            try:
                refined_query = component(original_query)
            except Exception as e:
                details.append({"original_query": original_query, "error": str(e)})
                continue

            detail = {
                "original_query": original_query,
                "refined_query": refined_query,
                "changed": original_query != refined_query,
            }

            # Text similarity to expected refined query
            if expected_refined:
                detail["similarity_to_expected"] = f1_token_overlap(expected_refined, refined_query)

            # Retrieval comparison (if retriever provided)
            if retriever is not None:
                try:
                    orig_docs = retriever.invoke(original_query)
                    orig_ids = [self.doc_id_fn(doc) for doc in orig_docs]
                    orig_recall = recall_at_k(orig_ids, relevant, k)
                    orig_precision = precision_at_k(orig_ids, relevant, k)

                    ref_docs = retriever.invoke(refined_query)
                    ref_ids = [self.doc_id_fn(doc) for doc in ref_docs]
                    ref_recall = recall_at_k(ref_ids, relevant, k)
                    ref_precision = precision_at_k(ref_ids, relevant, k)

                    original_recalls.append(orig_recall)
                    refined_recalls.append(ref_recall)
                    original_precisions.append(orig_precision)
                    refined_precisions.append(ref_precision)

                    detail["original_recall"] = orig_recall
                    detail["refined_recall"] = ref_recall
                    detail["recall_delta"] = ref_recall - orig_recall
                    detail["original_precision"] = orig_precision
                    detail["refined_precision"] = ref_precision
                except Exception as e:
                    detail["retrieval_error"] = str(e)

            details.append(detail)

        n = len(original_recalls) or 1
        changed_count = sum(1 for d in details if d.get("changed", False))

        metrics = {
            "queries_changed_ratio": changed_count / len(dataset) if dataset else 0.0,
        }

        if original_recalls:
            avg_orig_recall = sum(original_recalls) / n
            avg_ref_recall = sum(refined_recalls) / n
            avg_orig_prec = sum(original_precisions) / n
            avg_ref_prec = sum(refined_precisions) / n
            metrics.update({
                f"original_recall@{k}": avg_orig_recall,
                f"refined_recall@{k}": avg_ref_recall,
                f"recall_improvement@{k}": avg_ref_recall - avg_orig_recall,
                f"original_precision@{k}": avg_orig_prec,
                f"refined_precision@{k}": avg_ref_prec,
                f"precision_improvement@{k}": avg_ref_prec - avg_orig_prec,
            })

        return EvaluationResult(
            evaluator_name=self.name,
            component_name=component_name,
            config={"k": k, "has_retriever": retriever is not None},
            metrics=metrics,
            details=details,
            duration_seconds=time.time() - start,
        )
