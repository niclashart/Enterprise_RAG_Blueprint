"""Evaluator for retrieval components (vector store retrievers)."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from .base import BaseEvaluator, EvaluationResult
from ..metrics.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    hit_rate_at_k,
)


class RetrievalEvaluator(BaseEvaluator):
    """Evaluate retrieval quality against ground-truth relevant documents.

    Dataset format:
        List of dicts, each with:
            - "query": str
            - "relevant_docs": List[str]  (document identifiers)

    Component:
        A callable or object with .invoke(query) -> List[Document].
        Each Document must have .metadata["source"] or a custom id_fn.

    Example:
        from Testframework.evaluators import RetrievalEvaluator

        evaluator = RetrievalEvaluator(k=5)
        result = evaluator.evaluate(retriever, dataset)
        print(result.summary())
    """

    name = "RetrievalEvaluator"

    def __init__(self, k: int = 5, doc_id_fn: Optional[Callable] = None):
        """
        Args:
            k: Number of top documents to evaluate.
            doc_id_fn: Function to extract ID from a document. Defaults to doc.metadata["source"].
        """
        self.k = k
        self.doc_id_fn = doc_id_fn or (lambda doc: doc.metadata.get("source", str(doc)))

    def evaluate(
        self, component: Any, dataset: List[Dict], component_name: str = "retriever", **kwargs
    ) -> EvaluationResult:
        """Evaluate a retriever against a dataset of queries with known relevant documents.

        Args:
            component: Retriever with .invoke(query) method.
            dataset: List of {"query": str, "relevant_docs": List[str]}.
            component_name: Name for this component configuration.

        Returns:
            EvaluationResult with precision@k, recall@k, MRR, NDCG@k, hit_rate@k.
        """
        start = time.time()
        k = kwargs.get("k", self.k)

        all_precisions = []
        all_recalls = []
        all_ndcgs = []
        all_hits = []
        mrr_inputs = []
        details = []

        for item in dataset:
            query = item["query"]
            relevant = item["relevant_docs"]

            try:
                docs = component.invoke(query)
                retrieved_ids = [self.doc_id_fn(doc) for doc in docs]
            except Exception as e:
                details.append({"query": query, "error": str(e)})
                continue

            p = precision_at_k(retrieved_ids, relevant, k)
            r = recall_at_k(retrieved_ids, relevant, k)
            n = ndcg_at_k(retrieved_ids, relevant, k)
            h = hit_rate_at_k(retrieved_ids, relevant, k)

            all_precisions.append(p)
            all_recalls.append(r)
            all_ndcgs.append(n)
            all_hits.append(h)
            mrr_inputs.append((retrieved_ids, relevant))

            details.append({
                "query": query,
                "retrieved": retrieved_ids[:k],
                "relevant": relevant,
                "precision": p,
                "recall": r,
                "ndcg": n,
                "hit": h,
            })

        n_queries = len(all_precisions) or 1
        metrics = {
            f"precision@{k}": sum(all_precisions) / n_queries,
            f"recall@{k}": sum(all_recalls) / n_queries,
            "mrr": mean_reciprocal_rank(mrr_inputs),
            f"ndcg@{k}": sum(all_ndcgs) / n_queries,
            f"hit_rate@{k}": sum(all_hits) / n_queries,
        }

        return EvaluationResult(
            evaluator_name=self.name,
            component_name=component_name,
            config={"k": k},
            metrics=metrics,
            details=details,
            duration_seconds=time.time() - start,
        )
