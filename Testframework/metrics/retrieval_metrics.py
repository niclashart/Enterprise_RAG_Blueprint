"""Metrics for evaluating retrieval quality in RAG pipelines."""

from __future__ import annotations

import math
from typing import List, Optional


def precision_at_k(retrieved: List[str], relevant: List[str], k: Optional[int] = None) -> float:
    """
    Precision@K: fraction of retrieved documents (top-k) that are relevant.

    Args:
        retrieved: Ordered list of retrieved document identifiers.
        relevant: List of ground-truth relevant document identifiers.
        k: Cut-off rank. If None, uses len(retrieved).

    Returns:
        Precision score between 0.0 and 1.0.
    """
    if k is None:
        k = len(retrieved)
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc in top_k if doc in relevant_set)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: Optional[int] = None) -> float:
    """
    Recall@K: fraction of relevant documents that appear in the top-k retrieved.

    Args:
        retrieved: Ordered list of retrieved document identifiers.
        relevant: List of ground-truth relevant document identifiers.
        k: Cut-off rank. If None, uses len(retrieved).

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not relevant:
        return 1.0
    if k is None:
        k = len(retrieved)
    top_k = set(retrieved[:k])
    relevant_set = set(relevant)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(queries_results: List[tuple]) -> float:
    """
    Mean Reciprocal Rank (MRR) across multiple queries.

    Args:
        queries_results: List of (retrieved, relevant) tuples.
            Each retrieved is an ordered list, relevant is a set/list of ground-truth IDs.

    Returns:
        MRR score between 0.0 and 1.0.
    """
    if not queries_results:
        return 0.0

    total_rr = 0.0
    for retrieved, relevant in queries_results:
        relevant_set = set(relevant)
        rr = 0.0
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant_set:
                rr = 1.0 / rank
                break
        total_rr += rr
    return total_rr / len(queries_results)


def _dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """Discounted cumulative gain at position k."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: Optional[int] = None) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    Uses binary relevance (1 if in relevant set, 0 otherwise).

    Args:
        retrieved: Ordered list of retrieved document identifiers.
        relevant: List of ground-truth relevant document identifiers.
        k: Cut-off rank. If None, uses len(retrieved).

    Returns:
        NDCG score between 0.0 and 1.0.
    """
    if k is None:
        k = len(retrieved)
    if k == 0 or not relevant:
        return 0.0

    relevant_set = set(relevant)
    # Actual relevance scores for retrieved docs
    actual_scores = [1.0 if doc in relevant_set else 0.0 for doc in retrieved[:k]]
    # Ideal: all relevant docs at top
    ideal_scores = sorted(actual_scores, reverse=True)

    dcg = _dcg_at_k(actual_scores, k)
    idcg = _dcg_at_k(ideal_scores, k)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(retrieved: List[str], relevant: List[str], k: Optional[int] = None) -> float:
    """
    Hit Rate@K: 1.0 if at least one relevant document is in top-k, else 0.0.

    Args:
        retrieved: Ordered list of retrieved document identifiers.
        relevant: List of ground-truth relevant document identifiers.
        k: Cut-off rank. If None, uses len(retrieved).

    Returns:
        1.0 or 0.0.
    """
    if k is None:
        k = len(retrieved)
    top_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return 1.0 if top_k & relevant_set else 0.0
