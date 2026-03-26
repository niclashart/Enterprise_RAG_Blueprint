"""Evaluator for comparing chunking strategies."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from .base import BaseEvaluator, EvaluationResult
from ..metrics.retrieval_metrics import precision_at_k, recall_at_k, ndcg_at_k


class ChunkingEvaluator(BaseEvaluator):
    """Evaluate how different chunking configurations affect retrieval quality.

    This evaluator takes documents, chunks them with different strategies,
    indexes them into a vector store, and measures retrieval quality.

    Dataset format:
        List of dicts, each with:
            - "query": str
            - "relevant_content": List[str]  (content snippets that should be retrieved)

    Component:
        A dict describing the chunking config:
            - "splitter": A text splitter instance (e.g. RecursiveCharacterTextSplitter)
            - "embed_fn": Callable that embeds and indexes chunks, returns a retriever
            - "documents": List of Document objects to chunk

    Example:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from Testframework.evaluators import ChunkingEvaluator

        evaluator = ChunkingEvaluator(k=5)

        configs = [
            {"name": "small_chunks", "component": {
                "splitter": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
                "embed_fn": my_embed_and_index_fn,
                "documents": docs,
            }},
            {"name": "large_chunks", "component": {
                "splitter": RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200),
                "embed_fn": my_embed_and_index_fn,
                "documents": docs,
            }},
        ]
        comparison = evaluator.compare(configs, dataset, rank_metric="recall@5")
    """

    name = "ChunkingEvaluator"

    def __init__(self, k: int = 5):
        self.k = k

    def evaluate(
        self, component: Dict[str, Any], dataset: List[Dict], component_name: str = "chunking", **kwargs
    ) -> EvaluationResult:
        """Evaluate a chunking strategy.

        Args:
            component: Dict with "splitter", "embed_fn", and "documents".
                - splitter: TextSplitter with .split_documents(docs) method
                - embed_fn: Callable(chunks) -> retriever (indexes chunks, returns retriever)
                - documents: List of Document objects to chunk
            dataset: List of {"query": str, "relevant_content": List[str]}.
            component_name: Name for this chunking config.

        Returns:
            EvaluationResult with chunk statistics and retrieval metrics.
        """
        start = time.time()
        k = kwargs.get("k", self.k)

        splitter = component["splitter"]
        embed_fn = component["embed_fn"]
        documents = component["documents"]

        # Chunk the documents
        chunks = splitter.split_documents(documents)
        chunk_sizes = [len(c.page_content) for c in chunks]

        # Index and get retriever
        retriever = embed_fn(chunks)

        # Evaluate retrieval quality
        all_precisions = []
        all_recalls = []
        details = []

        for item in dataset:
            query = item["query"]
            relevant_content = item.get("relevant_content", [])

            try:
                retrieved_docs = retriever.invoke(query)
                retrieved_content = [doc.page_content for doc in retrieved_docs]
            except Exception as e:
                details.append({"query": query, "error": str(e)})
                continue

            # Content-based matching: check if relevant content appears in retrieved chunks
            hits = 0
            for rel in relevant_content:
                rel_lower = rel.lower()
                for ret in retrieved_content:
                    if rel_lower in ret.lower():
                        hits += 1
                        break

            p = hits / min(k, len(retrieved_content)) if retrieved_content else 0.0
            r = hits / len(relevant_content) if relevant_content else 1.0

            all_precisions.append(p)
            all_recalls.append(r)

            details.append({
                "query": query,
                "content_hits": hits,
                "total_relevant": len(relevant_content),
                "precision": p,
                "recall": r,
            })

        n_queries = len(all_precisions) or 1
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0

        metrics = {
            "num_chunks": float(len(chunks)),
            "avg_chunk_size": avg_chunk_size,
            "min_chunk_size": float(min(chunk_sizes)) if chunk_sizes else 0.0,
            "max_chunk_size": float(max(chunk_sizes)) if chunk_sizes else 0.0,
            f"content_precision@{k}": sum(all_precisions) / n_queries,
            f"content_recall@{k}": sum(all_recalls) / n_queries,
        }

        return EvaluationResult(
            evaluator_name=self.name,
            component_name=component_name,
            config={
                "chunk_size": getattr(splitter, "_chunk_size", "unknown"),
                "chunk_overlap": getattr(splitter, "_chunk_overlap", "unknown"),
            },
            metrics=metrics,
            details=details,
            duration_seconds=time.time() - start,
        )
