"""Evaluator for comparing embedding models."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from .base import BaseEvaluator, EvaluationResult
from ..metrics.text_metrics import cosine_similarity_score


class EmbeddingEvaluator(BaseEvaluator):
    """Evaluate and compare different embedding models.

    Measures how well embeddings capture semantic similarity
    by checking if similar texts are closer than dissimilar texts.

    Dataset format:
        List of dicts, each with:
            - "anchor": str (reference text)
            - "positive": str (semantically similar text)
            - "negative": str (semantically different text)

    Component:
        An embedding model with .embed_query(text) -> List[float]
        or .embed_documents(texts) -> List[List[float]].

    Example:
        from Testframework.evaluators import EmbeddingEvaluator

        evaluator = EmbeddingEvaluator()
        result = evaluator.evaluate(ollama_embeddings, similarity_dataset)
        print(result.summary())
    """

    name = "EmbeddingEvaluator"

    def evaluate(
        self, component: Any, dataset: List[Dict], component_name: str = "embedding", **kwargs
    ) -> EvaluationResult:
        """Evaluate an embedding model on a triplet similarity task.

        Args:
            component: Embedding model with embed_query(text) or embed_documents(texts).
            dataset: List of {"anchor": str, "positive": str, "negative": str}.
            component_name: Name for this embedding model.

        Returns:
            EvaluationResult with triplet accuracy, avg similarity scores.
        """
        start = time.time()

        embed_fn = getattr(component, "embed_query", None)
        if embed_fn is None:
            # Fallback: use embed_documents for single texts
            base_embed = getattr(component, "embed_documents")
            embed_fn = lambda text: base_embed([text])[0]

        correct_triplets = 0
        total_triplets = 0
        pos_similarities = []
        neg_similarities = []
        details = []

        for item in dataset:
            anchor = item["anchor"]
            positive = item["positive"]
            negative = item["negative"]

            try:
                anchor_vec = embed_fn(anchor)
                positive_vec = embed_fn(positive)
                negative_vec = embed_fn(negative)

                pos_sim = cosine_similarity_score(anchor_vec, positive_vec)
                neg_sim = cosine_similarity_score(anchor_vec, negative_vec)

                is_correct = pos_sim > neg_sim
                if is_correct:
                    correct_triplets += 1
                total_triplets += 1

                pos_similarities.append(pos_sim)
                neg_similarities.append(neg_sim)

                details.append({
                    "anchor": anchor[:100],
                    "positive_similarity": pos_sim,
                    "negative_similarity": neg_sim,
                    "correct": is_correct,
                    "margin": pos_sim - neg_sim,
                })
            except Exception as e:
                details.append({"anchor": anchor[:100], "error": str(e)})

        n = total_triplets or 1
        metrics = {
            "triplet_accuracy": correct_triplets / n,
            "avg_positive_similarity": sum(pos_similarities) / n if pos_similarities else 0.0,
            "avg_negative_similarity": sum(neg_similarities) / n if neg_similarities else 0.0,
            "avg_margin": (
                (sum(pos_similarities) - sum(neg_similarities)) / n
                if pos_similarities
                else 0.0
            ),
        }

        return EvaluationResult(
            evaluator_name=self.name,
            component_name=component_name,
            config={},
            metrics=metrics,
            details=details,
            duration_seconds=time.time() - start,
        )
