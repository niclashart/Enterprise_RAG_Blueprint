"""End-to-end evaluator for the full RAG pipeline."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from .base import BaseEvaluator, EvaluationResult
from ..metrics.text_metrics import bleu_score, rouge_scores, f1_token_overlap
from ..metrics.retrieval_metrics import recall_at_k, precision_at_k


class PipelineEvaluator(BaseEvaluator):
    """Evaluate a complete RAG pipeline end-to-end.

    Measures both retrieval and generation quality together,
    including latency. Use this to compare different pipeline configurations.

    Dataset format:
        List of dicts, each with:
            - "question": str
            - "reference_answer": str
        Optionally:
            - "relevant_docs": List[str] (for retrieval evaluation)

    Component:
        A callable(question: str) -> dict with:
            - "answer": str
            - "sources": List[str] (optional, for retrieval eval)
        OR a callable(question: str) -> str (answer only).

    Example:
        from Testframework.evaluators import PipelineEvaluator

        def my_pipeline(question):
            result, docs, _ = call_llm(question)
            return {"answer": result, "sources": [d.split("->")[0] for d in docs]}

        evaluator = PipelineEvaluator()
        result = evaluator.evaluate(my_pipeline, qa_dataset)
        print(result.summary())
    """

    name = "PipelineEvaluator"

    def __init__(self, retrieval_k: int = 5):
        self.retrieval_k = retrieval_k

    def evaluate(
        self, component: Any, dataset: List[Dict], component_name: str = "pipeline", **kwargs
    ) -> EvaluationResult:
        """Evaluate a full RAG pipeline.

        Args:
            component: Callable(question) -> {"answer": str, "sources": List[str]}
                or Callable(question) -> str.
            dataset: List of {"question", "reference_answer", optionally "relevant_docs"}.
            component_name: Name for this pipeline config.

        Returns:
            EvaluationResult with generation quality, retrieval quality, and latency metrics.
        """
        eval_start = time.time()
        k = kwargs.get("k", self.retrieval_k)

        all_bleu = []
        all_rouge1 = []
        all_f1 = []
        all_recalls = []
        all_precisions = []
        latencies = []
        details = []

        for item in dataset:
            question = item["question"]
            reference = item["reference_answer"]
            relevant_docs = item.get("relevant_docs", [])

            try:
                call_start = time.time()
                output = component(question)
                latency = time.time() - call_start
                latencies.append(latency)

                if isinstance(output, dict):
                    answer = output.get("answer", "")
                    sources = output.get("sources", [])
                elif isinstance(output, tuple) and len(output) >= 2:
                    answer, sources = output[0], output[1] or []
                else:
                    answer = str(output)
                    sources = []
            except Exception as e:
                details.append({"question": question, "error": str(e)})
                continue

            # Generation metrics
            b = bleu_score(reference, answer)
            r = rouge_scores(reference, answer)
            f = f1_token_overlap(reference, answer)
            all_bleu.append(b)
            all_rouge1.append(r["rouge1"])
            all_f1.append(f)

            detail = {
                "question": question,
                "reference": reference[:200],
                "answer": answer[:200],
                "latency": latency,
                "bleu": b,
                "rouge1": r["rouge1"],
                "f1_overlap": f,
            }

            # Retrieval metrics (if ground-truth docs available)
            if relevant_docs and sources:
                rec = recall_at_k(sources, relevant_docs, k)
                prec = precision_at_k(sources, relevant_docs, k)
                all_recalls.append(rec)
                all_precisions.append(prec)
                detail["retrieval_recall"] = rec
                detail["retrieval_precision"] = prec

            details.append(detail)

        n_gen = len(all_bleu) or 1
        n_ret = len(all_recalls) or 1

        metrics = {
            "bleu": sum(all_bleu) / n_gen,
            "rouge1": sum(all_rouge1) / n_gen,
            "f1_token_overlap": sum(all_f1) / n_gen,
            "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
            "p95_latency_seconds": (
                sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0
            ),
            "total_queries": float(len(dataset)),
            "successful_queries": float(len(all_bleu)),
        }

        if all_recalls:
            metrics[f"retrieval_recall@{k}"] = sum(all_recalls) / n_ret
            metrics[f"retrieval_precision@{k}"] = sum(all_precisions) / n_ret

        return EvaluationResult(
            evaluator_name=self.name,
            component_name=component_name,
            config={"retrieval_k": k},
            metrics=metrics,
            details=details,
            duration_seconds=time.time() - eval_start,
        )
