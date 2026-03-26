"""Evaluator for LLM generation quality."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from .base import BaseEvaluator, EvaluationResult
from ..metrics.text_metrics import bleu_score, rouge_scores, f1_token_overlap, exact_match


class GenerationEvaluator(BaseEvaluator):
    """Evaluate LLM answer generation quality against ground-truth answers.

    Dataset format:
        List of dicts, each with:
            - "question": str
            - "context": str (retrieved context to provide)
            - "reference_answer": str (ground-truth answer)

    Component:
        A callable(question, context) -> str  (the generated answer)
        OR an LLM chain with .invoke(question) -> str.

    Example:
        from Testframework.evaluators import GenerationEvaluator

        evaluator = GenerationEvaluator()

        def my_generator(question, context):
            return my_chain.invoke({"question": question, "context": context})

        result = evaluator.evaluate(my_generator, qa_dataset)
        print(result.summary())
    """

    name = "GenerationEvaluator"

    def evaluate(
        self, component: Any, dataset: List[Dict], component_name: str = "generator", **kwargs
    ) -> EvaluationResult:
        """Evaluate a generation component.

        Args:
            component: Callable(question, context) -> str, or object with .invoke(str) -> str.
            dataset: List of {"question": str, "context": str, "reference_answer": str}.
            component_name: Name for this generation config.

        Returns:
            EvaluationResult with BLEU, ROUGE, F1, exact match metrics.
        """
        start = time.time()

        all_bleu = []
        all_rouge1 = []
        all_rouge2 = []
        all_rougeL = []
        all_f1 = []
        all_em = []
        details = []

        for item in dataset:
            question = item["question"]
            context = item.get("context", "")
            reference = item["reference_answer"]

            try:
                if callable(component):
                    generated = component(question, context)
                elif hasattr(component, "invoke"):
                    generated = component.invoke(question)
                else:
                    generated = str(component)

                if not isinstance(generated, str):
                    generated = str(generated)
            except Exception as e:
                details.append({"question": question, "error": str(e)})
                continue

            b = bleu_score(reference, generated)
            r = rouge_scores(reference, generated)
            f = f1_token_overlap(reference, generated)
            em = exact_match(reference, generated)

            all_bleu.append(b)
            all_rouge1.append(r["rouge1"])
            all_rouge2.append(r["rouge2"])
            all_rougeL.append(r["rougeL"])
            all_f1.append(f)
            all_em.append(em)

            details.append({
                "question": question,
                "reference": reference[:200],
                "generated": generated[:200],
                "bleu": b,
                "rouge1": r["rouge1"],
                "rouge2": r["rouge2"],
                "rougeL": r["rougeL"],
                "f1_overlap": f,
                "exact_match": em,
            })

        n = len(all_bleu) or 1
        metrics = {
            "bleu": sum(all_bleu) / n,
            "rouge1": sum(all_rouge1) / n,
            "rouge2": sum(all_rouge2) / n,
            "rougeL": sum(all_rougeL) / n,
            "f1_token_overlap": sum(all_f1) / n,
            "exact_match": sum(all_em) / n,
        }

        return EvaluationResult(
            evaluator_name=self.name,
            component_name=component_name,
            config={},
            metrics=metrics,
            details=details,
            duration_seconds=time.time() - start,
        )
