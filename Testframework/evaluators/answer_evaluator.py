"""Evaluator for answer validation components (keyword check, LLM check)."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from .base import BaseEvaluator, EvaluationResult
from ..metrics.classification_metrics import accuracy, precision_recall_f1


class AnswerEvaluator(BaseEvaluator):
    """Evaluate answer validation components.

    Tests whether the answer checker correctly identifies good and bad answers.

    Dataset format:
        List of dicts, each with:
            - "question": str
            - "answer": str
            - "expected_evaluation": str ("yes", "no", "partial")
        Optionally:
            - "expected_keywords": List[str] (for keyword-based evaluation)
            - "relevant_documents": List[str] (for LLM-based evaluation)

    Component:
        A callable with one of these signatures:
            - (question, answer) -> dict with "evaluation" key
            - (question, answer, **kwargs) -> dict with "evaluation" key
        Compatible with chain.check_answers_llm.check_answer_with_keywords.

    Example:
        from Testframework.evaluators import AnswerEvaluator
        from chain.check_answers_llm import check_answer_with_keywords

        evaluator = AnswerEvaluator()
        result = evaluator.evaluate(check_answer_with_keywords, dataset)
        print(result.summary())
    """

    name = "AnswerEvaluator"

    def evaluate(
        self, component: Any, dataset: List[Dict], component_name: str = "answer_checker", **kwargs
    ) -> EvaluationResult:
        """Evaluate an answer validation component.

        Args:
            component: Callable(question, answer, **extra) -> dict with "evaluation" key.
            dataset: List of {"question", "answer", "expected_evaluation", ...}.
            component_name: Name for this validation config.

        Returns:
            EvaluationResult with accuracy and per-class precision/recall/F1.
        """
        start = time.time()

        predictions = []
        ground_truth = []
        details = []

        for item in dataset:
            question = item["question"]
            answer = item["answer"]
            expected = item["expected_evaluation"]

            # Build extra kwargs from dataset item
            extra = {}
            if "expected_keywords" in item:
                extra["expected_keywords"] = item["expected_keywords"]
            if "threshold" in item:
                extra["threshold"] = item["threshold"]
            if "relevant_documents" in item:
                extra["relevant_documents"] = item["relevant_documents"]

            try:
                result = component(question, answer, **extra)
                if isinstance(result, dict):
                    predicted = result.get("evaluation", "unknown")
                elif isinstance(result, str):
                    predicted = result
                else:
                    predicted = str(result)
            except Exception as e:
                details.append({"question": question[:100], "error": str(e)})
                continue

            predictions.append(predicted)
            ground_truth.append(expected)

            details.append({
                "question": question[:100],
                "answer": answer[:100],
                "expected": expected,
                "predicted": predicted,
                "correct": predicted == expected,
                "full_result": result if isinstance(result, dict) else None,
            })

        acc = accuracy(predictions, ground_truth) if predictions else 0.0

        metrics = {"accuracy": acc}

        # Per-class metrics
        for label in set(ground_truth):
            prf = precision_recall_f1(predictions, ground_truth, positive_label=label)
            metrics[f"{label}_precision"] = prf["precision"]
            metrics[f"{label}_recall"] = prf["recall"]
            metrics[f"{label}_f1"] = prf["f1"]

        return EvaluationResult(
            evaluator_name=self.name,
            component_name=component_name,
            config={},
            metrics=metrics,
            details=details,
            duration_seconds=time.time() - start,
        )
