"""Evaluator for guardrails / input validation components."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import BaseEvaluator, EvaluationResult
from ..metrics.classification_metrics import accuracy, precision_recall_f1, confusion_matrix


class GuardrailsEvaluator(BaseEvaluator):
    """Evaluate guardrails accuracy: does it correctly block harmful inputs and allow safe ones?

    Dataset format:
        List of dicts, each with:
            - "input": str (user input text)
            - "expected": str ("allowed" or "blocked")

    Component:
        A callable(input) -> Tuple[bool, Optional[str]]
        where True = allowed, False = blocked.
        Compatible with chain.guardrails_manager.ensure_input_allowed.

    Example:
        from Testframework.evaluators import GuardrailsEvaluator
        from chain.guardrails_manager import ensure_input_allowed

        evaluator = GuardrailsEvaluator()
        result = evaluator.evaluate(ensure_input_allowed, guardrails_dataset)
        print(result.summary())
    """

    name = "GuardrailsEvaluator"

    def evaluate(
        self, component: Any, dataset: List[Dict], component_name: str = "guardrails", **kwargs
    ) -> EvaluationResult:
        """Evaluate guardrails classification accuracy.

        Args:
            component: Callable(input_text) -> (allowed: bool, reason: Optional[str]).
            dataset: List of {"input": str, "expected": "allowed"|"blocked"}.
            component_name: Name for this guardrails config.

        Returns:
            EvaluationResult with accuracy, precision, recall, F1, confusion matrix.
        """
        start = time.time()

        predictions = []
        ground_truth = []
        details = []

        for item in dataset:
            input_text = item["input"]
            expected = item["expected"]  # "allowed" or "blocked"

            try:
                allowed, reason = component(input_text)
                predicted = "allowed" if allowed else "blocked"
            except Exception as e:
                details.append({"input": input_text[:100], "error": str(e)})
                continue

            predictions.append(predicted)
            ground_truth.append(expected)
            is_correct = predicted == expected

            details.append({
                "input": input_text[:100],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "reason": reason,
            })

        acc = accuracy(predictions, ground_truth) if predictions else 0.0
        # Precision/recall for "blocked" class (detecting harmful input)
        prf_blocked = precision_recall_f1(predictions, ground_truth, positive_label="blocked")
        # Precision/recall for "allowed" class (not blocking safe input)
        prf_allowed = precision_recall_f1(predictions, ground_truth, positive_label="allowed")
        cm = confusion_matrix(predictions, ground_truth, labels=["allowed", "blocked"])

        metrics = {
            "accuracy": acc,
            "blocked_precision": prf_blocked["precision"],
            "blocked_recall": prf_blocked["recall"],
            "blocked_f1": prf_blocked["f1"],
            "allowed_precision": prf_allowed["precision"],
            "allowed_recall": prf_allowed["recall"],
            "allowed_f1": prf_allowed["f1"],
        }

        return EvaluationResult(
            evaluator_name=self.name,
            component_name=component_name,
            config={},
            metrics=metrics,
            details=details,
            duration_seconds=time.time() - start,
            metadata={"confusion_matrix": cm},
        )
