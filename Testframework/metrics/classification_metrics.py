"""Metrics for evaluating classification tasks (e.g. guardrails, answer validation)."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple


def accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """
    Classification accuracy.

    Args:
        predictions: Predicted labels.
        ground_truth: True labels.

    Returns:
        Accuracy between 0.0 and 1.0.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have equal length")
    if not predictions:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions)


def precision_recall_f1(
    predictions: List[str], ground_truth: List[str], positive_label: str = "yes"
) -> Dict[str, float]:
    """
    Precision, Recall, and F1 for a binary classification task.

    Args:
        predictions: Predicted labels.
        ground_truth: True labels.
        positive_label: The label considered as the positive class.

    Returns:
        Dictionary with precision, recall, and f1 scores.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have equal length")

    tp = sum(1 for p, g in zip(predictions, ground_truth) if p == positive_label and g == positive_label)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p == positive_label and g != positive_label)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if p != positive_label and g == positive_label)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def confusion_matrix(
    predictions: List[str], ground_truth: List[str], labels: List[str] = None
) -> Dict[str, Dict[str, int]]:
    """
    Build a confusion matrix as a nested dictionary.

    Args:
        predictions: Predicted labels.
        ground_truth: True labels.
        labels: Ordered list of labels. If None, inferred from data.

    Returns:
        Nested dict: matrix[actual][predicted] = count.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have equal length")

    if labels is None:
        labels = sorted(set(predictions) | set(ground_truth))

    matrix = {actual: {pred: 0 for pred in labels} for actual in labels}
    for p, g in zip(predictions, ground_truth):
        if g in matrix and p in matrix[g]:
            matrix[g][p] += 1

    return matrix
