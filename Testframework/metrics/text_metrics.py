"""Metrics for evaluating text generation quality in RAG pipelines."""

from __future__ import annotations

import re
import math
from collections import Counter
from typing import Dict, List, Optional


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Extract n-grams from a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score(
    reference: str,
    hypothesis: str,
    max_n: int = 4,
    weights: Optional[List[float]] = None,
) -> float:
    """
    Compute BLEU score between a reference and hypothesis text.

    Simplified implementation using uniform weights by default.

    Args:
        reference: Ground-truth text.
        hypothesis: Generated text.
        max_n: Maximum n-gram order.
        weights: Weights for each n-gram order. Defaults to uniform.

    Returns:
        BLEU score between 0.0 and 1.0.
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n

    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not hyp_tokens:
        return 0.0

    # Brevity penalty
    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))

    log_avg = 0.0
    for n, weight in zip(range(1, max_n + 1), weights):
        ref_ngrams = Counter(_ngrams(ref_tokens, n))
        hyp_ngrams = Counter(_ngrams(hyp_tokens, n))

        # Clipped counts
        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())

        if total == 0 or clipped == 0:
            return 0.0

        precision = clipped / total
        log_avg += weight * math.log(precision)

    return bp * math.exp(log_avg)


def rouge_scores(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        reference: Ground-truth text.
        hypothesis: Generated text.

    Returns:
        Dictionary with rouge1, rouge2, rougeL F1 scores.
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    def _rouge_n(n: int) -> float:
        ref_ng = Counter(_ngrams(ref_tokens, n))
        hyp_ng = Counter(_ngrams(hyp_tokens, n))
        overlap = sum(min(ref_ng[ng], hyp_ng[ng]) for ng in hyp_ng)
        prec = overlap / max(sum(hyp_ng.values()), 1)
        rec = overlap / max(sum(ref_ng.values()), 1)
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    def _lcs_length(a: List[str], b: List[str]) -> int:
        m, n = len(a), len(b)
        # Space-optimized LCS
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        return prev[n]

    lcs_len = _lcs_length(ref_tokens, hyp_tokens)
    rouge_l_prec = lcs_len / max(len(hyp_tokens), 1)
    rouge_l_rec = lcs_len / max(len(ref_tokens), 1)
    rouge_l = (
        2 * rouge_l_prec * rouge_l_rec / (rouge_l_prec + rouge_l_rec)
        if (rouge_l_prec + rouge_l_rec) > 0
        else 0.0
    )

    return {
        "rouge1": _rouge_n(1),
        "rouge2": _rouge_n(2),
        "rougeL": rouge_l,
    }


def cosine_similarity_score(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity between -1.0 and 1.0.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimensions must match: {len(vec_a)} vs {len(vec_b)}")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def exact_match(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Exact match score.

    Args:
        reference: Ground-truth text.
        hypothesis: Generated text.
        normalize: If True, strips whitespace and lowercases before comparing.

    Returns:
        1.0 if texts match, 0.0 otherwise.
    """
    if normalize:
        return 1.0 if reference.strip().lower() == hypothesis.strip().lower() else 0.0
    return 1.0 if reference == hypothesis else 0.0


def f1_token_overlap(reference: str, hypothesis: str) -> float:
    """
    Token-level F1 overlap between reference and hypothesis.

    Useful for evaluating extractive answers.

    Args:
        reference: Ground-truth text.
        hypothesis: Generated text.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    ref_tokens = set(_tokenize(reference))
    hyp_tokens = set(_tokenize(hypothesis))

    if not ref_tokens or not hyp_tokens:
        return 0.0

    common = ref_tokens & hyp_tokens
    if not common:
        return 0.0

    precision = len(common) / len(hyp_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
