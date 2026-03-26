"""RAG Component Test Framework.

An independent evaluation framework for testing and comparing
RAG pipeline components. Import any evaluator and use it in your code:

    from Testframework.evaluators import RetrievalEvaluator, GenerationEvaluator
    from Testframework.metrics import precision_at_k, rouge_scores
    from Testframework.datasets import create_retrieval_dataset
    from Testframework.runners import BenchmarkRunner, ComparisonRunner
    from Testframework.reporting import ReportGenerator
"""

from .evaluators import (
    BaseEvaluator,
    EvaluationResult,
    ComparisonResult,
    RetrievalEvaluator,
    ChunkingEvaluator,
    EmbeddingEvaluator,
    GenerationEvaluator,
    GuardrailsEvaluator,
    QueryRefinementEvaluator,
    AnswerEvaluator,
    PipelineEvaluator,
)
from .runners import BenchmarkRunner, ComparisonRunner
from .reporting import ReportGenerator

__version__ = "1.0.0"

__all__ = [
    # Evaluators
    "BaseEvaluator",
    "EvaluationResult",
    "ComparisonResult",
    "RetrievalEvaluator",
    "ChunkingEvaluator",
    "EmbeddingEvaluator",
    "GenerationEvaluator",
    "GuardrailsEvaluator",
    "QueryRefinementEvaluator",
    "AnswerEvaluator",
    "PipelineEvaluator",
    # Runners
    "BenchmarkRunner",
    "ComparisonRunner",
    # Reporting
    "ReportGenerator",
]
