"""Generate evaluation reports in various formats."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..evaluators.base import ComparisonResult, EvaluationResult


class ReportGenerator:
    """Generate evaluation reports from benchmark and comparison results.

    Example:
        from Testframework.reporting import ReportGenerator

        reporter = ReportGenerator(output_dir="./eval_reports")
        reporter.generate_report(results, title="RAG Benchmark Q1")
        reporter.generate_comparison_report(comparison, title="Retriever A vs B")
    """

    def __init__(self, output_dir: str = "./eval_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        results: List[EvaluationResult],
        title: str = "Evaluation Report",
        filename: str = None,
    ) -> str:
        """Generate a markdown report from evaluation results.

        Args:
            results: List of EvaluationResult objects.
            title: Report title.
            filename: Output filename. Auto-generated if None.

        Returns:
            Path to the generated report file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.md"

        lines = [
            f"# {title}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total evaluations: {len(results)}",
            "",
        ]

        for result in results:
            lines.append(f"## {result.evaluator_name}: {result.component_name}")
            lines.append("")

            if result.config:
                lines.append("**Configuration:**")
                for k, v in result.config.items():
                    lines.append(f"- {k}: {v}")
                lines.append("")

            lines.append(f"**Duration:** {result.duration_seconds:.2f}s")
            lines.append("")

            lines.append("**Metrics:**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for metric_name, value in result.metrics.items():
                lines.append(f"| {metric_name} | {value:.4f} |")
            lines.append("")

            if result.details:
                lines.append(f"<details><summary>Details ({len(result.details)} items)</summary>")
                lines.append("")
                lines.append("```json")
                # Limit detail output for readability
                preview = result.details[:10]
                lines.append(json.dumps(preview, indent=2, default=str))
                if len(result.details) > 10:
                    lines.append(f"... and {len(result.details) - 10} more items")
                lines.append("```")
                lines.append("</details>")
                lines.append("")

        report_path = self.output_dir / filename
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return str(report_path)

    def generate_comparison_report(
        self,
        comparison: ComparisonResult,
        title: str = "Comparison Report",
        filename: str = None,
    ) -> str:
        """Generate a markdown comparison report.

        Args:
            comparison: ComparisonResult object.
            title: Report title.
            filename: Output filename. Auto-generated if None.

        Returns:
            Path to the generated report file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_{timestamp}.md"

        lines = [
            f"# {title}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Evaluator: {comparison.evaluator_name}",
            f"Components compared: {len(comparison.results)}",
            "",
        ]

        if comparison.winner:
            lines.append(f"**Winner: {comparison.winner}**")
            if comparison.ranking_metric:
                lines.append(f"Ranked by: {comparison.ranking_metric}")
            lines.append("")

        # Summary table
        if comparison.results:
            all_metrics = set()
            for r in comparison.results:
                all_metrics.update(r.metrics.keys())
            all_metrics = sorted(all_metrics)

            lines.append("## Results Overview")
            lines.append("")
            header = "| Component | " + " | ".join(all_metrics) + " | Duration |"
            separator = "|-----------|" + "|".join(["-------"] * len(all_metrics)) + "|----------|"
            lines.append(header)
            lines.append(separator)

            for result in comparison.results:
                values = " | ".join(f"{result.metrics.get(m, 0.0):.4f}" for m in all_metrics)
                prefix = "**" if result.component_name == comparison.winner else ""
                suffix = "**" if result.component_name == comparison.winner else ""
                lines.append(
                    f"| {prefix}{result.component_name}{suffix} | {values} | {result.duration_seconds:.2f}s |"
                )
            lines.append("")

        report_path = self.output_dir / filename
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return str(report_path)

    def export_json(
        self, results: List[EvaluationResult], filename: str = None
    ) -> str:
        """Export results as JSON for programmatic consumption.

        Args:
            results: List of EvaluationResult objects.
            filename: Output filename. Auto-generated if None.

        Returns:
            Path to the generated JSON file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"

        data = []
        for result in results:
            data.append({
                "evaluator": result.evaluator_name,
                "component": result.component_name,
                "config": result.config,
                "metrics": result.metrics,
                "duration_seconds": result.duration_seconds,
                "num_details": len(result.details),
                "metadata": result.metadata,
            })

        json_path = self.output_dir / filename
        json_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return str(json_path)
