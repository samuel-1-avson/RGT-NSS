"""
RAG Evaluation using Ragas

This module provides evaluation capabilities for RAG pipelines
using metrics like faithfulness, answer relevancy, and context precision.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy
)

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluates RAG pipeline performance using Ragas metrics.
    
    Supported Metrics:
    - Faithfulness: How factually accurate is the answer based on context
    - Answer Relevancy: How relevant is the answer to the question
    - Context Precision: Precision of retrieved contexts
    - Context Recall: Recall of relevant contexts
    - Context Relevancy: Relevancy of contexts to the question
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.evaluation_results = []
        logger.info("RAGEvaluator initialized")
    
    def prepare_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dataset:
        """
        Prepare evaluation dataset in Ragas format.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved contexts (list of lists)
            ground_truths: Optional ground truth answers
            
        Returns:
            HuggingFace Dataset for evaluation
            
        Raises:
            ValueError: If input lists have different lengths
        """
        if not (len(questions) == len(answers) == len(contexts)):
            raise ValueError("Input lists must have the same length")
        
        data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts
        }
        
        if ground_truths:
            if len(ground_truths) != len(questions):
                raise ValueError("ground_truths must have same length as questions")
            data['ground_truth'] = ground_truths
        
        dataset = Dataset.from_dict(data)
        logger.info(f"Prepared dataset with {len(questions)} samples")
        
        return dataset
    
    def evaluate(
        self,
        dataset: Dataset,
        metrics: Optional[List] = None,
        raise_exceptions: bool = False
    ) -> Dict[str, Any]:
        """
        Run evaluation on the dataset.
        
        Args:
            dataset: Prepared evaluation dataset
            metrics: List of metrics to compute (default: all)
            raise_exceptions: Whether to raise exceptions during evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        if metrics is None:
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        
        logger.info(f"Running evaluation with {len(metrics)} metrics")
        
        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                raise_exceptions=raise_exceptions
            )
            
            # Convert to dictionary
            results_dict = {
                metric: round(score, 4) 
                for metric, score in result.items()
            }
            
            logger.info("Evaluation completed successfully")
            logger.info(f"Results: {results_dict}")
            
            return results_dict
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single Q&A pair.
        
        Args:
            question: The question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary with metric scores
        """
        dataset = self.prepare_dataset(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth] if ground_truth else None
        )
        
        return self.evaluate(dataset)
    
    def set_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Set quality thresholds for metrics.
        
        Args:
            thresholds: Dictionary mapping metric names to threshold values
            
        Example:
            {
                'faithfulness': 0.7,
                'answer_relevancy': 0.8,
                'context_precision': 0.6
            }
        """
        self.thresholds = thresholds
        logger.info(f"Set thresholds: {thresholds}")
    
    def check_thresholds(self, results: Dict[str, float]) -> Dict[str, Any]:
        """
        Check if results meet quality thresholds.
        
        Args:
            results: Evaluation results
            
        Returns:
            Dictionary with threshold check results
        """
        if not hasattr(self, 'thresholds'):
            logger.warning("No thresholds set")
            return {"status": "no_thresholds"}
        
        checks = {}
        all_passed = True
        
        for metric, score in results.items():
            if metric in self.thresholds:
                threshold = self.thresholds[metric]
                passed = score >= threshold
                checks[metric] = {
                    "score": score,
                    "threshold": threshold,
                    "passed": passed
                }
                if not passed:
                    all_passed = False
        
        checks["all_passed"] = all_passed
        
        return checks
    
    def generate_report(
        self,
        results: Dict[str, float],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate an evaluation report.
        
        Args:
            results: Evaluation results
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report_lines = [
            "=" * 60,
            "RAG EVALUATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            "",
            "METRIC SCORES:",
            "-" * 60
        ]
        
        for metric, score in results.items():
            status = "✓" if score >= 0.7 else "⚠"
            report_lines.append(f"{status} {metric:25s}: {score:.4f}")
        
        # Threshold check if available
        if hasattr(self, 'thresholds'):
            report_lines.extend([
                "",
                "THRESHOLD CHECKS:",
                "-" * 60
            ])
            
            checks = self.check_thresholds(results)
            for metric, check in checks.items():
                if isinstance(check, dict):
                    status = "✓ PASS" if check['passed'] else "✗ FAIL"
                    report_lines.append(
                        f"{status} - {metric}: {check['score']:.4f} "
                        f"(threshold: {check['threshold']})"
                    )
        
        report_lines.extend([
            "",
            "=" * 60,
            "INTERPRETATION:",
            "=" * 60,
            "",
            "Faithfulness: Measures factual accuracy of the answer",
            "  based on the provided context.",
            "  Target: > 0.7 (Good), > 0.8 (Excellent)",
            "",
            "Answer Relevancy: Measures how relevant the answer is",
            "  to the question asked.",
            "  Target: > 0.7 (Good), > 0.8 (Excellent)",
            "",
            "Context Precision: Measures precision of retrieved contexts.",
            "  Target: > 0.6 (Good), > 0.7 (Excellent)",
            "",
            "Context Recall: Measures recall of relevant contexts.",
            "  Target: > 0.6 (Good), > 0.7 (Excellent)",
            "",
            "=" * 60
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report


def create_evaluation_dataset_from_qa_chain(
    qa_chain,
    test_questions: List[str],
    ground_truths: Optional[List[str]] = None
) -> Dict[str, List]:
    """
    Create evaluation dataset by running questions through a QA chain.
    
    Args:
        qa_chain: QA chain to test
        test_questions: List of test questions
        ground_truths: Optional ground truth answers
        
    Returns:
        Dictionary with questions, answers, and contexts
    """
    questions = []
    answers = []
    contexts = []
    
    for i, question in enumerate(test_questions):
        logger.info(f"Processing question {i+1}/{len(test_questions)}")
        
        # Get answer from chain
        result = qa_chain({"query": question})
        
        questions.append(question)
        answers.append(result["result"])
        
        # Extract context texts
        context_texts = [doc.page_content for doc in result.get("source_documents", [])]
        contexts.append(context_texts)
    
    data = {
        "questions": questions,
        "answers": answers,
        "contexts": contexts
    }
    
    if ground_truths:
        data["ground_truths"] = ground_truths
    
    return data


def main():
    """Example usage."""
    evaluator = RAGEvaluator()
    
    # Example evaluation data
    questions = [
        "What is machine learning?",
        "How does neural network training work?"
    ]
    
    answers = [
        "Machine learning is a subset of AI that enables computers to learn from data.",
        "Neural networks are trained using backpropagation and gradient descent."
    ]
    
    contexts = [
        ["Machine learning is a method of data analysis...", "AI encompasses machine learning..."],
        ["Neural networks use backpropagation...", "Gradient descent optimizes weights..."]
    ]
    
    ground_truths = [
        "Machine learning is a field of AI focused on algorithms that learn from data.",
        "Training involves forward propagation, loss calculation, and backpropagation."
    ]
    
    # Prepare and evaluate
    dataset = evaluator.prepare_dataset(questions, answers, contexts, ground_truths)
    results = evaluator.evaluate(dataset)
    
    # Generate report
    report = evaluator.generate_report(results)
    print(report)


if __name__ == "__main__":
    main()
