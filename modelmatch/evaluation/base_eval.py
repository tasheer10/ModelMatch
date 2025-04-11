import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

# Get a logger for this module
logger = logging.getLogger(__name__)

class EvaluationResult:
    """Simple data class to hold evaluation results."""
    def __init__(self, average_scores: Dict[str, float], detailed_scores: List[Dict[str, Any]]):
        self.average_scores: Dict[str, float] = average_scores # {model_id: avg_score}
        self.detailed_scores: List[Dict[str, Any]] = detailed_scores # List of dicts, one per data point

    def to_dict(self) -> Dict[str, Any]:
        """Converts the result to a dictionary."""
        return {
            "average_scores": self.average_scores,
            "detailed_scores": self.detailed_scores
        }

class BaseEvaluator(ABC):
    """Abstract Base Class for evaluation methods."""

    @abstractmethod
    def evaluate(
        self,
        run_results: List[Dict[str, Any]],
        prompt_template: str,
        **kwargs # Allow for extra arguments specific to implementations
    ) -> EvaluationResult:
        """
        Evaluates the LLM outputs based on the specific method.

        Args:
            run_results: A list of dictionaries, where each dictionary represents
                         a data point and contains:
                         - 'data_point_index': Index of the data point.
                         - 'data': The original input data for this point.
                         - 'outputs': A dictionary mapping model_id to the generated output string (or error message).
                         - Optional 'error' key if prompt formatting failed.
            prompt_template: The original prompt template used for generation.
            **kwargs: Additional keyword arguments needed by specific evaluators
                      (e.g., reasoning_model for ReasoningEvaluator).

        Returns:
            An EvaluationResult object containing aggregated and detailed scores.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
            Exception: Can raise specific exceptions related to the evaluation process.
        """
        raise NotImplementedError

    def _calculate_average_scores(self, detailed_scores: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Helper method to calculate average scores from detailed scores.
        Assumes detailed_scores is a list of dicts, where each dict contains
        a 'scores' key which is another dict {model_id: score}.
        """
        model_totals: Dict[str, float] = {}
        model_counts: Dict[str, int] = {}

        for item_scores in detailed_scores:
            scores = item_scores.get("scores")
            if not scores: # Handle cases where scoring might have failed for an item
                continue
            for model_id, score in scores.items():
                if score is not None and isinstance(score, (int, float)): # Check if score is valid
                    model_totals[model_id] = model_totals.get(model_id, 0) + score
                    model_counts[model_id] = model_counts.get(model_id, 0) + 1

        average_scores: Dict[str, float] = {}
        for model_id, total in model_totals.items():
            count = model_counts.get(model_id, 0)
            if count > 0:
                average_scores[model_id] = round(total / count, 2)
            else:
                average_scores[model_id] = 0.0 # Or None, or handle as error? Defaulting to 0.0

        return average_scores