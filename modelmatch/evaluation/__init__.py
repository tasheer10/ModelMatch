import logging
from .base_eval import BaseEvaluator, EvaluationResult
from .human import HumanEvaluator
from .reasoning import ReasoningEvaluator
from modelmatch.models.base import LLM # For type hinting

logger = logging.getLogger(__name__)

EVALUATORS = {
    "human": HumanEvaluator,
    "reasoning": ReasoningEvaluator,
}

def get_evaluator(method: str, **kwargs) -> BaseEvaluator:
    """
    Factory function to get an instance of an evaluator.

    Args:
        method: The evaluation method name ('human' or 'reasoning').
        **kwargs: Additional arguments needed to initialize the evaluator
                  (e.g., reasoning_model instance for ReasoningEvaluator, though
                   currently it's passed to the evaluate method instead).

    Returns:
        An instance of the requested evaluator.

    Raises:
        ValueError: If the method is not supported.
    """
    evaluator_class = EVALUATORS.get(method.lower())
    if evaluator_class:
        logger.info(f"Initializing evaluator: {method}")
        # Currently, evaluators don't take args in __init__, but this supports future expansion
        return evaluator_class(**kwargs)
    else:
        logger.error(f"Unsupported evaluation method requested: {method}")
        raise ValueError(f"Unsupported evaluation method: '{method}'. Supported methods are: {list(EVALUATORS.keys())}")