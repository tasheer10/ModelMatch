import logging
from typing import List, Dict, Any
from modelmatch.models import get_model
from modelmatch.utils.helper import format_prompt
# Import the factory and result class
from modelmatch.evaluation import get_evaluator, EvaluationResult

logger = logging.getLogger(__name__)

def run_comparison(
    prompt_template: str,
    data_points: List[Any],
    model_ids: List[str],
    eval_method: str,
    reasoning_model_id: str | None = None
) -> Dict[str, Any]:
    """
    Orchestrates the LLM comparison process.
    # ... (docstring mostly unchanged) ...
    """
    # --- Model Initialization ---
    logger.info(f"Initializing models: {', '.join(model_ids)}")
    try:
        models_to_run = {mid: get_model(mid) for mid in model_ids}
        reasoning_model = get_model(reasoning_model_id) if reasoning_model_id else None
        if reasoning_model_id:
             logger.info(f"Reasoning model '{reasoning_model_id}' initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}", exc_info=True)
        raise

    # --- Output Generation ---
    all_outputs_data: List[Dict[str, Any]] = [] # Renamed for clarity
    logger.info(f"Processing {len(data_points)} data points...")
    # ... (generation loop remains the same as before) ...
    # The loop appends dictionaries to all_outputs_data
    # Example dict in all_outputs_data:
    # {
    #     "data_point_index": i,
    #     "data": data_point,
    #     "outputs": {"model_id1": "output1", "model_id2": "ERROR: ..."},
    #     "error": None # Or error string if formatting failed
    # }

    logger.info("Generating outputs complete.")

    # --- Evaluation Phase ---
    logger.info(f"Starting evaluation using '{eval_method}' method...")
    evaluation_results_obj: EvaluationResult | None = None
    evaluation_error: str | None = None

    try:
        evaluator = get_evaluator(eval_method)
        eval_kwargs = {
            "run_results": all_outputs_data,
            "prompt_template": prompt_template,
        }
        # Pass the reasoning model instance only if needed
        if eval_method == 'reasoning':
            if not reasoning_model:
                 raise ValueError("Reasoning model instance is required for 'reasoning' evaluation but was not initialized.")
            eval_kwargs["reasoning_model"] = reasoning_model

        evaluation_results_obj = evaluator.evaluate(**eval_kwargs)
        logger.info("Evaluation phase completed successfully.")

    except ValueError as e:
        logger.error(f"Configuration error during evaluation setup: {e}")
        evaluation_error = f"Configuration Error: {e}"
    except KeyboardInterrupt:
         logger.warning("Evaluation interrupted by user (Ctrl+C).")
         evaluation_error = "Evaluation Interrupted by User"
         # Re-raise to stop the script gracefully if desired, or just record the error
         # raise
    except Exception as e:
        logger.error(f"An error occurred during the '{eval_method}' evaluation: {e}", exc_info=True)
        evaluation_error = f"Runtime Error: {e}"

    # --- Prepare Final Results ---
    final_evaluation_output = {}
    if evaluation_results_obj:
        final_evaluation_output = evaluation_results_obj.to_dict()
    elif evaluation_error:
         final_evaluation_output = {"error": evaluation_error}
    else:
         # Should not happen if try block completes without error or result
         final_evaluation_output = {"error": "Unknown evaluation state."}


    return {
        "parameters": {
            "prompt_template": prompt_template,
            "models_compared": model_ids,
            "evaluation_method": eval_method,
            "reasoning_model_id": reasoning_model_id,
            "num_data_points": len(data_points),
        },
        "raw_outputs_per_data_point": all_outputs_data,
        "evaluation": final_evaluation_output, # Changed key name
    }