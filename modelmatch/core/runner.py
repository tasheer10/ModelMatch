import logging
from typing import List, Dict, Any, Tuple
from modelmatch.models import get_model, LLM
import concurrent.futures
from modelmatch.utils.helper import format_prompt
from modelmatch.evaluation import get_evaluator, EvaluationResult

logger = logging.getLogger(__name__)


# --- Helper function to run generation in a thread ---
def _generate_single_output(model_id: str, model: LLM, prompt: str) -> Tuple[str, str]:
    """
    Calls the model's generate method and handles exceptions.
    Runs within a thread.

    Args:
        model_id: The ID of the model being run.
        model: The instantiated LLM object.
        prompt: The prompt to send to the model.

    Returns:
        A tuple containing (model_id, generated_output_or_error_string).
    """
    try:
        logger.debug(f"Thread starting generation for model: {model_id}")
        output = model.generate(prompt)
        logger.debug(f"Thread finished generation for model: {model_id} (Output length: {len(output)})")
        return model_id, output
    except Exception as e:
        logger.error(f"Thread error during generation for model {model_id}: {e}", exc_info=True)
        # Return the error message string, prefixed for clarity
        return model_id, f"ERROR: {e}"
# --- End Helper Function ---

def run_comparison(
    prompt_template: str,
    data_points: List[Any],
    model_ids: List[str],
    eval_method: str,
    reasoning_model_id: str | None = None,
    max_workers: int | None = None 
) -> Dict[str, Any]:
    """
    Orchestrates the LLM comparison process, running model generations in parallel.

    Args:
        prompt_template: The base prompt template string.
        data_points: A list of data items to be inserted into the template.
        model_ids: A list of model identifiers to compare.
        eval_method: The evaluation method ('human' or 'reasoning').
        reasoning_model_id: The model ID for reasoning evaluation (if applicable).
        max_workers: Max number of threads for parallel generation per data point.

    Returns:
        A dictionary containing the comparison results and evaluation.
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

    all_outputs_data: List[Dict[str, Any]] = []
    logger.info(f"Processing {len(data_points)} data points...")

    for i, data_point in enumerate(data_points):
        logger.info(f"--- Processing data point {i+1}/{len(data_points)} ---")
        point_results = {"data_point_index": i, "data": data_point, "outputs": {}}

        # 1. Format the prompt for this data point
        try:
            base_prompt = format_prompt(prompt_template, data_point)
            logger.debug(f"  Formatted prompt for data point {i+1} (length: {len(base_prompt)}).")
        except Exception as e:
             logger.warning(f"  Skipping data point {i+1} due to formatting error: {e}", exc_info=True)
             point_results["error"] = f"Prompt formatting error: {e}"
             all_outputs_data.append(point_results)
             continue # Skip to the next data point

        # 2. Run models in parallel for this data point
        # Use ThreadPoolExecutor to manage threads
        # The context manager ensures threads are cleaned up properly
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for all models for the current data point
            future_to_model_id = {
                executor.submit(_generate_single_output, model_id, model, base_prompt): model_id
                for model_id, model in models_to_run.items()
            }
            logger.info(f"  Submitted {len(future_to_model_id)} generation tasks to thread pool for data point {i+1}.")

            # 3. Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_model_id):
                model_id = future_to_model_id[future]
                try:
                    # Get the result from the future. This will be the tuple (model_id, output_or_error)
                    # The model_id from the tuple isn't strictly needed here as we have it, but the helper returns it.
                    _, output_or_error = future.result()
                    point_results["outputs"][model_id] = output_or_error
                    if isinstance(output_or_error, str) and output_or_error.startswith("ERROR:"):
                        logger.warning(f"  Received error result for model '{model_id}' on data point {i+1}.")
                    else:
                        logger.debug(f"  Received successful result for model '{model_id}' on data point {i+1}.")
                except Exception as e:
                    # This catches exceptions that might occur during future.result() itself,
                    # though _generate_single_output should ideally catch internal errors.
                    logger.error(f"  Exception retrieving result for model {model_id} on data point {i+1}: {e}", exc_info=True)
                    point_results["outputs"][model_id] = f"ERROR: Failed to retrieve result - {e}"

        all_outputs_data.append(point_results)
        logger.info(f"--- Finished processing data point {i+1}/{len(data_points)} ---")

    logger.info("Generating outputs complete for all data points.")

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