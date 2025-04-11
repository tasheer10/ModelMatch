import logging # Import logging
from typing import List, Dict, Any
from llm_comparator.models import get_model
from llm_comparator.utils.helpers import format_prompt
# from llm_comparator.evaluation.human import evaluate_human
# from llm_comparator.evaluation.reasoning import evaluate_reasoning

# Get a logger for this module
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
    # ... (docstring unchanged) ...
    """
    logger.info(f"Initializing models: {', '.join(model_ids)}")
    try:
        models_to_run = {mid: get_model(mid) for mid in model_ids}
        reasoning_model = get_model(reasoning_model_id) if reasoning_model_id else None
        if reasoning_model_id:
             logger.info(f"Reasoning model '{reasoning_model_id}' initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}", exc_info=True)
        raise # Re-raise the exception to stop execution

    all_outputs = []

    logger.info(f"Processing {len(data_points)} data points...")
    for i, data_point in enumerate(data_points):
        logger.debug(f"Processing data point {i+1}/{len(data_points)}: {str(data_point)[:100]}...") # Log start, truncate data
        point_results = {"data_point_index": i, "data": data_point, "outputs": {}}

        try:
            base_prompt = format_prompt(prompt_template, data_point) # helpers.py needs logging
            logger.debug(f"  Formatted prompt for data point {i+1} (length: {len(base_prompt)}).")
        except Exception as e:
             logger.warning(f"Skipping data point {i+1} due to formatting error: {e}", exc_info=True)
             point_results["error"] = f"Prompt formatting error: {e}"
             all_outputs.append(point_results)
             continue

        for model_id, model in models_to_run.items():
            logger.info(f"  Running model '{model_id}' for data point {i+1}...")
            try:
                # TODO: Consider Async Execution Here
                output = model.generate(base_prompt) # model provider needs logging
                point_results["outputs"][model_id] = output
                logger.debug(f"    -> Output received from '{model_id}' (length: {len(output)} chars)")
            except Exception as e:
                logger.error(f"    -> Error generating output from {model_id}: {e}", exc_info=True)
                point_results["outputs"][model_id] = f"ERROR: {e}" # Store error

        all_outputs.append(point_results)
        logger.info(f"  Finished processing data point {i+1}/{len(data_points)}.")

    logger.info("Generating outputs complete.")

    # --- Evaluation Phase ---
    logger.info(f"Starting evaluation using '{eval_method}' method...")
    final_scores = {}
    if eval_method == 'human':
        # final_scores = evaluate_human(all_outputs)
        logger.info("--> Human evaluation required (manual step - TBD)")
        final_scores = {"message": "Human evaluation needed. Raw outputs collected."}
        # TODO: Implement llm_comparator.evaluation.human.evaluate_human
    elif eval_method == 'reasoning':
        if not reasoning_model:
             # This condition should ideally be caught earlier in main.py, but good to double-check
             logger.error("Reasoning model required for 'reasoning' evaluation method but not available.")
             raise ValueError("Reasoning model not initialized for evaluation.")
        # final_scores = evaluate_reasoning(all_outputs, reasoning_model, prompt_template)
        logger.info(f"--> Evaluating using reasoning model: {reasoning_model_id}")
        final_scores = {"message": f"Reasoning evaluation using {reasoning_model_id} not yet implemented."}
        # TODO: Implement llm_comparator.evaluation.reasoning.evaluate_reasoning

    logger.info("Evaluation phase complete.")

    return {
        "parameters": {
            "prompt_template": prompt_template,
            "models_compared": model_ids,
            "evaluation_method": eval_method,
            "reasoning_model_id": reasoning_model_id,
            "num_data_points": len(data_points),
        },
        "raw_outputs_per_data_point": all_outputs,
        "evaluation_scores": final_scores,
    }