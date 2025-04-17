import argparse
import json
import sys
import os
import logging # Import logging

# Import your setup function and runner etc.
from modelmatch.logging_config import setup_logging, LOG_FORMAT_DETAILED # Import setup function
from modelmatch.core import runner
from modelmatch.models import list_available_models
from modelmatch.utils.helper import load_json_data
# from llm_comparator.config import check_config

# Get a logger for this module
logger = logging.getLogger(__name__)

def main():
    """
    Main function to parse arguments and run the LLM comparison.
    """
    # --- Setup Logging FIRST ---
    # We could potentially control the level via CLI arg later
    # For now, default to INFO level
    setup_logging(level=logging.INFO, log_format=LOG_FORMAT_DETAILED)

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Compares outputs from different LLMs based on input data and evaluates them."
    )
    # Add arguments (keep as before)
    parser.add_argument('-i', '--input-file', required=True, type=str, help='Path to the input JSON file.')
    parser.add_argument('-m', '--models', required=True, type=str, help='Comma-separated list of model IDs to compare (e.g., "gpt-4o,gpt-3.5-turbo"). Max 3 enforced.')
    parser.add_argument('-e', '--eval-method', required=True, choices=['human', 'reasoning'], type=str.lower, help='Evaluation method.')
    parser.add_argument('-r', '--reasoning-model', type=str, default=None, help='Model ID to use for reasoning evaluation (required if --eval-method=reasoning).')
    parser.add_argument('-o', '--output-file', type=str, default=None, help='Optional path to save the results JSON.')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level (default: INFO)')
    parser.add_argument('--max-workers',type=int,default=None, help='Maximum number of threads to use for parallel model calls per data point. (default: Python decides)'
    )

    args = parser.parse_args()

    # --- Re-configure logging level based on argument ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    # Find the root logger's handlers and update their level
    # This is slightly more complex than just calling setup_logging again
    # An alternative is to pass the level directly to setup_logging initially
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level) # Set logger level
    for handler in root_logger.handlers:
        handler.setLevel(log_level) # Set handler levels too
    logger.info(f"Logging level set to: {args.log_level.upper()}")


    # --- Argument Validation (Use logger for errors/warnings) ---
    if not os.path.exists(args.input_file) or not os.path.isfile(args.input_file):
        logger.error(f"Input file not found or is not a file: {args.input_file}")
        sys.exit(1)

    # Optional: Run configuration checks at the start
    # check_config()

    logger.info("--- ModelMatch Initializing ---")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Evaluation method: {args.eval_method}")

    model_list = [m.strip() for m in args.models.split(',') if m.strip()]
    if not model_list:
        logger.error("No models specified.")
        sys.exit(1)

    if len(model_list) > 3:
         logger.warning("Currently limiting comparison to a maximum of 3 models.")
         model_list = model_list[:3] # Enforce limit for now

    logger.info(f"Models to compare: {', '.join(model_list)}")

    available_models = list_available_models()
    for model_id in model_list:
        if model_id not in available_models:
            logger.error(f"Model '{model_id}' is not supported or configured. Available: {', '.join(available_models)}")
            sys.exit(1)

    reasoning_model_id = args.reasoning_model

    if args.eval_method == 'reasoning':
        if not reasoning_model_id:
            logger.error("--reasoning-model is required when --eval-method is 'reasoning'.")
            sys.exit(1)
        if reasoning_model_id not in available_models:
             logger.error(f"Reasoning model '{reasoning_model_id}' is not supported or configured. Available: {', '.join(available_models)}")
             sys.exit(1)
        logger.info(f"Reasoning model: {reasoning_model_id}")
    elif reasoning_model_id:
        logger.warning("--reasoning-model is ignored when --eval-method is 'human'.")
        reasoning_model_id = None

    # --- Load Data ---
    try:
        input_data = load_json_data(args.input_file)
        logger.info(f"Successfully loaded data for prompt template and {len(input_data.get('data',[]))} data points.")
    except Exception as e:
        logger.error(f"Failed to load input data: {e}", exc_info=True)
        sys.exit(1)

    # --- Call the core runner ---
    logger.info("Starting comparison process...")
    try:
        results = runner.run_comparison(
            prompt_template=input_data["prompt_template"],
            data_points=input_data["data"],
            model_ids=model_list,
            eval_method=args.eval_method,
            reasoning_model_id=reasoning_model_id,
            max_workers=args.max_workers
        )
        logger.info("Comparison process finished successfully.")
    except Exception as e:
        logger.error(f"Error during comparison: {e}", exc_info=True) # Log traceback
        sys.exit(1)

    # --- Display/Save results ---
    logger.info("--- Comparison Results ---")
    # Use print for final result output unless you want it logged
    print(json.dumps(results, indent=2))

    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output_file}")
        except IOError as e:
            logger.error(f"Error saving results to {args.output_file}: {e}", exc_info=True)

if __name__ == '__main__':
    main()