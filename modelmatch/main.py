import argparse
import json
import sys
import os
import logging
from rich.console import Console
from modelmatch.logging_config import setup_logging, LOG_FORMAT_DETAILED
from modelmatch.core import runner
from modelmatch.utils.helper import (
    load_json_data,
    process_model_list_input,
    process_reasoning_model_input,
    list_models_and_exit,
    display_results
    )
# Import only needed functions from models

logger = logging.getLogger(__name__)




# --- Main Execution Logic ---
def main():
    # Pre-check for --list-models
    if '--list-models' in sys.argv:
        list_models_and_exit()

    # Setup Logging & Console
    setup_logging(level=logging.INFO, log_format=LOG_FORMAT_DETAILED)
    console = Console()

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Compares outputs from different LLMs based on input data and evaluates them.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Required Args
    parser.add_argument('-i', '--input-file', required=True, type=str, help='Path to the input JSON file.')
    parser.add_argument('-m', '--models', required=True, type=str, help='Comma-separated list of model IDs or display names...\nUse quotes if names contain spaces. Max 3 enforced.')
    parser.add_argument('-e', '--eval-method', required=True, choices=['human', 'reasoning'], type=str.lower, help='Evaluation method.')
    # Optional Args
    parser.add_argument('-r', '--reasoning-model', type=str, default=None, help='Model ID or display name for reasoning evaluation.')
    parser.add_argument('-o', '--output-file', type=str, default='modelmatch_results.json', help='Path to save the results JSON (default: modelmatch_results.json in current directory).')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level (default: INFO)')
    parser.add_argument('--max-workers',type=int,default=None, help='Max threads for parallel model calls per data point. (default: Python decides)')
    parser.add_argument('--list-models', action='store_true', help='List all configured models and exit.')
    parser.add_argument('--show-details', action='store_true', help='Display detailed evaluation results for each data point.')

    args = parser.parse_args()

    # Configure log level based on args
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers: handler.setLevel(log_level)
    logger.info(f"Logging level set to: {args.log_level.upper()}")

    # --- Validate Input File ---
    if not os.path.exists(args.input_file) or not os.path.isfile(args.input_file):
        logger.error(f"Input file not found or is not a file: {args.input_file}")
        console.print(f"[bold red]Error:[/bold red] Input file not found: {args.input_file}")
        sys.exit(1)

    logger.info("--- ModelMatch Initializing ---")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Evaluation method: {args.eval_method}")

    # --- Process and Validate Models using Helper ---
    try:
        final_model_ids = process_model_list_input(args.models, console)
        reasoning_model_id = process_reasoning_model_input(args.reasoning_model, args.eval_method, console)
    except SystemExit:
         raise # Allow SystemExit from helpers to terminate
    except Exception as e: # Catch unexpected errors during processing
         logger.error(f"Failed during model argument processing: {e}", exc_info=True)
         console.print(f"[bold red]Error processing model arguments:[/bold red] {e}")
         sys.exit(1)


    # --- Load Data ---
    try:
        input_data = load_json_data(args.input_file)
        logger.info(f"Successfully loaded data for prompt template and {len(input_data.get('data',[]))} data points.")
    except Exception as e:
        logger.error(f"Failed to load input data: {e}", exc_info=True)
        console.print(f"[bold red]Error loading input data from {args.input_file}:[/bold red] {e}")
        sys.exit(1)

    # --- Call the Core Runner ---
    logger.info("Starting comparison process...")
    try:
        results = runner.run_comparison(
            prompt_template=input_data["prompt_template"],
            data_points=input_data["data"],
            model_ids=final_model_ids,
            eval_method=args.eval_method,
            reasoning_model_id=reasoning_model_id,
            max_workers=args.max_workers
        )
        logger.info("Comparison process finished successfully.")
        # --- Display Results ---
        display_results(results, console, args.show_details) # Pass show_details flag

    except Exception as e:
        logger.error(f"Error during comparison: {e}", exc_info=True)
        console.print(f"\n[bold red]An error occurred during the comparison process:[/bold red]\n{e}")
        sys.exit(1)

    # --- Save results to file ---
    if args.output_file: # This condition will now always be true unless saving fails
        try:
            # Get the absolute path for clearer logging/messaging
            output_path = os.path.abspath(args.output_file)
            logger.info(f"Attempting to save full results to: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                # Make sure 'results' variable holds the data from runner.run_comparison
                json.dump(results, f, indent=2)
            logger.info(f"Full results saved successfully to: {output_path}")
            console.print(f"\n[green]Full results JSON saved to:[/green] {output_path}")
        except NameError:
             # Handle case where 'results' might not be defined if runner failed early
             logger.error("Cannot save results because the 'results' variable is not defined (comparison likely failed).")
             console.print("[bold red]Error:[/bold red] Cannot save results as comparison may have failed before generating them.")
        except IOError as e:
            output_path = os.path.abspath(args.output_file) # Get path again for error
            logger.error(f"Error saving results to {output_path}: {e}", exc_info=True)
            console.print(f"\n[bold red]Error saving results to {output_path}:[/bold red]\n{e}")
        except Exception as e: # Catch other potential errors like json serialization issues
            output_path = os.path.abspath(args.output_file)
            logger.error(f"Unexpected error saving results to {output_path}: {e}", exc_info=True)
            console.print(f"\n[bold red]Unexpected error saving results to {output_path}:[/bold red]\n{e}")

if __name__ == '__main__':
    main()