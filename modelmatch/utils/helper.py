import sys
import logging
from typing import List, Optional, Set, Tuple, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule 
import json
from modelmatch.models import (
    list_available_models,
    list_available_models_display,
    get_model_id_from_display_name
)


# Get a logger for this module
logger = logging.getLogger(__name__)

def load_json_data(filepath: str) -> Dict[str, Any]:
    """Loads data from a JSON file."""
    logger.debug(f"Attempting to load JSON data from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Successfully decoded JSON from {filepath}")

        # Validation
        if "prompt_template" not in data or "data" not in data:
             msg = "Input JSON must contain 'prompt_template' and 'data' keys."
             logger.error(f"{msg} File: {filepath}")
             raise ValueError(msg)
        if not isinstance(data["data"], list):
             msg = "'data' key must contain a list of data points."
             logger.error(f"{msg} File: {filepath}")
             raise ValueError(msg)

        logger.info(f"Loaded JSON data successfully from {filepath}.")
        return data
    except FileNotFoundError:
        logger.error(f"Input file not found at {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Could not decode JSON from {filepath}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred loading {filepath}: {e}", exc_info=True)
        raise


def format_prompt(template: str, data_point: Any) -> str:
    """Formats the prompt template with the given data point."""
    logger.debug(f"Formatting prompt template with data: {str(data_point)[:100]}...")
    try:
        if isinstance(data_point, dict):
             formatted = template.format(**data_point)
        else:
             try:
                 formatted = template.format(data=data_point)
             except KeyError:
                 logger.warning(f"Could not find '{{data}}' placeholder in template when formatting with non-dict data. Using template directly. Data: {data_point}")
                 formatted = template # Return template as is, maybe log warning

        logger.debug(f"Prompt formatted successfully (length: {len(formatted)}).")
        return formatted
    except KeyError as e:
        logger.error(f"Missing key '{e}' in data point {data_point} needed for prompt template: '{template}'", exc_info=True)
        raise ValueError(f"Missing key '{e}' in data point needed for prompt template") from e
    except Exception as e:
        logger.error(f"Error formatting prompt template '{template}' with data '{data_point}': {e}", exc_info=True)
        raise ValueError(f"Failed to format prompt template") from e

def process_model_list_input(models_arg: str, console: Console) -> List[str]:
    """
    Parses, translates display names, validates, and deduplicates model names/IDs.

    Args:
        models_arg: The comma-separated string from the -m argument.
        console: Rich console object for printing errors.

    Returns:
        A validated list of unique model IDs.

    Raises:
        SystemExit: If validation fails.
    """
    user_model_names = [name.strip() for name in models_arg.split(',') if name.strip()]
    if not user_model_names:
        logger.error("No models specified in the -m argument.")
        console.print("[bold red]Error:[/bold red] No models provided via the -m argument.")
        sys.exit(1)

    final_model_ids: List[str] = []
    valid_model_ids: Set[str] = set(list_available_models())
    all_display_formats: List[Tuple[str, str]] = list_available_models_display()

    for name in user_model_names:
        translated_id = None
        if name in valid_model_ids:
            translated_id = name
            logger.debug(f"Model name '{name}' is a valid model ID.")
        else:
            found_id = get_model_id_from_display_name(name)
            if found_id:
                translated_id = found_id
                logger.debug(f"Translated display name '{name}' to model ID '{translated_id}'.")
            else:
                logger.error(f"Model name '{name}' is not a recognized model ID or display name.")
                help_msg = "Please use one of the following configured models:\n"
                for disp_name, model_id in all_display_formats:
                    help_msg += f"  - \"{disp_name}\"  (ID: {model_id})\n"
                console.print(f"[bold red]Error:[/bold red] Model '{name}' not found.\n\n{help_msg}")
                sys.exit(1)

        if translated_id:
            if translated_id in final_model_ids:
                logger.warning(f"Model ID '{translated_id}' (from input '{name}') specified multiple times. Using only once.")
            else:
                final_model_ids.append(translated_id)

    if not final_model_ids:
        logger.error("No valid models could be resolved from the input.")
        console.print("[bold red]Error:[/bold red] No valid models resolved from the -m argument.")
        sys.exit(1)

    # Apply limit *after* validation and deduplication
    if len(final_model_ids) > 3:
        logger.warning(f"Limiting comparison to the first 3 valid models provided: {', '.join(final_model_ids[:3])}")
        final_model_ids = final_model_ids[:3]

    logger.info(f"Models to compare (resolved IDs): {', '.join(final_model_ids)}")
    return final_model_ids


def process_reasoning_model_input(
    reasoning_model_arg: Optional[str],
    eval_method: str,
    console: Console
) -> Optional[str]:
    """
    Validates and translates the reasoning model name/ID if needed.

    Args:
        reasoning_model_arg: The string from the -r argument (or None).
        eval_method: The evaluation method ('human' or 'reasoning').
        console: Rich console object for printing errors.

    Returns:
        The validated reasoning model ID string, or None if not applicable/provided.

    Raises:
        SystemExit: If validation fails for reasoning method.
    """
    reasoning_model_id: Optional[str] = None
    if eval_method == 'reasoning':
        if not reasoning_model_arg:
            logger.error("--reasoning-model is required when --eval-method is 'reasoning'.")
            console.print("[bold red]Error:[/bold red] --reasoning-model is required when --eval-method is 'reasoning'.")
            sys.exit(1)

        reasoning_input_name = reasoning_model_arg.strip()
        valid_model_ids: Set[str] = set(list_available_models())
        all_display_formats: List[Tuple[str, str]] = list_available_models_display()

        if reasoning_input_name in valid_model_ids:
            reasoning_model_id = reasoning_input_name
        else:
            found_id = get_model_id_from_display_name(reasoning_input_name)
            if found_id:
                reasoning_model_id = found_id
            else:
                logger.error(f"Reasoning model name '{reasoning_input_name}' is not a recognized model ID or display name.")
                help_msg = "Available models:\n"
                for disp_name, model_id in all_display_formats:
                    help_msg += f"  - \"{disp_name}\" (ID: {model_id})\n"
                console.print(f"[bold red]Error:[/bold red] Reasoning model '{reasoning_input_name}' not found.\n\n{help_msg}")
                sys.exit(1)

        logger.info(f"Reasoning model (resolved ID): {reasoning_model_id}")

    elif reasoning_model_arg: # Provided -r but eval_method is 'human'
        logger.warning("--reasoning-model is ignored when --eval-method is 'human'.")
        reasoning_model_id = None

    return reasoning_model_id

def display_results(results: Dict[str, Any], console: Console, show_details: bool):
    """Displays the results dictionary using Rich components."""
    console.print(Rule(title="ModelMatch Results", style="bold blue"))

    # --- Parameters ---
    params = results.get("parameters", {})
    if params: # Only print if parameters exist
        param_str = "\n".join([f"[cyan]{k}:[/cyan] {v}" for k, v in params.items()])
        console.print(Panel(param_str, title="Run Parameters", border_style="green", expand=False))
    else:
        logger.warning("No parameters found in results to display.")

    # --- Evaluation ---
    evaluation = results.get("evaluation", {})
    if not evaluation:
        console.print(Panel("[yellow]No evaluation results generated.[/yellow]", title="Evaluation", border_style="yellow"))
        return

    if error := evaluation.get("error"):
        console.print(Panel(f"[bold red]Evaluation Error:[/bold red]\n{error}", title="Evaluation Failed", border_style="red"))
        return # Don't attempt to show scores if eval failed

    avg_scores = evaluation.get("average_scores")
    detailed_scores = evaluation.get("detailed_scores", [])

    if avg_scores:
        console.print(Rule(title="Evaluation Summary", style="bold blue"))
        # --- Average Scores Table ---
        table = Table(title="Average Evaluation Scores", show_header=True, header_style="bold magenta")
        table.add_column("Model ID", style="dim", min_width=30, overflow="fold") # Use min_width
        table.add_column("Average Score", justify="right")
        table.add_column("Rank", justify="right")

        # Filter out potential None scores before sorting if necessary, or handle in key
        valid_scores = {k: v for k, v in avg_scores.items() if v is not None}
        # Sort scores descending. Handle potential None values if not filtered.
        sorted_scores = sorted(
            valid_scores.items(),
            # avg_scores.items(), # Use this if you want N/A rows included but ranked last
            key=lambda item: item[1] if item[1] is not None else -float('inf'), # Rank None scores last
            reverse=True
        )

        # --- Corrected Ranking Logic v3 ---
        ranks = {}
        current_rank = 0
        last_score = None
        for i, (model_id, score) in enumerate(sorted_scores):
            # Only increment rank number if score is different from the previous score
            if score != last_score:
                current_rank = i + 1 # Rank is determined by position in sorted list
                last_score = score
            ranks[model_id] = current_rank # Store the calculated rank for this model

        # Add rows using the calculated ranks
        for model_id, score in sorted_scores: # Iterate again or use the stored ranks
            score_str = f"{score:.2f}" # Already filtered Nones
            rank_str = str(ranks.get(model_id, "-")) # Get rank from stored dict
            table.add_row(model_id, score_str, rank_str)

        # Optional: Add rows for models with N/A scores if they weren't filtered
        # for model_id, score in avg_scores.items():
        #     if score is None:
        #         table.add_row(model_id, "[red]N/A[/red]", "-")

        # --- End Corrected Ranking Logic v3 ---

        console.print(table)
    else:
        console.print(Panel("[yellow]Average scores could not be calculated.[/yellow]", title="Evaluation Summary", border_style="yellow"))


    # --- Display Detailed Scores (If requested and available) ---
    if show_details and detailed_scores:
        console.print(Rule(title="Detailed Results per Data Point", style="bold blue"))
        for item_detail in detailed_scores:
            dp_index = item_detail.get("data_point_index", "N/A")
            dp_data = item_detail.get("data", "N/A")
            dp_scores = item_detail.get("scores", {})
            dp_reasoning = item_detail.get("reasoning") # Might be None or dict

            # Panel for the data point context
            data_panel_content = f"[bold cyan]Index:[/bold cyan] {dp_index}\n[bold cyan]Input Data:[/bold cyan]\n{json.dumps(dp_data, indent=2)}"
            console.print(Panel(data_panel_content, title=f"Data Point {dp_index}", border_style="yellow", expand=False))

            if not dp_scores:
                 console.print("[dim]  No scores available for this data point.[/dim]")
                 console.print() # Add spacing
                 continue

            # Table for scores for this data point
            detail_table = Table(show_header=True, header_style="bold cyan", expand=False, show_edge=False) # Cleaner look maybe
            detail_table.add_column("Model", style="green")
            detail_table.add_column("Score", justify="center")
            if dp_reasoning and isinstance(dp_reasoning, dict) and any(dp_reasoning.values()): # Add reasoning column only if reasoning data exists and is not empty
                 detail_table.add_column("Reasoning", overflow="fold", min_width=40) # Give reasoning more space

            # Sort models within the detail table for consistency if desired (e.g., alphabetically)
            sorted_model_scores = sorted(dp_scores.items())

            for model_id, score in sorted_model_scores:
                score_str = str(score) if score is not None else "[red]N/A[/red]"

                row_data = [model_id, score_str]
                if dp_reasoning and isinstance(dp_reasoning, dict): # Check again before accessing
                    reason_str = dp_reasoning.get(model_id)
                    # Add reasoning only if it exists for this model and reasoning column is present
                    if reason_str:
                       row_data.append(reason_str)
                    # If the column exists but this model has no reasoning, add placeholder?
                    # elif detail_table.columns[2].header == "Reasoning": # Check if reasoning column exists
                    #     row_data.append("[dim]N/A[/dim]")


                # Handle case where reasoning column exists but this model has no reasoning
                if len(detail_table.columns) == 3 and len(row_data) == 2:
                    row_data.append("[dim]N/A[/dim]")


                detail_table.add_row(*row_data)

            console.print(detail_table)
            console.print() # Add a blank line for spacing

    elif detailed_scores:
        # Inform user details are available if they didn't ask
        console.print("\n[dim]Detailed scores per data point available. Use --show-details to display.[/dim]")


# --- Function to list models (moved logic here) ---
def list_models_and_exit():
    console = Console()
    try:
        available_models_list = list_available_models_display()
        if not available_models_list:
            console.print("[yellow]No models configured or config/models.yaml not found/invalid.[/yellow]")
        else:
            table = Table(title="Available Models", show_header=True, header_style="bold cyan")
            table.add_column("Display Name", style="green", min_width=20)
            table.add_column("Model ID (for API/internal use)", style="dim", min_width=30)
            for disp_name, model_id in available_models_list:
                table.add_row(disp_name, model_id)
            console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error loading model list:[/bold red] {e}")
    sys.exit(0)
