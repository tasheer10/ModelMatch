import logging
import random
from typing import List, Dict, Any, Tuple

# Consider using Rich for nicer terminal interactions
try:
    from rich.console import Console
    from rich.prompt import Prompt, IntPrompt
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Define dummy classes if rich is not available
    class Console: pass
    class Prompt:
        @staticmethod
        def ask(prompt, default=None): return input(prompt)
    class IntPrompt:
         @staticmethod
         def ask(prompt, choices=None, default=None):
             while True:
                try:
                    val_str = input(prompt)
                    val = int(val_str)
                    if choices and str(val) not in choices:
                        print(f"Please enter one of {choices}.")
                        continue
                    return val
                except ValueError:
                    print("Invalid input. Please enter an integer.")
    class Panel:
         def __init__(self, content, title="", border_style=""): self.content=content; self.title=title
         def __str__(self): return f"--- {self.title} ---\n{self.content}\n------------------"


from modelmatch.evaluation.base_eval import BaseEvaluator, EvaluationResult
from modelmatch.utils.helper import format_prompt # To show the exact prompt

# Get logger
logger = logging.getLogger(__name__)

class HumanEvaluator(BaseEvaluator):
    """Evaluator that relies on human scoring."""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        logger.info(f"HumanEvaluator initialized. Rich support: {RICH_AVAILABLE}")

    def _display_item(self, index: int, total: int, prompt: str, data: Any):
        """Helper to display the current item context."""
        title = f"Data Point {index + 1}/{total}"
        content = f"[bold]Data:[/bold]\n{data}\n\n[bold]Prompt Used:[/bold]\n{prompt}"
        if self.console:
            self.console.print(Panel(content, title=title, border_style="blue"))
        else:
            print(f"\n--- {title} ---")
            print(f"Data:\n{data}\n")
            print(f"Prompt Used:\n{prompt}")
            print("------------------")

    def _get_human_score(self, output_text: str, display_index: int, total_outputs: int) -> int | None:
        """Gets a score from the user for a single output."""
        prompt_text = f"\n[bold]Evaluate Output {display_index}/{total_outputs}[/bold]\n---\n{output_text}\n---\nEnter score (1-10, or 0 to skip):"
        score = None
        while score is None:
            try:
                if self.console:
                    # Using IntPrompt for validation
                    score_val = IntPrompt.ask(
                        prompt_text,
                        choices=[str(i) for i in range(11)], # Allow 0-10
                        show_choices=False # Optional: Keep prompt cleaner
                    )
                else:
                     # Basic input validation
                     val_str = input(f"\nEvaluate Output {display_index}/{total_outputs}\n---\n{output_text}\n---\nEnter score (1-10, or 0 to skip): ")
                     score_val = int(val_str)

                if 0 <= score_val <= 10:
                    if score_val == 0:
                        logger.info("User chose to skip scoring for this output.")
                        return None # Indicate skipped score
                    return score_val
                else:
                    msg = "Invalid score. Please enter a number between 1 and 10 (or 0 to skip)."
                    if self.console: self.console.print(f"[red]{msg}[/red]")
                    else: print(msg)
                    score = None # Loop again
            except ValueError:
                msg = "Invalid input. Please enter an integer."
                if self.console: self.console.print(f"[red]{msg}[/red]")
                else: print(msg)
                score = None # Loop again
            except EOFError: # Handle Ctrl+D or similar
                 logger.warning("Input stream closed unexpectedly. Aborting scoring for this item.")
                 return None # Treat as skipped
            except KeyboardInterrupt: # Handle Ctrl+C
                 logger.warning("Keyboard interrupt detected. Aborting evaluation.")
                 raise # Re-raise to stop the whole process

    def evaluate(
        self,
        run_results: List[Dict[str, Any]],
        prompt_template: str,
        **kwargs # Not used here, but kept for signature consistency
    ) -> EvaluationResult:
        """Performs human evaluation by prompting the user."""
        logger.info("Starting human evaluation process...")
        detailed_scores: List[Dict[str, Any]] = []
        total_items = len(run_results)

        for i, result_item in enumerate(run_results):
            data_point = result_item.get("data")
            outputs = result_item.get("outputs", {})
            item_index = result_item.get("data_point_index", i)
            item_scores = {"data_point_index": item_index, "data": data_point, "scores": {}}

            # Filter out models that produced errors for this data point
            valid_outputs = {
                model_id: output for model_id, output in outputs.items()
                if not isinstance(output, str) or not output.startswith("ERROR:")
            }

            if not valid_outputs:
                logger.warning(f"No valid outputs to evaluate for data point {item_index + 1}. Skipping.")
                detailed_scores.append(item_scores) # Append with empty scores
                continue

            try:
                # Format the prompt as it was actually sent to the models
                formatted_prompt = format_prompt(prompt_template, data_point)
            except Exception as e:
                 logger.error(f"Failed to format prompt for display for data point {item_index + 1}: {e}. Skipping.")
                 detailed_scores.append(item_scores)
                 continue

            self._display_item(item_index, total_items, formatted_prompt, data_point)

            # Prepare outputs for random presentation
            output_items = list(valid_outputs.items()) # List of (model_id, output_text)
            random.shuffle(output_items) # Shuffle in place

            logger.info(f"Presenting {len(output_items)} outputs for scoring (random order).")

            for display_idx, (model_id, output_text) in enumerate(output_items):
                 # Get score from human - pass display index and total outputs for context
                 score = self._get_human_score(output_text, display_idx + 1, len(output_items))
                 item_scores["scores"][model_id] = score # Store score (or None if skipped)

            detailed_scores.append(item_scores)

        logger.info("Human evaluation scoring complete.")

        # Calculate averages
        average_scores = self._calculate_average_scores(detailed_scores)
        logger.info(f"Calculated average scores: {average_scores}")

        return EvaluationResult(average_scores, detailed_scores)