import logging
import json
from typing import List, Dict, Any, Tuple
import os
from modelmatch.models.base import LLM # Need the LLM type hint
from modelmatch.evaluation.base_eval import BaseEvaluator, EvaluationResult
from modelmatch.utils.helper import format_prompt # To show the exact prompt

# Get logger
logger = logging.getLogger(__name__)

class ReasoningEvaluator(BaseEvaluator):
    """Evaluator that uses another LLM to score outputs."""
    _PROMPT_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'evaluation_prompt.txt')
    JSON_FORMAT_EXAMPLE = """
{
  "scores": {
    "Output A": { "score": <score_A>, "reasoning": "<brief_reasoning_A>" },
    "Output B": { "score": <score_B>, "reasoning": "<brief_reasoning_B>" }
    // Add entries for all outputs provided
  }
}
"""
    def __init__(self):
        """Initializes the evaluator and loads the reasoning prompt template."""
        super().__init__() # Call base class init if it exists/needed
        self.REASONING_PROMPT_TEMPLATE = self._load_prompt_template()
        logger.info("ReasoningEvaluator initialized.")

    def _load_prompt_template(self) -> str:
        """Loads the reasoning prompt template from the external file."""
        logger.info(f"Loading reasoning prompt template from: {self._PROMPT_FILE_PATH}")
        try:
            if not os.path.exists(self._PROMPT_FILE_PATH):
                logger.error(f"Reasoning prompt file not found at: {self._PROMPT_FILE_PATH}")
                raise FileNotFoundError(f"Reasoning prompt file not found: {self._PROMPT_FILE_PATH}")

            with open(self._PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
                template = f.read()

            if not template:
                 logger.error(f"Reasoning prompt file is empty: {self._PROMPT_FILE_PATH}")
                 raise ValueError(f"Reasoning prompt file is empty: {self._PROMPT_FILE_PATH}")

            logger.debug("Reasoning prompt template loaded successfully.")
            return template
        except Exception as e:
            logger.error(f"Failed to load or validate reasoning prompt template: {e}", exc_info=True)
            # Re-raise as a runtime error to prevent evaluator usage without a prompt
            raise RuntimeError(f"Failed to load reasoning prompt from {self._PROMPT_FILE_PATH}") from e

    def _build_reasoning_prompt(
        self,
        original_prompt: str,
        data_point: Any,
        outputs_to_evaluate: Dict[str, str] # Temporary label -> output text
    ) -> str:
        """Constructs the prompt for the reasoning model."""

        outputs_section = ""
        for label, text in outputs_to_evaluate.items():
            outputs_section += f"--- {label} ---\n{text}\n\n"

        prompt = self.REASONING_PROMPT_TEMPLATE.format(
            original_prompt=original_prompt,
            data_point=str(data_point), # Ensure data point is stringified
            outputs_section=outputs_section.strip(),
            json_format_example=self.JSON_FORMAT_EXAMPLE.strip()
        )
        return prompt

    def _parse_reasoning_response(self, response_text: str, output_labels: List[str]) -> Dict[str, Tuple[int | None, str | None]]:
        """
        Parses the JSON response from the reasoning model.

        Args:
            response_text: The raw text output from the reasoning model.
            output_labels: The list of temporary labels used (e.g., ["Output A", "Output B"]).

        Returns:
            A dictionary mapping temporary label to a tuple of (score, reasoning).
            Score or reasoning might be None if parsing failed for that part.
        """
        parsed_scores = {label: (None, None) for label in output_labels} # Initialize with None
        try:
            # Basic cleaning: Find first '{' and last '}'
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start == -1 or end == -1 or start >= end:
                 raise json.JSONDecodeError("No valid JSON object found.", response_text, 0)

            json_str = response_text[start:end+1]
            data = json.loads(json_str)

            if "scores" not in data or not isinstance(data["scores"], dict):
                logger.warning(f"Reasoning response JSON missing 'scores' dictionary: {json_str}")
                return parsed_scores # Return defaults

            scores_dict = data["scores"]
            for label in output_labels:
                if label in scores_dict:
                    score_entry = scores_dict[label]
                    score = None
                    reasoning = None
                    if isinstance(score_entry, dict):
                        raw_score = score_entry.get("score")
                        reasoning = score_entry.get("reasoning", "")
                        # Attempt to convert score to int, handle errors gracefully
                        if isinstance(raw_score, (int, float)):
                             score = int(raw_score)
                        elif isinstance(raw_score, str) and raw_score.isdigit():
                             score = int(raw_score)
                        else:
                             logger.warning(f"Invalid score format for {label}: {raw_score}. Setting score to None.")
                    elif isinstance(score_entry, (int, float)): # Handle case where only score is provided
                         score = int(score_entry)
                         reasoning = "N/A"
                    else:
                        logger.warning(f"Unexpected format for score entry {label}: {score_entry}")

                    # Validate score range
                    if score is not None and not (1 <= score <= 10):
                        logger.warning(f"Score for {label} ({score}) is outside the valid 1-10 range. Setting score to None.")
                        score = None

                    parsed_scores[label] = (score, reasoning)
                else:
                    logger.warning(f"Label '{label}' not found in reasoning model's scores dictionary.")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from reasoning model: {e}\nResponse: {response_text[:500]}...") # Log snippet
        except Exception as e:
            logger.error(f"Unexpected error parsing reasoning response: {e}", exc_info=True)

        return parsed_scores


    def evaluate(
        self,
        run_results: List[Dict[str, Any]],
        prompt_template: str,
        reasoning_model: LLM, # Expecting an initialized LLM instance
        **kwargs # To catch any other args passed
    ) -> EvaluationResult:
        """Performs evaluation using a reasoning LLM."""
        if not reasoning_model:
             raise ValueError("ReasoningEvaluator requires a 'reasoning_model' instance.")

        logger.info(f"Starting evaluation using reasoning model: {reasoning_model.model_id}")
        detailed_scores: List[Dict[str, Any]] = []
        total_items = len(run_results)

        for i, result_item in enumerate(run_results):
            data_point = result_item.get("data")
            outputs = result_item.get("outputs", {})
            item_index = result_item.get("data_point_index", i)
            item_eval_details = {"data_point_index": item_index, "data": data_point, "scores": {}, "reasoning": {}}

            # Filter out errors and map model_id to temporary labels (Output A, B, C...)
            valid_outputs_map: Dict[str, str] = {} # model_id -> output_text
            temp_label_map: Dict[str, str] = {} # temp_label -> model_id
            current_label_code = ord('A')

            for model_id, output in outputs.items():
                if isinstance(output, str) and not output.startswith("ERROR:"):
                    temp_label = f"Output {chr(current_label_code)}"
                    valid_outputs_map[model_id] = output
                    temp_label_map[temp_label] = model_id
                    current_label_code += 1

            if not valid_outputs_map:
                logger.warning(f"No valid outputs to evaluate for data point {item_index + 1}. Skipping.")
                detailed_scores.append(item_eval_details) # Append with empty scores/reasoning
                continue

            # Need the actual prompt used for context
            try:
                formatted_prompt = format_prompt(prompt_template, data_point)
            except Exception as e:
                 logger.error(f"Failed to format original prompt for reasoning context for data point {item_index + 1}: {e}. Skipping.")
                 detailed_scores.append(item_eval_details)
                 continue

            # Build the prompt for the reasoning model
            outputs_for_reasoner = {label: valid_outputs_map[model_id] for label, model_id in temp_label_map.items()}
            reasoning_prompt = self._build_reasoning_prompt(formatted_prompt, data_point, outputs_for_reasoner)
            logger.debug(f"Generated reasoning prompt for data point {item_index + 1} (length: {len(reasoning_prompt)}).")

            # Call the reasoning model
            try:
                logger.info(f"Sending request to reasoning model ({reasoning_model.model_id}) for data point {item_index + 1}...")
                reasoning_response = reasoning_model.generate(reasoning_prompt)
                logger.debug(f"Received response from reasoning model (length: {len(reasoning_response)}).")

                # Parse the response
                parsed_results = self._parse_reasoning_response(reasoning_response, list(temp_label_map.keys()))

                # Map scores back to original model_ids
                for temp_label, (score, reason) in parsed_results.items():
                    original_model_id = temp_label_map.get(temp_label)
                    if original_model_id:
                        item_eval_details["scores"][original_model_id] = score # Score might be None
                        item_eval_details["reasoning"][original_model_id] = reason # Reasoning might be None or empty
                        logger.debug(f"Parsed score for {original_model_id}: {score}, Reasoning: {str(reason)[:100]}...")
                    else:
                         logger.warning(f"Could not map temporary label '{temp_label}' back to an original model ID.")

            except Exception as e:
                logger.error(f"Failed to get or parse evaluation from reasoning model for data point {item_index + 1}: {e}", exc_info=True)
                # Add scores as None for all models for this item if reasoning fails entirely
                for model_id in valid_outputs_map.keys():
                     item_eval_details["scores"][model_id] = None
                     item_eval_details["reasoning"][model_id] = f"ERROR: Reasoning failed - {e}"


            detailed_scores.append(item_eval_details)

        logger.info("Reasoning model evaluation complete.")

        # Calculate averages from the 'scores' part
        average_scores = self._calculate_average_scores(detailed_scores)
        logger.info(f"Calculated average scores from reasoning model: {average_scores}")

        return EvaluationResult(average_scores, detailed_scores)