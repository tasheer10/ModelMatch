# ModelMatch
_Choose the right LLM, every time._

<!-- Add badges here later: e.g., build status, version, license -->
<!-- [![Build Status](https://img.shields.io/travis/your-username/ModelMatch.svg?style=flat-square)](https://travis-ci.org/your-username/ModelMatch) -->
<!-- [![PyPI version](https://img.shields.io/pypi/v/modelmatch.svg?style=flat-square)](https://pypi.org/project/modelmatch/) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

ModelMatch is a Python framework designed to help developers and researchers compare and evaluate the outputs of various Large Language Models (LLMs) for specific prompts and datasets. Selecting the optimal LLM for a particular use case can be challenging; ModelMatch provides a structured approach to gather outputs and assess their quality through either human judgment or automated evaluation using a separate reasoning LLM.

## ✨ Key Features

*   **Side-by-Side Comparison:** Run the same prompt and data points across multiple configured LLMs.
*   **Multiple LLM Provider Support:** Easily integrate models from different providers (e.g., OpenAI, OpenRouter) with an extensible provider system.
*   **Flexible Evaluation Methods:**
    *   **Human Evaluation:** Presents model outputs blindly (random order, anonymized) for human scoring (1-10).
    *   **Reasoning Model Evaluation:** Uses a designated LLM to assess and score the outputs from the models being compared.
*   **Configuration Driven:** Define supported models and their properties via external YAML files (`model_config.yaml`).
*   **Customizable Reasoning:** Edit the prompt used for the reasoning model evaluator (`evaluation_prompt.txt`).

*   **Parallel Execution:** Utilizes threading to run API calls to different models concurrently for faster results.
*   **Command-Line Interface:** Easy-to-use CLI for running comparisons and configuring options.
*   **User-Friendly Model Selection:** Refer to models by either their API ID or a user-defined display name in CLI arguments.
*   **Structured Output:** Results are provided in JSON format and summarized neatly in the console using Rich tables.

## 🚀 Installation

1.  **Prerequisites:**
    *   Python (>=3.11 recommended, check `pyproject.toml`)
    *   Poetry (>=1.2 recommended) for dependency management. See [Poetry installation guide](https://python-poetry.org/docs/#installation).

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/tasheer10/ModelMatch.git
    cd ModelMatch
    ```

3.  **(Optional but Recommended) Configure Poetry for In-Project Virtual Environment:**
    ```bash
    poetry config virtualenvs.in-project true
    ```

4.  **Install Dependencies:**
    ```bash
    poetry install
    ```
    This will create a virtual environment (`.venv` folder if configured above) and install all necessary packages (including `PyYAML`, `openai`, `rich`, etc.).

5.  **Activate Virtual Environment:**
    ```bash
    poetry shell
    ```
    Alternatively, prefix all commands with `poetry run`.

6.  **Configure API Keys:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your API keys for the LLM providers you intend to use (e.g., `OPENAI_API_KEY`, `OPENROUTER_API_KEY`). Ensure the variable names match those expected in `modelmatch/config.py`.

## ▶️ Usage

ModelMatch is run from the command line. Ensure your virtual environment is active (`poetry shell`) or prefix commands with `poetry run`.

**1. List Available Models:**
Check which models are configured in `model_config.yaml`:
```bash
modelmatch --list-models
```

**2. Run a Comparison (Human Evaluation Example):**
Use model display names (quotes needed if they contain spaces).
```bash
modelmatch -i data/example_input.json \
           -m "llama-3.3-70b-instruct,mistral-24b-instruct" \
           -e human
```
*(You will be prompted interactively to score outputs.)*

**3. Run a Comparison (Reasoning Model Example):**
Use a mix of display names and model IDs.
```bash
modelmatch -i data/example_input.json \
           -m llama-4-maverick,mistral-24b-instruct \
           -e reasoning \
           -r "gemini-2.0-flash-thinking-exp" # Use display name or ID for reasoning model
```

**4. Show Detailed Results on Console:**
Add the `--show-details` flag to any evaluation run to see scores/reasoning per data point.
```bash
modelmatch -i data/example_input.json \
           -m llama-4-maverick,mistral-24b-instruct \
           -e reasoning  -r "gemini-2.0-flash-thinking-exp" \
           --show-details
```

**5. Save Full JSON Output to a Specific File:**
Results are saved to `modelmatch_results.json` by default. Use `-o` to specify a different path.
```bash
modelmatch -i data/input.json -m m1,m2 -e human -o my_comparison_results.json
```

**Command-Line Arguments:**
*   `-i, --input-file`: (Required) Path to the input JSON data file.
*   `-m, --models`: (Required) Comma-separated list of model IDs or display names to compare (use quotes if names have spaces). Max 3 enforced.
*   `-e, --eval-method`: (Required) Evaluation method (`human` or `reasoning`).
*   `-r, --reasoning-model`: Model ID or display name for the reasoning evaluator (required if `-e reasoning`).
*   `-o, --output-file`: Path to save the detailed JSON results (default: `modelmatch_results.json`).
*   `--log-level`: Set logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO.
*   `--max-workers`: Max parallel threads for API calls per data point (default: Python decides).
*   `--list-models`: List configured models and exit.
*   `--show-details`: Display detailed evaluation results for each data point on the console.

## Example Scenario: Meeting Summary & Action Item Generation

This example demonstrates comparing LLMs for processing meeting transcripts to generate concise summaries, identify key decisions, and extract actionable items, all formatted as structured JSON. This task requires understanding context, identifying importance, and adhering to a specific output schema.

**1. Prepare Input Data (`data/example_data.json`):**

Create a JSON file containing snippets of meeting transcripts (or full transcripts).

```json
{
  "prompt_template": "You are an expert meeting analyst. Analyze the following meeting transcript excerpt. Provide a concise summary (max 3 sentences), list key decisions made, and extract specific action items with assigned owners (if mentioned). Respond ONLY with a JSON object containing 'summary', 'decisions' (list of strings), and 'action_items' (list of objects with 'task' and 'owner' keys).\n\nTranscript Excerpt:\n```\n{transcript}\n```\n\nAnalysis JSON:",
  "data": [
    {
      "transcript": "Okay team, project Alpha update. Marketing campaign launch is set for July 15th, that's final - Sarah to confirm budget by EOD Friday. John mentioned the server migration needs a P0 fix, he'll coordinate with Infra Ops immediately. Design mockups look good, Lisa needs feedback by Wednesday. Any blockers? No? Great."
    },
    {
      "transcript": "Discussion point: Q4 roadmap. We agreed to prioritize the 'User Profile V2' feature. Feature 'Analytics Dashboard' is pushed to Q1. Mark will draft the initial spec for V2 and share it next Monday for review. We need to de-scope 'Real-time Collaboration' for now due to resource constraints. Decision stands."
    },
    {
      "transcript": "Client feedback call summary: Positive overall, but they highlighted slow loading times on the main dashboard again. Critical issue. Peter, can you investigate the backend performance this week? We need to provide them an update in the Friday sync. Also discussed expanding the reporting features, but no decision made, tabled for next time."
    }
  ]
}
```
*(Note: The prompt requires specific JSON keys and value types, testing schema adherence.)*

**2. Configure Models:**

Ensure models like "llama-4-maverick,mistral-24b-instruct", and potentially a more specialized or fine-tuned model are defined in `model_config.yaml` with keys in `.env`. We'll also use "gemini-2.0-flash-thinking-exp" as the reasoning evaluator.

**3. Run the Comparison (using Reasoning Model with Details):**

Execute ModelMatch, requesting detailed output to analyze the reasoning:

```bash
modelmatch -i data/example_data.json \
           -m llama-4-maverick,mistral-24b-instruct \
           -e reasoning \
           -r "gemini-2.0-flash-thinking-exp" \
           --show-details
```

**4. Review Results:**

ModelMatch performs the following:
*   Executes the analysis prompt for each transcript excerpt on both "Llama 4" and "Mistral".
*   Sends each transcript, the prompt, and the two generated JSON outputs to the reasoning model ("Gemini Thinking Experimental"). The reasoning prompt (`evaluation_prompt.txt`) should guide it to evaluate:
    *   Accuracy of summary, decisions, and action items.
    *   Completeness (were all items captured?).
    *   Correctness of owner assignment.
    *   Strict adherence to the requested JSON schema (keys, types).
    *   Conciseness of the summary.
*   Parses the scores and reasoning.
*   Displays the summary table and detailed results (because of `--show-details`) on the console.
*   Saves the comprehensive results to `modelmatch_results.json`.

By reviewing the detailed output, especially the reasoning provided by the evaluator for each data point, you can gain insights beyond just the average score. For example, one model might consistently produce valid JSON but miss subtle action items, while another might capture everything but occasionally fail on the JSON structure. This helps you choose the model that best balances accuracy, completeness, and reliability for *this specific structured generation task*.

## Sample Output

After running a comparison, ModelMatch provides output both on the console and in a detailed JSON file.

**1. Console Output (Summary)**

The console output provides a high-level summary, including run parameters and the ranked average scores from the evaluation.

![Console output summary table](data/example_outputs/SampleAnalysisReport.png "ModelMatch Console Summary Output")

**2. Console Output (Detailed View)**

Using the `--show-details` flag, you can see the evaluation results (scores and reasoning, if applicable) broken down for each data point and model directly in the console.

![Detailed console output with reasoning per data point](data/example_outputs/SampleDetailedAnalysis.png "ModelMatch Detailed Console Output")

**3. JSON Output File (`modelmatch_results.json` by default)**

A comprehensive JSON file is saved, containing all parameters, the raw outputs generated by each model for every data point, and the full evaluation results (average and detailed scores/reasoning). This is useful for archival, programmatic analysis, or deeper dives.

![Snippet of the output JSON file structure](data/example_outputs/SampleResultJson.png "ModelMatch JSON Output Snippet")

This detailed JSON allows you to programmatically analyze results, review specific outputs, and understand the reasoning behind the scores (if using the reasoning evaluator).

## ⚙️ Configuration

*   **API Keys (`.env`):** Store your sensitive API keys in the `.env` file in the project root. See `.env.example`.
*   **Models (`model_config.yaml`):** Define the LLMs available for comparison. Each entry under the `models:` list requires:
    *   `display_name`: User-friendly name used in CLI arguments and output. Should ideally be unique.
    *   `model_id`: The exact identifier required by the provider's API. Must be unique.
    *   `provider`: The name of the Python class in `modelmatch/models/providers/` that handles this model type (e.g., `OpenAIModel`, `OpenRouterModel`). Ensure the class exists and is mapped in `modelmatch/models/__init__.py`.
*   **Reasoning Prompt (`evaluation_prompt.txt`):** Modify this file to customize the instructions given to the reasoning model during AI-based evaluation. Ensure the placeholders (`{original_prompt}`, `{data_point}`, `{outputs_section}`, `{json_format_example}`) are kept.

## 💾 Input Data Format

Input data is provided via a JSON file specified with the `-i` argument. It should have the following structure:

```json
{
  "prompt_template": "Your prompt template here. Use placeholders like {data} or named placeholders like {field1}, {field2} that correspond to keys in your data points if they are dictionaries.",
  "data": [
    "Data point 1 (e.g., a string to fill {data})",
    "Data point 2",
    { "field1": "value A", "field2": "value B" },
    "..."
  ]
}
```

*   `prompt_template`: The base prompt. Placeholders will be filled with items from the `data` list using standard Python `.format()` logic.
*   `data`: A list of data items. Each item will be used to format the `prompt_template` once. Items can be simple strings (used if the template contains `{data}`) or dictionaries (used if the template contains named placeholders like `{field1}`).

## 📊 Evaluation Methods

*   **Human (`-e human`):**
    *   Outputs for each data point are shown one by one in random order.
    *   The model generating the output is hidden.
    *   The user is prompted to provide a score from 1 (worst) to 10 (best) or 0 to skip.
    *   Scores are averaged per model across all data points.
*   **Reasoning (`-e reasoning -r <reasoning_model>`):**
    *   For each data point, the original prompt, input data, and all valid model outputs are sent to the specified reasoning LLM.
    *   The reasoning LLM is prompted (using `evaluation_prompt.txt`) to score each output (1-10) and provide justification, returning structured JSON.
    *   These scores are parsed and averaged per model. Requires careful prompt engineering in `evaluation_prompt.txt` for reliable results.

## 📄 Output Format

*   **Console:** Uses the Rich library to display:
    *   Run parameters.
    *   A summary table of average scores, ranked correctly (handling ties).
    *   (Optional) Detailed tables showing scores/reasoning for each model per data point if `--show-details` is used.
*   **JSON File (`-o`):** Saves a detailed JSON file (default: `modelmatch_results.json`) containing:
    *   `parameters`: The configuration used for the run.
    *   `raw_outputs_per_data_point`: A list containing the original data and the raw text output (or error string like `"ERROR: ..."`) from each compared model for every data point.
    *   `evaluation`: Contains the results of the evaluation:
        *   `average_scores`: Dictionary mapping `model_id` to average score.
        *   `detailed_scores`: A list where each item corresponds to a data point and contains the scores (key: `scores`) and reasoning (key: `reasoning`, if applicable) assigned to each model for that specific data point.
        *   `error`: An error message if the evaluation phase itself failed.

## 🧩 Extensibility

Adding support for new LLM providers is designed to be straightforward:

1.  **Create Provider Class:** Create a new Python file in `modelmatch/models/providers/` (e.g., `anthropic.py`). Define a class (e.g., `AnthropicModel`) that inherits from `modelmatch.models.base.LLM`.
2.  **Implement Methods:** Implement the `__init__` method (to handle API key loading from `modelmatch.config.settings` and any client initialization) and the abstract `generate(self, prompt: str) -> str` method (to call the new provider's API and return the generated text). Handle API errors gracefully.
3.  **Map Provider:** In `modelmatch/models/__init__.py`:
    *   Import your new class (e.g., `from .providers.anthropic import AnthropicModel`).
    *   Add an entry to the `PROVIDER_MAP` dictionary (e.g., `"AnthropicModel": AnthropicModel`).
4.  **Configure Models:** Add entries for the specific models using this provider in `model_config.yaml`, ensuring the `provider` key matches the class name string used in `PROVIDER_MAP`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## 📜 License

This project is licensed under the [MIT License](LICENSE).