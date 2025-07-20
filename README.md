# llm-call

Prototype demonstrating a minimal multi-agent system using the OpenAI API. Two or more agents cooperate by exchanging messages through the API.

## Requirements

Install dependencies (including `openai`) via `pip`:

```bash
pip install -r requirements.txt
```

Set your `OPENAI_API_KEY` environment variable before running the example.

## Running the main script

Execute the main module to see a short chat between two agents:

```bash
python -m llm_call.main
```

If the `openai` package is not installed, or the API key is missing, the script will instruct you accordingly.

## Running tests

Run the test suite with:

```bash
pytest
```
