# Usage Guide for the Stage 1 Judge Environment

This repository is the local evaluation toolkit for the Mathematics Distillation Challenge Stage 1: Equational Theories.
It uses OpenRouter to call remote models and then extracts a final `TRUE` or `FALSE` verdict.

## Environment setup

1. Install Python 3.9 or newer.
2. From the repository root, install the package and development dependencies:

```sh
pip install -e ".[dev]"
```

3. Set your OpenRouter API key in the environment:

```sh
setx OPENROUTER_API_KEY "<your-openrouter-api-key>"
```

For the current session in PowerShell, run:

```powershell
$env:OPENROUTER_API_KEY = "<your-openrouter-api-key>"
```

## How to run the example smoke test

The bundled runner is `examples/run_smoke.py`.

To run the default smoke test against the first 2 example problems:

```sh
python examples/run_smoke.py --limit 2 --models gpt-oss-120b
```

To run a larger smoke test with multiple official models:

```sh
python examples/run_smoke.py --limit 20 --models gpt-oss-120b llama-3-3-70b-instruct
```

## How to run your own prompt

The repository includes `examples/example_complete_prompt.txt`.
You can also provide your own prompt file.

```sh
python examples/run_smoke.py --prompt-file examples/example_complete_prompt.txt --limit 10 --models gpt-oss-120b
```

Model IDs supported by `examples/run_smoke.py` include the official evaluation models such as:

- `gpt-oss-120b`
- `llama-3-3-70b-instruct`
- `gemma-4-31b-it`

These are OpenRouter model aliases configured by `evaluation_models.json`.

## Do I require Ollama?

No.
This repository does not use Ollama.
It is built to call models through OpenRouter using `httpx`.
If you want to run the same evaluation environment, you only need:

- Python 3.9+
- `pip install -e ".[dev]"`
- an `OPENROUTER_API_KEY`

## Running the actual model evaluation flow

The main execution path is:

1. `examples/run_smoke.py` loads a prompt template.
2. `prompt.py` fills the placeholder variables `{{equation1}}` and `{{equation2}}`.
3. `llm.py` sends the request to OpenRouter.
4. `judge.py` extracts and grades the model response into `TRUE` or `FALSE`.

## Verifying the installation

Run the tests:

```sh
pytest tests/
```

## Summary

- This repo already implements the Stage 1 evaluation environment.
- You do not need Ollama.
- Use OpenRouter and the provided `evaluation_models.json`.
- Run `python examples/run_smoke.py` to test the models locally.
