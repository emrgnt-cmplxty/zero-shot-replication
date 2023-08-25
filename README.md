# Zero-Shot Replication Framework

## Overview

The Zero-Shot Replication Framework is a minimal environment designed to replicate zero-shot results from past academic papers. It currently supports OpenAI models to generate completions for various datasets and provides tools for handling, evaluating, and storing these completions.

## Features

- Easy configuration of models and parameters.
- Ability to choose datasets to run on.
- Extensibility through a pluggable problem generator.

## Requirements

- Python >= 3.10 and < 3.12
- Poetry for package management

## Min. Dependencies

- anthropic: "0.3.10"
- astunparse: "1.6.3"
- black: ^23.3.0
- evalplus: ^0.1.6
- numpy: "^1.25.2"
- openai: 0.27.8
- pandas: ^2.0.3
- python-dotenv: ^1.0.0
- python-leetcode: "1.2.1"

## Dev Dependencies

- flake8: "6.1.0"
- isort: "5.12.0"
- mypy: "^1.5.1"
- pre-commit: "^3.3.3"
- sourcery: "^1.6.0"
- types-requests: "^2.31.0.2"
- types-attrs: "^19.1.0"
- yapf: "0.40.1"

## Installation

Make sure you have [Poetry](https://python-poetry.org/) installed, then clone the repository and install the dependencies.

```bash
git clone https://github.com/your-username/zero-shot-replication.git
cd zero-shot-replication
poetry install
pre-commit install
cp .env.example .env # Copy the example environment file
# Edit the .env file to add your OpenAI API key, etc.
```

## Usage

You can run the zero-shot replication by executing the `runner.py` file with various command-line arguments.

```bash
poetry run python runner.py --provider openai --dataset human-eval --model gpt-4-0613 --temperature 0.7
```

### Command-Line Arguments

- `--provider`: Which provider to use for zero-shot completions (default: "openai").
- `--dataset`: Which dataset to run on (default: "human-eval").
- `--model`: Model name to load from the provider (default: "gpt-3.5-turbo").
- `--temperature`: Temperature parameter for the provided model (default: 0.7).
- `--output_file_name`: Filename to override the default output file name with.

To see explicit commands ran to generate the reported results, check out the [commands.md](commands.md) menu.

## Results (all models accessed on 08/24)

| Category         | gpt-3.5-turbo-0301 | gpt-3.5-turbo-0613 | Claude 2 | GPT-4-0314 | GPT-4-0613 | GPT-4 Baseline | Sources  |
|------------------|--------------------|--------------------|----------|------------|------------|----------------|----------|
| HumanEval        | 81.7               | XX                 | 65.2     | 87.2       | 84.1       | 67             | [1]      |
| EvalPlus         | 71.3               | XX                 | 54.9     | 79.2       | 74.4       | N/A            |          |
| Leetcode Easy    | XX                 | XX                 | XX       | 91.0       | 88.0       | 72.2-75.6      | [1,2]    |
| Leetcode Medium  | XX                 | XX                 | XX       | 26.0       | 17.0       | 26.2-38.7      | [1,2]    |
| Leetcode Hard    | XX                 | XX                 | XX       | 6.0        | 4.0        | 6.7-7          | [1,2]    |
| GSM8K            | XX                 | XX                 | XX       | X          | X          | 87.1           |          |
| MATH             | XX                 | XX                 | XX       | 49.0       | 46.4       | 42.2           | [3]      |

## License

This project is licensed under the Apache-2.0 License.

## Sources

[1] [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

[2] [Sparks of Artificial General Intelligence](https://arxiv.org/pdf/2303.12712.pdf)

[3] [Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based Self-Verification](https://paperswithcode.com/paper/solving-challenging-math-word-problems-using)