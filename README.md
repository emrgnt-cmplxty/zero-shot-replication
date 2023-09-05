# Zero-Shot Replication Framework

## Overview

The Zero-Shot Replication Framework is a tool designed to replicate zero-shot results from recent academic papers or model reports. Additionally, it aims to extend evaluations to better understand the strengths and weaknesses of various approaches. The framework currently supports OpenAI, Anthropic, and HuggingFace models.

## Features

- Simple model and parameter configuration.
- Choice of datasets for evaluation.
- Extensibility through a modular provider / model / dataset setup.

## pass@1 results (all proprietary models accessed on 08/24-08/25, 2023)

To better understand these results, please check the notes below

### Proprietary Models

| Category             | gpt-3.5-turbo-0301 | gpt-3.5-turbo-0613 | claude-2 | gpt-4-0314 | gpt-4-0613| gpt-4 Baseline | Sources  |
|----------------------|--------------------|--------------------|----------|------------|-----------|----------------|----------|
| *Standard Bench*     |                    |                    |          |            |           |                |          |
| HumanEval            | 67.0               | 61.5               | 65.2     | 86.0       | 84.1      | 67.0           | [1]      |
| HumanEval+           | 59.1               | 54.2               | 54.9     | 80.5       | 74.4      | N/A            |          |
| MATH                 | 35.4               | 37.2               | 17.6     | 51.6       | 50.3      | 42.2           | [3]      |
| **LeetCodeSparks**   |                    |                    |          |            |           |                | [1,2]    |
| Easy                 | 60.0               | 76.2               | 52.4     | 76.2       | 61.2      | 68.2-75.6      | [1,2]*   |
| Medium               | 15.0               | 22.0               | 9.8      | 19.5       | 31.7      | 26.7-40.0      | [1,2]*   |
| Hard                 | 0.0                | 0.0                | 0.0      | 4.6        | 13.6      | 6.6-10.7       | [1,2]*   |
| **LeetCode100**      |                    |                    |          |            |           |                |          |
| Easy                 | 83.0               | 80.0               | 73.0     | 91.0       | 88.0      | N/A            |          |
| Medium               | 16.0               | 16.0               | 16.0     | 26.0       | 21.0      | N/A            |          |
| Hard                 | 1.0                | 3.0                | 2.0      | 6.0        | 6.0       | N/A            |          |

### OpenSource Models (vs latest GPT-4)

| Category             | code-llama-34b | wizard-coder-34b | phind-v2-34b |
|----------------------|----------------|------------------|--------------|
| *Standard Bench*     |                |                  |              |
| HumanEval            | 56.7           | 69.5             | 75.0         |
| HumanEval+           | 48.2           | 60.3             | 70.1         |
| **LeetCodeSparks**   |                |                  |              |
| Easy                 | 33.3           | 42.9             | 52.4         |
| Medium               | 2.4            | 12.2             | 7.3          |
| Hard                 | 0.0            | 0.0              | 0.0          |
| **LeetCode100**      |                |                  |              |
| Easy                 | 53.0           | 68.0             | 63.0         |
| Medium               | 3.0            | 9.0              | 5.0          |
| Hard                 | 0.0            | 0.0              | 3.0          |


### Notes on Results

- Our modified prompting for HumanEval may differ from other benchmarks.
- The GPT-4 LeetCodeSparks baseline is approximate. We don't have a precise list of LeetCode problems from the referenced reports.
- We define 'LeetCodeSparks' as the 84 problems used for the human evaluation measurement mentioned in [2].
- 'LeetCode_100' is our out-of-sample dataset, introducing 100 recent easy, medium, and hard LeetCode problems ranging from 2554-2818.

## Installation

```bash
# Repository setup
git clone https://github.com/your-username/zero-shot-replication.git
cd zero-shot-replication
git submodule update --init --recursive
# Install dependencies
poetry install
```

### Optional Dependencies

- `vllm_support`: For VLLM functionalities, required for `WizardCoder` model.
- `automata`: For `automata` agent evaluations.
- `python-leetcode`: For `leetcode` evaluations.
- `evalplus`: For `HumanEval` and `HumanEval+` evaluations.
- `quantized_support`: For running 4 or 8 bit models.

### Possible Weirdness

I sometimes see that setting `torch==2.0.1` results in issues with the cuda environment initialization on my remote machine. One workaround was to first install `torch=2.0.0`, which requires commenting out of vllm, and to then increment the torch version and uncoment vllm. This may solve some user issues.

---

## Requirements

- Python >= 3.11 and < 3.12
- Poetry for package management

### Optional Feature Requirements

For additional features, you can install the optional dependencies:

```bash
poetry install -E <extra_name>
```

- **WizardCode Model Gen.**: `vllm_support`
- **Phind Model Gen.**: `transformers must be installed from git (currently by hand)`
- **Automata Agent Gen.**: `automata`
- **Leetcode Evaluation**: `python-leetcode`

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

## License

This project is licensed under the Apache-2.0 License.

## Sources

[1] [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

[2] [Sparks of Artificial General Intelligence](https://arxiv.org/pdf/2303.12712.pdf)

[3] [Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based Self-Verification](https://paperswithcode.com/paper/solving-challenging-math-word-problems-using)
