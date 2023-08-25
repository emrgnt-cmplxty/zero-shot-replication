# Commands

## HumanEval

### HEval Generation

```bash
poetry run python zero_shot_replication/runner.py --model=gpt-3.5-turbo-0301 --pset=human-eval

poetry run python zero_shot_replication/runner.py --model=gpt-3.5-turbo-0613 --pset=human-eval

poetry run python zero_shot_replication/runner.py --model=gpt-4-0314 --pset=human-eval

poetry run python zero_shot_replication/runner.py --model=gpt-4-0613 --pset=human-eval

poetry run python zero_shot_replication/runner.py --model=claude-2 --pset=human-eval --provider=anthropic
```

### HEval Evaluation

```bash
poetry run evalplus.evaluate --dataset humaneval --samples=... --parallel 4 --min-time-limit 0.5 --gt-time-limit-factor 5
```

## LeetCode

### LC Generation

```bash

poetry run python zero_shot_replication/runner.py --model=gpt-3.5-turbo-0301 --pset=leetcode

poetry run python zero_shot_replication/runner.py --model=gpt-3.5-turbo-0613 --pset=leetcode

poetry run python zero_shot_replication/runner.py --model=gpt-4-0314 --pset=leetcode

poetry run python zero_shot_replication/runner.py --model=gpt-4-0613 --pset=leetcode

poetry run python zero_shot_replication/runner.py --model=claude-2 --pset=leetcode --provider=anthropic
```

### LC Evaluation

```bash
poetry run python zero_shot_replication/evals/run_leetcode_eval.py --model=...
```

## GMS8K

### GMS8K Generation

```bash
poetry run python zero_shot_replication/runner.py --model=... --pset=gsm8k

```

## MATH

### Generation

```bash
poetry run python zero_shot_replication/runner.py --model=... --pset=math
```
