import argparse
import logging
import os
from time import sleep
import re

import pandas as pd
from evalplus.data import write_jsonl
from leetcode_env.environment import LeetCodeEnv
from leetcode_env.leetcode_types import LeetCodeSubmission, ProgrammingLanguage
from pandas.core.array_algos.take import take_1d

from zero_shot_replication.helpers.base import OUTPUT_FILE_NAME
from zero_shot_replication.helpers.utils import (
    get_configured_logger,
    get_root_dir,
    load_file_or_raise,
    parse_arguments,
    prep_for_file_path,
    load_existing_jsonl
)

def read_existing_results(out_path: str) -> list[dict]:
    """Reads existing results from out_path if it exists, otherwise returns empty list"""
    return (
        load_file_or_raise(out_path).to_dict(orient="records")
        if os.path.exists(out_path)
        else []
    )


def _create_submission_result(
    solution: dict, extracted_code: str, status: str, reward: bool, done: str
) -> dict:
    return {
        "task_id": solution["task_id"],
        "status": status,
        "reward": int(reward),
        "done": done,
        "raw_completion": solution["raw_completion"],
        "extracted_code": extracted_code,
        "difficulty": solution["difficulty"],
    }


def process_solution(
    solution: dict,
    logger: logging.Logger,
) -> int:
    task_id = solution.get('task_id')
    answer = solution.get('answer', 'no answer')
    completion = solution.get('completion', 'model failed to complete')
    task_id = solution.get('task_id')
    re_match = re.search(r'#### (\d+(\.\d+)?)', answer)
    if re_match:
        canonical_answer = re_match.group(1)
    else:
        logger.warn(f"Cannot find canonical answer for task_id: {task_id}.")
        return 0
    box_match = re.search(r'\\boxed{(\d+(\.\d+)?)\}', completion)
    if box_match:
        model_answer = box_match.group(1)
    else:
        logger.warn(f"Cannot find model answer for task_id: {task_id}.")
        return 0
    return int(canonical_answer == model_answer) 


def get_input_path(args: argparse.Namespace) -> str:
    """Get the output path for the given arguments."""
    input_dir = os.path.join(
        get_root_dir(),
        "results",
        prep_for_file_path(args.provider),
        prep_for_file_path(args.pset),
        prep_for_file_path(args.model),
    )

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    return os.path.join(
        input_dir,
        args.input_file_name
        or OUTPUT_FILE_NAME.format(
            PROVIDER=prep_for_file_path(args.provider),
            pset=prep_for_file_path(args.pset),
            MODEL=prep_for_file_path(args.model),
            TEMPERATURE=prep_for_file_path(str(args.temperature)),
        ),
    )

if __name__ == "__main__":
    args = parse_arguments()
    args.pset = "gsm8k"
    results_input_path = get_input_path(args)

    logger = get_configured_logger(__name__, "DEBUG")
    logger.info(f"Loading solutions from {results_input_path}")
    results = load_existing_jsonl(results_input_path)
    output_path = results_input_path.replace(".jsonl", "_eval_results.jsonl")
    logger.info("Checking the answers, this should be very fast")
    score = [process_solution(result, logger) for result in results]
    avg_score = sum(score)/len(score)
    logger.info(f"Saving results to {output_path}")
    scores = load_existing_jsonl(output_path)
    scores.append(
        {
            "provider": args.provider,
            "pset": args.pset,
            "model": args.model,
            "temperature": args.temperature,
            "score": avg_score
        }
    )
    write_jsonl(output_path, scores)
