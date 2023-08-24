"""Generates problems to be run in the runner."""
from typing import Any, Generator, Tuple

from evalplus.data import get_human_eval_plus

<<<<<<< HEAD:zero_shot_replication/generators.py
from zero_shot_replication.base import ProblemType
import json
=======
from zero_shot_replication.helpers.base import ProblemType

>>>>>>> 6240d63f5f1bd806b207fcf2d6adf4b37dca2e43:zero_shot_replication/helpers/generators.py

class ProblemGenerator:
    """A class for generating problems for the runner."""

    def __init__(self, problem_type: ProblemType) -> None:
        self.problem_type = problem_type
        # use for big files with no task_id
        self.task_count = 0

    @property
    def generator(self) -> Generator[Tuple[str, Any], None, None]:
        """
        Get a generator over the given problems

        Returns events of the form should be of the form:
            Generator[[task_id: str, problem: dict], None None]

        """
        match self.problem_type:
            case ProblemType.HUMAN_EVAL:
                #  Fields on the yielded problem are ['task_id', 'prompt', 'entry_point', 'canonical_solution', 'test', 'contract', 'base_input', 'atol', 'plus_input']
                yield from get_human_eval_plus().items()
            case ProblemType.GSM8K:
                #  Fields on the yielded problem are ['question', 'answer']
                filename = 'datasets/inputs/GSM8K/all.jsonl'
                with open(filename, 'r', encoding='utf-8') as file:
                    for line in file:
                        try:
                            self.task_count += 1
                            yield (str(self.task_count), json.loads(line))
                        except:
                            continue
            case _:
                raise NotImplementedError(f"Problem type not implemented for {self.problem_type}.")