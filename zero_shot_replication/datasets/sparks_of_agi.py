import os
import textwrap
from typing import Any, Generator, List, Tuple

from zero_shot_replication.core import BaseDataset, ProblemType
from zero_shot_replication.core.utils import (
    get_pset_inputs_dir,
    load_file_or_raise,
)
from zero_shot_replication.model.base import PromptMode


class SparksOfAGIDataset(BaseDataset):
    """A concrete class to provide Sparks Of AGI problems for the runner."""

    INPUT_FILE = "all.jsonl"

    SPARKS_TEMPLATE = textwrap.dedent(
        """
        {TASK_PROMPT}
        {QUESTION}
        """
    )

    @property
    def raw_prompt(self) -> str:
        """Concrete method to get the raw prompt for MATH problems."""
        return SparksOfAGIDataset.SPARKS_TEMPLATE

    @property
    def input_paths(self) -> List[str]:
        """Concrete method to get a list over the Sparks Of AGI dataset paths."""
        return [
            os.path.join(
                get_pset_inputs_dir(),
                ProblemType.MSFT_SPARKS_AGI.value.upper(),
                SparksOfAGIDataset.INPUT_FILE,
            )
        ]
    
    @property
    def generator(self) -> Generator[Tuple[str, Any], None, None]:
        """Concrete method to get a generator over the MATH problems."""
        # Load the dataset using the utility function
        problems = load_file_or_raise(self.input_paths[0])

        # Iterate over each row in the dataframe
        for index, problem in problems.iterrows():
            # Convert the row to a dictionary and yield
            yield f"sparks_of_agi/{int(index)}", problem.to_dict()

    def get_formatted_prompt(
        self,
        problem: dict,
        prompt_mode: PromptMode = PromptMode.HUMAN_FEEDBACK,
    ) -> str:
        """Concrete method to get the formatted prompt for MATH problems."""
        return self.raw_prompt.format(TASK_PROMPT="Please answer the following:", QUESTION=problem.get("question"))
