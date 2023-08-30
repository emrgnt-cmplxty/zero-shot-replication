import json
import os

import pandas as pd
from evalplus.data import write_jsonl

from zero_shot_replication.core.utils import extract_code


def load_file_or_raise(path: str):
    """Utility function to load a file or raise an error if not found."""
    try:
        file_extension = os.path.splitext(path)[-1].lower()
        if file_extension == ".csv":
            return pd.read_csv(path)
        elif file_extension == ".jsonl":
            with open(path, "r", encoding="utf-8") as file:
                return pd.DataFrame(
                    json.loads(line) for line in file if line.strip()
                )
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Please check the expected data at {path}."
        ) from e


df = load_file_or_raise(
    "hugging_face_human_eval__model_eq_WizardLM_WizardCoder_Python_34B_V1p0__temperature_eq_0p2__quant_eq_float16.jsonl"
    # hugging_face_human_eval__model_eq_WizardLM_WizardCoder_Python_34B_V1p0__temperature_eq_0p2__quant_eq_float16.jsonl"
)

results = []
for ind, row in df.iterrows():
    row = row.to_dict()
    # print("raw completion = ", row["raw_completion"] )
    # try:
    #     if "### Response" in row["raw_completion"]:
    #         row["completion"] = extract_code(
    #             row["raw_completion"].split("### Response:")[1]
    #         )
    #     else:
    row["raw_completion_temp"] = row["raw_completion"].replace("<s>", "")
    row["raw_completion_temp"] = row["raw_completion"].replace("<//s>", "")
    row["raw_completion_temp"] = row["raw_completion"].replace("</s>", "")
    row["raw_completion_temp"] = row["raw_completion_temp"].strip()
    # row["raw_completion_temp"] = row["raw_completion_temp"].split("\n\n\n")[0]
    if "# Test" in row["raw_completion_temp"]:
        row["raw_completion_temp"] = extract_code(
            row["raw_completion_temp"].split("# Test")[0]
        )

    # print(f"completion:\n{row['raw_completion_temp']}\n\n")

    row["completion"] = extract_code(row["raw_completion_temp"])

    print("-" * 100)
    print(f"\task_id:\n{row['task_id']}\n")
    print(f"\nextracted completion:\n{row['completion']}\n\n")
    print("-" * 100)

    # except Exception as e:
    #     print(f"failing..., raw_completion={row['raw_completion']}")
    # row["completion"] = row["completion"].replace('\t', '    ')
    # row["completion"] = row["completion"].split('#')[0] #replace('\t', '    ')

    results.append(row)

write_jsonl("test.jsonl", results)
# df['completion'] = [extract_code(ele) for ele in df['raw_completion']]
# print('df = ', df)
# import pdb; pdb.set_trace()
