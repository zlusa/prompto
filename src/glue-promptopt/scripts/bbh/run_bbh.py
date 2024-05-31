import os
import re
from typing import Any, Dict, List
from uuid import uuid4

from datasets import load_dataset
from glue.common.utils.file import read_jsonl_row, save_jsonlist, yaml_to_dict
from tqdm import tqdm

from glue.promptopt.instantiate import GluePromptOpt
from glue.promptopt.techniques.common_logic import DatasetSpecificProcessing
from paramlogger import ParamLogger

"""
few_shot_count: 3
Task description for each task defined in bbh_task_description.jsonl
answer_format: "For each question present the reasoning followed correct option out of [{options}]."
"""

iolog = ParamLogger()


class BBH(DatasetSpecificProcessing):

    def __init__(self, option_type):
        super().__init__()
        self.options = option_type
        verbal_options = {"Yes,No", "True,False", "Valid,Invalid"}
        self.option_class = None
        if option_type in verbal_options:
            self.option_class = "verbal"

    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any):
        examples_set = []
        option_string = ""

        if self.option_class == "verbal":
            option_string = f"Options:\n "
            option_list = self.options.split(",")
            for i, opt in enumerate(option_list):
                option_string += f"({chr(ord('A') + i)}) {option_list[i]} "

        for _, sample in tqdm(enumerate(kwargs["dataset"]), desc="Iterating through samples"):
            question = sample["input"]
            if self.option_class == "verbal":
                if "Options:" in question:
                    question = question[:question.rfind("Options:")].strip()

            # TODO: when answer has parenthesis ?
            sample["target"] = sample["target"].replace("(", "").replace(")", "")

            question_n_options = f"{question}\n{option_string}"
            example = {
                DatasetSpecificProcessing.QUESTION_LITERAL: question_n_options,
                DatasetSpecificProcessing.FINAL_ANSWER_LITERAL: sample["target"]
            }
            examples_set.append(example)

        save_jsonlist(dataset_jsonl, examples_set, "w")

    def _extract_mcq_answer(self, answer: str):
        match = re.search(r"\(([" + self.options + "])\)", answer)
        if match:
            option = match.group(1)
            return option.strip()

        match = re.search(r"([" + self.options + "])", answer)
        if match:
            option = match.group(1)
            return option.strip()

        return self.INVALID_ANS

    def _extract_verbal_answer(self, answer: str):
        option_list = self.options.split(",")
        for option in option_list:
            if option in answer:
                return option

    def extract_final_answer(self, answer: str):
        if not answer:
            return self.INVALID_ANS

        matches = re.search(DatasetSpecificProcessing.ANSWER_DELIMITER_PATTERN, answer)
        if matches:
            answer = matches.group(1)

        answer = answer.strip()

        if self.option_class == "verbal":
            return self._extract_verbal_answer(answer)
        elif self.options == "string":
            return answer

        return self._extract_mcq_answer(answer)


@iolog.log_io_params_for_method
def process_for_task(task: Dict, common_args: Dict):
    dataset = load_dataset("lukaemon/bbh", task["task_id"])
    task_dir = os.path.join(current_dir, task["task_id"])
    os.makedirs(task_dir, exist_ok=True)
    test_file_name = os.path.join(task_dir, "test.jsonl")
    train_file_name = os.path.join(task_dir, "train.jsonl")

    # Split the dataset into train and test
    total_examples = len(dataset["test"])
    test_size = (total_examples - common_args["seen_set_size"]) / total_examples

    train_set, test_set = dataset["test"].train_test_split(test_size=test_size).values()

    bbh_processor = BBH(task["answer_type"])
    bbh_processor.dataset_to_jsonl(train_file_name, dataset=train_set, task=task)
    bbh_processor.dataset_to_jsonl(test_file_name, dataset=test_set, task=task)

    gp = GluePromptOpt(llm_config_path,
                       promptopt_config_path,
                       setup_config_path,
                       train_file_name,
                       bbh_processor)

    gp.setup_config.experiment_name = f"bbh_{task['task_id']}_{uuid4()}"
    gp.prompt_opt_param.task_description = task["task_description"]
    gp.prompt_opt_param.base_instruction = gp.prompt_opt_param.base_instruction.format(task_name=task["task_name"])
    # TODO check when answer_type=string
    gp.prompt_opt_param.answer_format = gp.prompt_opt_param.answer_format.format(options=task["answer_type"])
    best_prompt, expert_profile = gp.get_best_prompt()
    print(f"Best prompt: {best_prompt} \nExpert profile: {expert_profile}")

    accuracy = gp.evaluate(test_file_name)
    print(f"accuracy: {accuracy}")

    data_to_save = {
        "Best prompt": best_prompt,
        "Expert profile": expert_profile,
        "Accuracy": accuracy
    }
    return data_to_save


if __name__ == '__main__':
    current_dir = os.getcwd()
    path_to_config = os.path.join(current_dir, "..", "..", "..", "..", "configs")
    llm_config_path = os.path.join(path_to_config, "llm_config.yaml")
    promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
    setup_config_path = os.path.join(path_to_config, "setup_config.yaml")

    common_args = yaml_to_dict(promptopt_config_path)
    #for task in read_jsonl_row("bbh_task_description.jsonl"):
    task={"task_id": "dyck_languages", "task_name": "Dyck Languages", "task_description": "Predict the sequence of the closing parentheses of a Dyck-4 word without its last few closing parentheses.", "answer_type": "string"}
    process_for_task(task, common_args)
