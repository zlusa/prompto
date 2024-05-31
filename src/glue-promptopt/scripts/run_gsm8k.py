from datasets import load_dataset
from re import compile, findall
import os
import subprocess
from tqdm import tqdm
from typing import Any

from glue.common.utils.file import save_jsonlist
from glue.promptopt.instantiate import GluePromptOpt
from glue.promptopt.techniques.common_logic import DatasetSpecificProcessing

"""
# Description of task. This will be fed to prompt
task_description: "Provide question answering on mathematical school grade questions that require multi-step reasoning. The problems should take between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ - / *) to reach the final answer."
# Base instruction, in line with your dataset. This will be fed to prompt
base_instruction: "Lets think step by step to arrive at the solution of this mathematical problem"
answer_format: "You need to answer each of the questions separately. Arabic numeral should be at the end in the format"
"""


class GSM8k(DatasetSpecificProcessing):

    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any) -> None:
        def extract_answer_from_output(completion):
            # Your functions for metrics and prompt building
            ans_re = compile(r"#### (\-?[0-9\.\,]+)")
            self.INVALID_ANS = "[invalid]"

            match = ans_re.search(completion)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                return match_str
            else:
                return self.INVALID_ANS

        examples_set = []

        for _, sample in tqdm(enumerate(kwargs["dataset"]), desc="Evaluating samples"):
            example = {
              DatasetSpecificProcessing.QUESTION_LITERAL: sample['question'],
              DatasetSpecificProcessing.ANSWER_WITH_REASON_LITERAL: sample['answer'],
              DatasetSpecificProcessing.FINAL_ANSWER_LITERAL: extract_answer_from_output(sample["answer"])
            }
            examples_set.append(example)

        save_jsonlist(dataset_jsonl, examples_set, "w")

    def extract_final_answer(self, answer: str):
        if not answer:
            return self.INVALID_ANS

        model_pred = answer.lower()
        preds = model_pred.split(self.ANSWER_START.lower())
        answer_flag = True if len(preds) > 1 else False

        pred = preds[-1].replace(",", "")
        pred = [s for s in findall(r'-?\d+\.?\d*', pred)]

        if len(pred) == 0:
            return self.INVALID_ANS

        if answer_flag:
            # choose the first element in list
            pred = pred[0]
        else:
            # choose the last element in list
            pred = pred[-1]

        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred[-1] == ".":
            pred = pred[:-1]

        return pred


if __name__ == '__main__':
    current_dir = os.getcwd()
    train_file_name = os.path.join(current_dir, "train.jsonl")
    test_file_name = os.path.join(current_dir, "test.jsonl")
    run_file = os.path.join(current_dir, "..", "glue", "promptopt", "runner.py")
    path_to_config = os.path.join(current_dir, "..", "..", "..", "configs")
    llm_config_path = os.path.join(path_to_config, "llm_config.yaml")
    promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
    setup_config_path = os.path.join(path_to_config, "setup_config.yaml")

    gsm8k_processor = GSM8k()

    dataset = load_dataset("gsm8k", "main")
    
    gsm8k_processor.dataset_to_jsonl(train_file_name, dataset=dataset["train"])
    gsm8k_processor.dataset_to_jsonl(test_file_name, dataset=dataset["test"])

    gp = GluePromptOpt(llm_config_path,
                       promptopt_config_path,
                       setup_config_path,
                       train_file_name,
                       gsm8k_processor)

    best_prompt, expert_profile = gp.get_best_prompt()
    print(f"Best prompt: {best_prompt} \nExpert profile: {expert_profile}")

    accuracy = gp.evaluate(test_file_name)
    print(f"accuracy: {accuracy}")
