from datasets import load_dataset
import os
from tqdm import tqdm
from typing import Any

from glue.common.utils.file import save_jsonlist
from glue.promptopt.instantiate import GluePromptOpt
from glue.promptopt.techniques.common_logic import DatasetSpecificProcessing

"""
# Description of task. This will be fed to prompt
task_description: "Identify if the given text contains hate speech or not. Output 1 if the text contains hate speech. Output 0 if text does not contain hatespeech."
# Base instruction, in line with your dataset. This will be fed to prompt
base_instruction: "Lets think step by step to identify if the given text is a hate speech or not."
# Instruction for specifying answer format
answer_format: "You need to answer each of the questions separately with 1 if its a hate speech and 0 if its not a hate speech. "
"""


class EthosProcessor(DatasetSpecificProcessing):

    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any):
        examples_set = []

        for _, sample in tqdm(enumerate(kwargs["dataset"]), desc="Iterating through samples"):
            example = {
              DatasetSpecificProcessing.QUESTION_LITERAL: sample["text"],
              DatasetSpecificProcessing.FINAL_ANSWER_LITERAL: str(sample["label"])
            }
            examples_set.append(example)

        save_jsonlist(dataset_jsonl, examples_set, "w")

    def extract_final_answer(self, answer: str):
        if not answer:
            return self.INVALID_ANS

        answer = answer.strip()
        if self.ANSWER_START in answer:
            answer = answer.split(self.ANSWER_START)[-1].lower()

        valid_responses = ["1", "0"]

        for valid_response in valid_responses:
            if valid_response in answer:
                return valid_response

        return self.INVALID_ANS


if __name__ == '__main__':
    current_dir = os.getcwd()
    train_file_name = os.path.join(current_dir, "train.jsonl")
    test_file_name = os.path.join(current_dir, "test.jsonl")
    run_file = os.path.join(current_dir, "..", "glue", "promptopt", "runner.py")
    path_to_config = os.path.join(current_dir, "..", "..", "..", "configs")
    llm_config_path = os.path.join(path_to_config, "llm_config.yaml")
    promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
    setup_config_path = os.path.join(path_to_config, "setup_config.yaml")
    prompt_pool_path = os.path.join(current_dir, "..", "..", "..", "hyperparams", "ethos_prompt_pool.yaml")
    ethos_processor = EthosProcessor()

    dataset = load_dataset("ethos", "binary")
    dataset = dataset["train"]
    dataset = dataset.train_test_split(test_size=0.80)

    ethos_processor.dataset_to_jsonl(train_file_name, dataset=dataset["train"])
    ethos_processor.dataset_to_jsonl(test_file_name, dataset=dataset["test"])

    gp = GluePromptOpt(llm_config_path=llm_config_path,
                       prompt_config_path=promptopt_config_path,
                       setup_config_path=setup_config_path,
                       dataset_jsonl=train_file_name,
                       data_processor=ethos_processor,
                       dataset_processor_pkl_path=None,
                       prompt_pool_path=prompt_pool_path)

    best_prompt, expert_profile = gp.get_best_prompt()
    print(f"Best prompt: {best_prompt} \nExpert profile: {expert_profile}")

    accuracy = gp.evaluate(test_file_name)
    print(f"accuracy: {accuracy}")

