from datasets import load_dataset
import os
import pickle
from tqdm import tqdm
from typing import Any, List
import re

from glue.common.utils.file import save_jsonlist
from glue.promptopt.instantiate import GluePromptOpt
from glue.promptopt.techniques.common_logic import DatasetSpecificProcessing

"""
# Description of task. This will be fed to prompt
task_description: "4 choice MCQ style of the Medical Licensing Examination questions used to test medical specialist competency in the United States. Questions are in English in the style of the United States
Medical Licensing Exam (USMLE). The dataset is collected from the professional medical board exams.You need to output the correct option [A/B/C/D] for each question using your medical knowledge and reasoning."
# Base instruction, in line with your dataset. This will be fed to prompt
base_instruction: "Lets think step by step to arrive at the answer to this Medical Licensing Examination question"
# Instruction for specifying answer format
answer_format: "You need to output the correct option among [A/B/C/D] for each question separately using your medical knowledge and reasoning."
"""


class MedQA(DatasetSpecificProcessing):
    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any):
        examples_set = []

        for _, sample in tqdm(enumerate(kwargs["dataset"]), desc="Iterating through samples"):
            question = sample["question"]
            options = sample["options"]
            options_text = ""
            for k_v_dict in options:
                options_text += f"{k_v_dict['key']}: {k_v_dict['value']}\n"
            question_n_options = f"{question}\nOptions:\n{options_text}"
            example = {
                DatasetSpecificProcessing.QUESTION_LITERAL: question_n_options,
                DatasetSpecificProcessing.FINAL_ANSWER_LITERAL: sample["answer_idx"]
            }
            examples_set.append(example)

        save_jsonlist(dataset_jsonl, examples_set, "w")

    def extract_final_answer(self, answer: str):
        if not answer:
            return self.INVALID_ANS

        matches = re.search(DatasetSpecificProcessing.ANSWER_DELIMITER_PATTERN, answer)
        if matches:
            answer = matches.group(1)

        answer = answer.strip()

        match = re.search(r'([A-D])', answer)
        if match:
            option = match.group(1)
            return option.strip()

        return self.INVALID_ANS


if __name__ == '__main__':
    current_dir = os.getcwd()
    train_file_name = os.path.join(current_dir, "train.jsonl")
    test_file_name = os.path.join(current_dir, "test.jsonl")
    path_to_config = os.path.join(current_dir, "..", "..", "..", "configs")
    llm_config_path = os.path.join(path_to_config, "llm_config.yaml")
    promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
    setup_config_path = os.path.join(path_to_config, "setup_config.yaml")
    prompt_pool_path = os.path.join(current_dir, "..", "..", "..", "hyperparams", "medqa_prompt_pool.yaml")

    medqa_processor = MedQA()
    dataset = load_dataset("bigbio/med_qa", "med_qa_en_4options_source")

    medqa_processor.dataset_to_jsonl(train_file_name, dataset=dataset["train"])
    medqa_processor.dataset_to_jsonl(test_file_name, dataset=dataset["test"])

    gp = GluePromptOpt(llm_config_path=llm_config_path,
                       prompt_config_path=promptopt_config_path,
                       setup_config_path=setup_config_path,
                       dataset_jsonl=train_file_name,
                       data_processor=medqa_processor,
                       dataset_processor_pkl_path=None,
                       prompt_pool_path=prompt_pool_path)

    best_prompt, expert_profile = gp.get_best_prompt()
    print(f"Best prompt: {best_prompt} \nExpert profile: {expert_profile}")

    accuracy = gp.evaluate(test_file_name)
    print(f"accuracy: {accuracy}")
