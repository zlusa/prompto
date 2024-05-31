from datasets import load_dataset, concatenate_datasets
import os
import pickle
import re
from tqdm import tqdm
from typing import Any, List
from uuid import uuid4

from glue.common.utils.file import save_jsonlist, yaml_to_class, yaml_to_dict
from glue.promptopt.instantiate import GluePromptOpt
from glue.promptopt.techniques.common_logic import DatasetSpecificProcessing
from paramlogger import ParamLogger

"""
# Description of task. This will be fed to prompt
task_description: "4 choice multiple-choice questions from various branches of knowledge. These questions spans from subjects "
# Base instruction, in line with your dataset. This will be fed to prompt
base_instruction: "Lets think step by step."
# Instruction for specifying answer format
answer_format: "You need to output the correct option [A/B/C/D] for each question with your reasoning."
# `questions_batch_size` examples from training data with replacement.
seen_set_size: 5
# Number of examples to be given for few shots
few_shot_count: 5
"""


iolog = ParamLogger()


class MMLU(DatasetSpecificProcessing):
    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any):
        examples_set = []

        for _, sample in tqdm(enumerate(kwargs["dataset"]), desc="Iterating through samples"):
            question = sample["question"]
            options = sample["choices"]
            subject = sample["subject"]
            option_keys = ["A", "B", "C", "D"]
            options_text = ""
            for key, value in zip(option_keys, options):
                options_text += f"{key}: {value}\n"
            question_n_options = f"{question}\nOptions:\n{options_text}"
            example = {
                DatasetSpecificProcessing.QUESTION_LITERAL: question_n_options,
                DatasetSpecificProcessing.ANSWER_WITH_REASON_LITERAL: DatasetSpecificProcessing.ANSWER_START + \
                                                                      option_keys[sample["answer"]] + \
                                                                      DatasetSpecificProcessing.ANSWER_END,
                DatasetSpecificProcessing.FINAL_ANSWER_LITERAL: option_keys[sample["answer"]],
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


@iolog.log_io_params_for_method
def process_for_task(mmlu_processor: MMLU, task: str):
    dataset = load_dataset("cais/mmlu", task)
    task_dir = os.path.join(current_dir, task)
    os.makedirs(task_dir, exist_ok=True)
    test_file_name = os.path.join(task_dir, "test.jsonl")
    train_file_name = os.path.join(task_dir, "train.jsonl")

    combined_train_set = concatenate_datasets([dataset["dev"], dataset["validation"]])
    print(task)
    print(f"dataset['train'] {combined_train_set}")
    print(f"dataset['test'] {dataset['test']}")

    mmlu_processor.dataset_to_jsonl(train_file_name, dataset=combined_train_set)
    mmlu_processor.dataset_to_jsonl(test_file_name, dataset=dataset["test"])

    gp = GluePromptOpt(llm_config_path=llm_config_path,
                       prompt_config_path=promptopt_config_path,
                       setup_config_path=setup_config_path,
                       dataset_jsonl=train_file_name,
                       data_processor=mmlu_processor)

    original_task_description = gp.prompt_opt_param.task_description

    gp.setup_config.experiment_name = f"mmlu_{task}_{uuid4()}"
    gp.prompt_opt_param.task_description = original_task_description + f"{task.replace('_', ' ')} subject.\n"
    best_prompt, expert_profile = gp.get_best_prompt()
    print(f"Task: {task} \nBest prompt: {best_prompt} \nExpert profile: {expert_profile}")

    accuracy = gp.evaluate(train_file_name)
    print(f"accuracy: {accuracy}")

    data_to_save = {
        "Best prompt": best_prompt,
        "Expert profile": expert_profile,
        "Accuracy": accuracy
    }
    return data_to_save


if __name__ == '__main__':
    current_dir = os.getcwd()
    path_to_config = os.path.join(current_dir, "..", "..", "..", "configs")
    llm_config_path = os.path.join(path_to_config, "llm_config.yaml")
    promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
    setup_config_path = os.path.join(path_to_config, "setup_config.yaml")

    mmlu_processor = MMLU()
    tasks = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    medical_tasks = ['clinical_knowledge', 'college_biology', 'college_medicine', 'anatomy', 'medical_genetics',
                     'professional_medicine']

    results = []
    for task in medical_tasks:
        _ = process_for_task(mmlu_processor, task)
