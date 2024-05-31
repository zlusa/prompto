from datasets import load_dataset, concatenate_datasets
import math
import os
import pickle
import random
import re
from tqdm import tqdm
from typing import Any

from glue.common.utils.file import save_jsonlist
from glue.promptopt.instantiate import GluePromptOpt
from glue.promptopt.techniques.common_logic import DatasetSpecificProcessing


"""
Params for PubMedQA dataset

# Description of task. This will be fed to prompt
task_description: "Answer biomedical research questions with yes/no/maybe using the corresponding abstracts."
# Base instruction, in line with your dataset. This will be fed to prompt
base_instruction: "Lets think step by step to arrive at the answer to this biomedical research question."
# Instruction for specifying answer format
answer_format: "You need to answer each of the questions separately with yes/ no/ maybe."
"""


class PubMed(DatasetSpecificProcessing):
    possible_answers_list = ["yes", "no", "maybe"]

    def split_test_hf(dataset, fold):
        """
        dataset: Hugging Face dataset object
        fold: number of splits

        output list of split datasets

        Split the dataset for each label to ensure label proportion of different subsets are similar
        based on the official implementation of split in PubMedQA
        https://github.com/pubmedqa/pubmedqa/blob/master/preprocess/split_dataset.py
        """

        def split_label(dataset, fold):
            """
            Splits the dataset into fold parts with equal size as much as possible.
            """

            return [dataset.shard(num_shards=fold, index=i) for i in range(fold)]

        output = []
        for decision in ['yes', 'no', 'maybe']:
            filtered_dataset = dataset.filter(lambda example: example['final_decision'] == decision)
            output.extend(split_label(filtered_dataset, fold))

        balanced_output = []
        for i in range(fold):
            combined_split = None
            for j in range(0, len(output), fold):
                if combined_split is None:
                    combined_split = output[j + i]
                else:
                    combined_split = concatenate_datasets([combined_split, output[j + i]])
            balanced_output.append(combined_split)

        # Balancing the last split
        if len(balanced_output[-1]) != len(balanced_output[0]):
            for i in range(fold - 1):
                extra_indices = balanced_output[i].shuffle(seed=random.randint(0, 10000)).select(range(1)).indices
                balanced_output[-1] = concatenate_datasets([balanced_output[-1], balanced_output[i].select(extra_indices)])
                balanced_output[i] = balanced_output[i].select(range(1, len(balanced_output[i])))

        return balanced_output

    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any):
        examples_set = []

        for _, sample in tqdm(enumerate(kwargs["dataset"]), desc="Iterating through samples"):
            context = " ".join(sample["context"]["contexts"])
            question_n_context = sample["question"] + "\n[Abstract] :" + context
            example = {
              DatasetSpecificProcessing.QUESTION_LITERAL: question_n_context,
              DatasetSpecificProcessing.ANSWER_WITH_REASON_LITERAL: sample["long_answer"],
              DatasetSpecificProcessing.FINAL_ANSWER_LITERAL: sample["final_decision"],
            }
            examples_set.append(example)

        save_jsonlist(dataset_jsonl, examples_set, "w")

    def extract_final_answer(self, answer: str):
        if not answer:
            return self.INVALID_ANS

        matches = re.search(DatasetSpecificProcessing.ANSWER_DELIMITER_PATTERN, answer)
        if matches:
            answer = matches.group(1)

        answer = answer.strip().lower()

        for ans in self.possible_answers_list:
            if ans in answer:
                return ans

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

    pubmed_processor = PubMed()
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    # PubMed doesn't have test dataset split, hence we split training dataset
    dataset = dataset["train"]

    train_set, test_set = PubMed.split_test_hf(dataset, 2)

    pubmed_processor.dataset_to_jsonl(train_file_name, dataset=train_set)
    pubmed_processor.dataset_to_jsonl(test_file_name, dataset=test_set)

    gp = GluePromptOpt(llm_config_path=llm_config_path,
                       prompt_config_path=promptopt_config_path,
                       setup_config_path=setup_config_path,
                       dataset_jsonl=train_file_name,
                       data_processor=pubmed_processor)

    best_prompt, expert_profile = gp.get_best_prompt()
    print(f"Best prompt: {best_prompt} \nExpert profile: {expert_profile}")

    accuracy = gp.evaluate(test_file_name)
    print(f"accuracy: {accuracy}")
