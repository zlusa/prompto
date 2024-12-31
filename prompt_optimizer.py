import os
from dotenv import load_dotenv
import json
from promptwizard.glue.promptopt.instantiate import GluePromptOpt
from promptwizard.glue.promptopt.techniques.common_logic import DatasetSpecificProcessing

# Load environment variables
load_dotenv()

class SimpleProcessor(DatasetSpecificProcessing):
    def extract_answer_from_output(self, answer):
        return answer.strip()

    def extract_final_answer(self, llm_output):
        return llm_output.strip()

    def access_answer(self, llm_output, gt_answer):
        predicted = self.extract_final_answer(llm_output)
        is_correct = predicted.lower() == gt_answer.lower()
        return is_correct, predicted

    def dataset_to_jsonl(self, dataset_path, output_path):
        with open(output_path, 'w') as f:
            for item in self.training_data:
                f.write(json.dumps(item) + '\n')
        return True

def generate_examples(task_description):
    """Automatically generate examples based on task description using Gemini"""
    from promptwizard.glue.common.llm.llm_mgr import call_api
    
    # Prompt to generate examples
    messages = [
        {
            "role": "system",
            "content": "You are an expert at creating training examples. Generate diverse and representative examples for the given task."
        },
        {
            "role": "user",
            "content": f"""
            Task Description: {task_description}
            
            Generate 5 different question-answer pairs for this task.
            Format each example as a JSON object with 'question' and 'answer' fields.
            Make the examples diverse and of varying difficulty.
            
            Example format:
            {{"question": "What is 2 + 2?", "answer": "4"}}
            """
        }
    ]
    
    try:
        response = call_api(messages)
        # Extract JSON objects from response
        examples = []
        import re
        json_matches = re.finditer(r'\{[^}]+\}', response)
        for match in json_matches:
            try:
                example = json.loads(match.group())
                if 'question' in example and 'answer' in example:
                    examples.append(example)
            except:
                continue
        
        return examples[:5]  # Return up to 5 examples
    except Exception as e:
        print(f"Error generating examples: {str(e)}")
        # Return some default examples if generation fails
        return [
            {"question": "What is 2 + 2?", "answer": "4"},
            {"question": "What is 10 * 5?", "answer": "50"},
            {"question": "What is 100 - 25?", "answer": "75"}
        ]

def optimize_prompt():
    # Get task description
    print("\n=== Task Configuration ===")
    print("\nExample task descriptions:")
    print("1. You are a math expert who solves arithmetic problems")
    print("2. You are a coding tutor who explains Python concepts")
    print("3. You are a science expert who explains physics concepts")
    print("4. You are a writing assistant who improves text clarity")
    task_description = input("\nEnter task description: ")
    
    # Use fixed base instruction
    base_instruction = "Solve each problem step by step showing your work"
    print(f"\nUsing base instruction: {base_instruction}")
    
    # Generate examples automatically
    print("\nGenerating training examples...")
    examples = generate_examples(task_description)
    print("\nGenerated Examples:")
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Q: {example['question']}")
        print(f"A: {example['answer']}")

    # Create data directory and save examples to JSONL
    os.makedirs("data", exist_ok=True)
    train_file = "data/train.jsonl"
    
    # Save examples to JSONL file
    with open(train_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    # Create configs directory if it doesn't exist
    os.makedirs("configs", exist_ok=True)
    
    # Create config file
    config = {
        "prompt_technique_name": "critique_n_refine",
        "unique_model_id": "gemini-2.0-flash-exp",
        "mutate_refine_iterations": 2,
        "mutation_rounds": 2,
        "refine_instruction": True,
        "refine_task_eg_iterations": 2,
        "style_variation": 2,
        "questions_batch_size": 1,
        "min_correct_count": 1,
        "max_eval_batches": 2,
        "top_n": 1,
        "task_description": task_description,
        "base_instruction": base_instruction,
        "answer_format": "Provide a clear and concise answer.",
        "seen_set_size": len(examples),
        "few_shot_count": min(2, len(examples)),
        "num_train_examples": len(examples),
        "generate_reasoning": True,
        "generate_expert_identity": True,
        "generate_intent_keywords": False
    }
    
    with open("configs/promptopt_config.yaml", "w") as f:
        import yaml
        yaml.dump(config, f)
    
    # Create setup config
    setup_config = {
        "assistant_llm": {
            "prompt_opt": "gemini-2.0-flash-exp"
        },
        "dir_info": {
            "base_dir": "logs",
            "log_dir_name": "glue_logs"
        },
        "experiment_name": "custom_prompt",
        "mode": "offline",
        "description": "Custom prompt optimization"
    }
    
    with open("configs/setup_config.yaml", "w") as f:
        yaml.dump(setup_config, f)

    # Initialize processor with examples
    processor = SimpleProcessor()
    processor.training_data = examples
    
    # Initialize PromptWizard
    gp = GluePromptOpt(
        "configs/promptopt_config.yaml",
        "configs/setup_config.yaml",
        train_file,  # Use the created JSONL file path
        processor
    )

    print("\nStarting prompt optimization...")
    
    try:
        best_prompt, expert_profile = gp.get_best_prompt(
            use_examples=True,
            run_without_train_examples=False,
            generate_synthetic_examples=True
        )

        print("\n=== Results ===")
        print("\nBest Prompt:")
        print("-" * 80)
        print(best_prompt)
        print("-" * 80)
        
        print("\nExpert Profile:")
        print("-" * 80)
        print(expert_profile)
        print("-" * 80)

        # Save results
        with open("optimized_prompt.txt", "w") as f:
            f.write("Best Prompt:\n")
            f.write(best_prompt)
            f.write("\n\nExpert Profile:\n")
            f.write(expert_profile)

        print("\nResults have been saved to 'optimized_prompt.txt'")
        
        return best_prompt, expert_profile

    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        return None, None

if __name__ == "__main__":
    print("Welcome to PromptWizard Optimizer!")
    optimize_prompt() 