import os
from dotenv import load_dotenv
import sys
sys.path.insert(0, "./")  # Add root directory to path
from promptwizard.glue.promptopt.instantiate import GluePromptOpt
from promptwizard.glue.promptopt.techniques.common_logic import GSM8KProcessor

# Load environment variables
load_dotenv()

def optimize_prompts():
    # Configuration paths
    promptopt_config_path = "demos/gsm8k/configs/promptopt_config.yaml"
    setup_config_path = "demos/gsm8k/configs/setup_config.yaml"
    
    # Use GSM8K processor and dataset
    processor = GSM8KProcessor()
    
    # Initialize PromptWizard
    gp = GluePromptOpt(
        promptopt_config_path,
        setup_config_path,
        "train_synthetic.jsonl",  # This will be created by the processor
        processor
    )

    print("Starting prompt optimization...")
    
    # Get optimized prompt
    best_prompt, expert_profile = gp.get_best_prompt(
        use_examples=True,
        run_without_train_examples=False,
        generate_synthetic_examples=True
    )

    print("\nBest Prompt:")
    print("-" * 80)
    print(best_prompt)
    print("-" * 80)
    
    print("\nExpert Profile:")
    print("-" * 80)
    print(expert_profile)
    print("-" * 80)

    return best_prompt, expert_profile

if __name__ == "__main__":
    optimize_prompts() 