import sys
sys.path.insert(0, "../../")
from dotenv import load_dotenv
import promptwizard
from promptwizard.glue.promptopt.instantiate import GluePromptOpt
from promptwizard.glue.promptopt.techniques.common_logic import DatasetSpecificProcessing

# Load environment variables
load_dotenv()

# Load configuration
promptopt_config_path = "configs/promptopt_config.yaml"
setup_config_path = "configs/setup_config.yaml"

# Initialize the data processor
processor = DatasetSpecificProcessing()

# Initialize the prompt optimizer
gp = GluePromptOpt(
    promptopt_config_path,
    setup_config_path,
    "train_synthetic.jsonl",
    processor
)

# Test a simple math problem
test_problem = {
    "question": "John has 5 apples. He buys 3 more. How many apples does he have now?",
    "answer": "8"
}

# Get prediction
try:
    result = gp.predict(test_problem["question"])
    print(f"Question: {test_problem['question']}")
    print(f"Expected Answer: {test_problem['answer']}")
    print(f"Model Answer: {result}")
except Exception as e:
    print("Error type:", type(e))
    print("Error:", str(e)) 