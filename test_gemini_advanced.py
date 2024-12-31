import os
from dotenv import load_dotenv
from promptwizard.glue.common.llm.llm_mgr import call_api

# Load environment variables
load_dotenv()

def test_different_prompts():
    test_cases = [
        # Basic math
        {
            "messages": [
                {"role": "user", "content": "What is 25 * 4?"}
            ],
            "description": "Basic math"
        },
        
        # Multi-turn conversation
        {
            "messages": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is 5 + 3?"},
                {"role": "assistant", "content": "That would be 8."},
                {"role": "user", "content": "And if I add 2 more?"}
            ],
            "description": "Multi-turn conversation"
        },
        
        # Long context
        {
            "messages": [
                {"role": "user", "content": "Solve this word problem: If a train travels at 60 mph for 2.5 hours, then increases speed to 75 mph for 1.5 hours, what is the total distance covered?"}
            ],
            "description": "Long context problem"
        }
    ]
    
    for test in test_cases:
        print(f"\nTesting: {test['description']}")
        print("-" * 50)
        try:
            response = call_api(test["messages"])
            print(f"Response: {response}")
        except Exception as e:
            print("Error type:", type(e))
            print("Error:", str(e))
        print("-" * 50)

if __name__ == "__main__":
    test_different_prompts() 