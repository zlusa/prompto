import os
from dotenv import load_dotenv
from promptwizard.glue.common.llm.llm_mgr import call_api
from promptwizard.glue.common.exceptions import GlueLLMException

# Load environment variables
load_dotenv()

def test_model():
    test_cases = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful math expert."
                },
                {
                    "role": "user",
                    "content": "What is 15 + 27?"
                }
            ],
            "description": "Basic math"
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write a short poem about AI"
                }
            ],
            "description": "Creative writing"
        }
    ]

    for test in test_cases:
        print(f"\nTesting: {test['description']}")
        print("-" * 50)
        try:
            response = call_api(test["messages"])
            print(f"Response: {response}")
        except GlueLLMException as e:
            print(f"LLM Error: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
        print("-" * 50)

if __name__ == "__main__":
    test_model() 