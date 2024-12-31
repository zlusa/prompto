import os
from dotenv import load_dotenv
from promptwizard.glue.common.llm.llm_mgr import call_api
from promptwizard.glue.common.exceptions import GlueLLMException

# Load environment variables
load_dotenv()

def test_model():
    test_cases = [
        # Math and reasoning
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a math expert who explains step by step."
                },
                {
                    "role": "user",
                    "content": "If a train travels at 60 mph for 2.5 hours, then at 75 mph for 1.5 hours, what's the total distance?"
                }
            ],
            "description": "Complex math with reasoning"
        },
        
        # Multi-turn conversation
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful coding assistant."
                },
                {
                    "role": "user",
                    "content": "Write a Python function to check if a string is a palindrome."
                },
                {
                    "role": "assistant",
                    "content": "Here's a simple function to check palindromes:\n\ndef is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]"
                },
                {
                    "role": "user",
                    "content": "Can you modify it to ignore spaces and punctuation?"
                }
            ],
            "description": "Multi-turn coding conversation"
        },
        
        # Long context
        {
            "messages": [
                {
                    "role": "user",
                    "content": """Analyze this paragraph and summarize the main points:
                    The development of artificial intelligence has seen remarkable progress in recent years. 
                    Large language models have demonstrated unprecedented capabilities in understanding and generating human-like text. 
                    However, these advances come with challenges, including ethical considerations, bias in training data, 
                    and the need for responsible AI development. Researchers are actively working on making AI systems more 
                    transparent, accountable, and aligned with human values. The future of AI depends on balancing innovation 
                    with safety and ethical considerations."""
                }
            ],
            "description": "Long context analysis"
        },
        
        # Edge case - very short input
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Hi!"
                }
            ],
            "description": "Very short input"
        },
        
        # System prompt with specific formatting
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a JSON formatter. Always respond in valid JSON format."
                },
                {
                    "role": "user",
                    "content": "Create a JSON object with name: John Doe, age: 30, city: New York"
                }
            ],
            "description": "Structured output (JSON)"
        },
        
        # Testing error handling
        {
            "messages": [
                {
                    "role": "user",
                    "content": " " * 10000  # Very long empty content
                }
            ],
            "description": "Error handling - long empty content"
        }
    ]

    for test in test_cases:
        print(f"\nTesting: {test['description']}")
        print("-" * 80)
        try:
            response = call_api(test["messages"])
            print(f"Response:\n{response}")
        except GlueLLMException as e:
            print(f"LLM Error: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
        print("-" * 80)

def test_performance():
    """Test response time and consistency"""
    import time
    
    test_message = {
        "messages": [
            {
                "role": "user",
                "content": "What is 123 + 456?"
            }
        ]
    }
    
    num_trials = 5
    times = []
    
    print("\nTesting Performance")
    print("-" * 80)
    
    for i in range(num_trials):
        start_time = time.time()
        try:
            response = call_api(test_message["messages"])
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            print(f"Trial {i+1}:")
            print(f"Response: {response}")
            print(f"Time taken: {elapsed:.2f} seconds")
            print("-" * 40)
        except Exception as e:
            print(f"Error in trial {i+1}: {str(e)}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\nAverage response time: {avg_time:.2f} seconds")
    print("-" * 80)

if __name__ == "__main__":
    print("Starting comprehensive tests...")
    test_model()
    print("\nStarting performance tests...")
    test_performance() 