import json

# Create a simple math dataset
data = [
    {
        "question": "What is 2 + 2?",
        "answer": "4"
    },
    {
        "question": "What is 3 * 4?",
        "answer": "12"
    },
    {
        "question": "What is 10 - 5?",
        "answer": "5"
    }
]

# Save as JSON (input format)
with open('data/math_problems.json', 'w') as f:
    json.dump(data, f, indent=2) 