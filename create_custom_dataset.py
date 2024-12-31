import json

# Create sample training data
train_data = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris"
    },
    {
        "question": "What is 25 * 4?",
        "answer": "100"
    },
    {
        "question": "Explain what is machine learning in simple terms.",
        "answer": "Machine learning is when computers learn from examples to perform tasks without being explicitly programmed."
    }
]

# Save as JSONL
with open('data/train.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')

# Create test data
test_data = [
    {
        "question": "What is the capital of Italy?",
        "answer": "Rome"
    },
    {
        "question": "What is 15 + 27?",
        "answer": "42"
    }
]

with open('data/test.jsonl', 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n') 