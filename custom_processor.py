from promptwizard.glue.promptopt.techniques.common_logic import DatasetSpecificProcessing
import json
import re

class CustomProcessor(DatasetSpecificProcessing):
    def extract_answer_from_output(self, answer):
        """Extract answer from dataset answer field"""
        return answer.strip()

    def extract_final_answer(self, llm_output):
        """Extract final answer from LLM output"""
        pattern = r'<ANS_START>(.*?)<ANS_END>'
        match = re.search(pattern, llm_output)
        if match:
            return match.group(1).strip()
        return llm_output.strip()

    def access_answer(self, llm_output, gt_answer):
        """Compare predicted answer with ground truth"""
        predicted_answer = self.extract_final_answer(llm_output)
        is_correct = False
        if predicted_answer and (predicted_answer.lower() == gt_answer.lower()):
            is_correct = True
        return is_correct, predicted_answer

    def dataset_to_jsonl(self, dataset_path, output_path):
        """Convert dataset to JSONL format"""
        try:
            # Read input data
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            # Convert and write to JSONL
            with open(output_path, 'w') as f:
                for item in data:
                    # Ensure the required fields are present
                    if 'question' in item and 'answer' in item:
                        f.write(json.dumps({
                            'question': item['question'],
                            'answer': item['answer']
                        }) + '\n')
            return True
        except Exception as e:
            print(f"Error converting dataset to JSONL: {str(e)}")
            return False

    def collate_to_str(self, examples, template):
        """Collate examples into a string format"""
        result = ""
        for example in examples:
            result += template.format(
                question=example.get('question', ''),
                answer=example.get('answer', '')
            )
        return result 