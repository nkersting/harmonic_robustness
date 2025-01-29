#!/uAsr/bin/pythonAA

import time
import json
import requests
import os
from functools import partial
from LLM_model_tester import LLM_model_tester


class DeepSeekLocalTester(LLM_model_tester):
    def __init__(self, model_name, radius=0, ord_limit=31, ord_size=3, temperature=0):
        """
        Assumes Ollama installed and running on localhost
        """
        super().__init__(partial(self.DeepSeekLocal_submitter, model_name), radius, ord_limit, ord_size)
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY")

            
    def DeepSeekLocal_submitter(self, model, question):
        def clean_response(text):
            text = text.replace('<think>', '').replace('</think>', '').strip()
            return text

        url = "http://localhost:11434/api/generate"
        data = {"model": model, "prompt": question}

        response = requests.post(url, json=data, stream=True)

        full_response = ""

        for line in response.iter_lines():
            if line:
                # Decode and load JSON safely
                line_json = line.decode('utf-8')
                try:
                    data = json.loads(line_json)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

                # Extract and accumulate the response part
                response_part = data.get('response', '')
                full_response += response_part

                # Check completion status
                if data.get('done', False):
                    break

        return clean_response(full_response)
       
def main():

    deepseek_tester = DeepSeekLocalTester("deepseek-r1:8b", radius=10, ord_limit=31, ord_size=3, temperature=0)
    in_text = "what is 1/7/7/7/7/..."
    #in_text = "what kind of car does Michael Weston drive?"
    results = deepseek_tester.anharmoniticity(in_text)
    print(f"Anharmoniticity: {results['gamma']}\n")
    print(f"Central Answer: {results['answer']}\n")
    print(f"Perturbed Queries: {results['queries']}\n")
    print(f"Perturbed Outputs: {results['outputs']}\n")


if __name__ == "__main__":
    main()

