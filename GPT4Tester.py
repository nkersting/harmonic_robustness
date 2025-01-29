#!/uAsr/bin/pythonAA

import time
import json
import requests
import os
from functools import partial
from LLM_model_tester import LLM_model_tester


class GPT4Tester(LLM_model_tester):
    def __init__(self, model_name, radius=0, ord_limit=31, ord_size=3, temperature=0):
        """
        Args:
        API_key: string for OpenAI access
        radius: denotes number of random string insertions to include in ball
        temperature: GPT4o parameter
        """
        super().__init__(partial(self.GPT4_submitter, model_name), radius, ord_limit, ord_size)
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY")

    def get_models(self):
        url = "https://api.openai.com/v1/models"
        headers = {'content-type': 'application/json', 
                    'Authorization': self.api_key}
        payload = {'headers': headers}
        while(1):
            r = None
            try:
                r = requests.get(url, headers=headers)
                return r.text
            except Exception as e:
                print("LLM EXCEPTION: ", e, r)
                time.sleep(60)
            
    def GPT4_submitter(self, model, question):
        url = 'https://api.openai.com/v1/chat/completions'
        data = {"model": model, "temperature": self.temperature, 
                "messages": [{"role": "user",
                                "content": question
                                }],
                "n": 1
                }
        headers = {'content-type': 'application/json', 
                    'Authorization': self.api_key}
        payload = {'data': data, 'headers': headers}
        while(1):
            r = None
            try:
                r = requests.post(url, data=json.dumps(data), headers=headers)
                content = json.loads(r.content.decode('utf8'))
                return content["choices"][0]['message']['content']
            except Exception as e:
                print("LLM EXCEPTION: ", e, r)
                time.sleep(60)
       
def main():

    gpt4o_mini_tester = GPT4Tester("gpt-4o-mini", radius=10, ord_limit=31, ord_size=3, temperature=0)

    #print(gpt4o_mini_tester.get_models())
    #exit()
    in_text = "what is 1/7/7/7/7/..."
    #in_text = "what kind of car does Michael Weston drive?"
    results = gpt4o_mini_tester.anharmoniticity(in_text)
    print(f"Anharmoniticity: {results['gamma']}\n")
    print(f"Central Answer: {results['answer']}\n")
    print(f"Perturbed Queries: {results['queries']}\n")
    print(f"Perturbed Outputs: {results['outputs']}\n")


if __name__ == "__main__":
    main()

