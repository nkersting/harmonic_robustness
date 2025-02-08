import time
import json
import requests
import os
from functools import partial
from LLM_model_tester import LLM_model_tester
from openai import OpenAI



class DeepSeekTester(LLM_model_tester):
    def __init__(self, model_name, radius=0, ord_limit=31, ord_size=3, temperature=0):
        """
        Args:
        API_key: string for DeepSeek access
        radius: denotes number of random string insertions to include in ball
        temperature: DeepSeek parameter
        """
        super().__init__(partial(self.DeepSeek_submitter, model_name), radius, ord_limit, ord_size)
        self.temperature = temperature
        self.client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


    def get_models(self):
        url = "https://api.deepseek.com/v1/models"
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
            
    def DeepSeek_submitter(self, model, question):
        while(1):
            try:
                response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": question},
                ],
                temperature=self.temperature,
                stream=False)
                return response.choices[0].message.content
            except Exception as e:
                print("LLM EXCEPTION: ", e)
                time.sleep(60)

def main():

    deepseek_tester = DeepSeekTester("deepseek-reasoner", radius=2, ord_limit=31, ord_size=3, temperature=0)

    #print(gpt4o_mini_tester.get_models())
    #exit()
    in_text = "what is 1/7/7/7/7/..."
    #in_text = "what kind of car does Michael Weston drive?"
    results = deepseek_tester.anharmoniticity(in_text)
    print(f"Anharmoniticity: {results['gamma']}\n")
    print(f"Central Answer: {results['answer']}\n")
    print(f"Perturbed Queries: {results['queries']}\n")
    print(f"Perturbed Outputs: {results['outputs']}\n")


if __name__ == "__main__":
    main()