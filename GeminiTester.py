import time
import json
import requests
import os
from functools import partial
from LLM_model_tester import LLM_model_tester
from openai import OpenAI
from google import genai



class GeminiTester(LLM_model_tester):
    def __init__(self, model_name, radius=0, ord_limit=31, ord_size=3, temperature=0):
        """
        Args:
        API_key: string for Gemini access
        radius: denotes number of random string insertions to include in ball
        temperature: Gemini parameter
        """
        super().__init__(partial(self.Gemini_submitter, model_name), radius, ord_limit, ord_size)
        self.temperature = temperature
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


            
    def Gemini_submitter(self, model, question):
        while(1):
            try:
                response = self.client.models.generate_content(
                    model=model, contents=question
                )
                return response.text
            except Exception as e:
                print("LLM EXCEPTION: ", e)
                time.sleep(60)

def main():

    gemini_tester = GeminiTester("gemini-2.0-flash", radius=2, ord_limit=31, ord_size=3, temperature=0)

    #print(gpt4o_mini_tester.get_models())
    #exit()
    in_text = "what is 1/7/7/7/7/..."
    #in_text = "what kind of car does Michael Weston drive?"
    results = gemini_tester.anharmoniticity(in_text)
    print(f"Anharmoniticity: {results['gamma']}\n")
    print(f"Central Answer: {results['answer']}\n")
    print(f"Perturbed Queries: {results['queries']}\n")
    print(f"Perturbed Outputs: {results['outputs']}\n")


if __name__ == "__main__":
    main()
