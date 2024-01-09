#!/uAsr/bin/pythonA

import scipy
import time
import json
import requests
import random
import numpy as np
from LLM_tester import LLMTester
from harmonic_tester import Point
from utils import normalize


class ChatGPTTester(LLMTester):
    def __init__(self, API_key, radius=0, temperature=0):
        """                                                                                                                      
        Specific to OpenAI's ChatGPT model
        Args:
        API_key: string for OpenAI access
        radius: denotes number of random string insertions to include in ball
        temperature: ChatGPT parameter
        """
        super().__init__(self.ChatGPT_submit, radius, embedding=self.ADA_embedding)
        self.api_key = API_key
        self.temperture = temperature


    def ball(self, point:Point, radius) -> list[Point]:
        """
        Here we generate strings 'close' to the original string by appending random control characters (ASCII 0-31)

        Args:
        radius: the number of strings on the 'ball' to generate
        """

        ball_points = []
        for _ in range(radius):
            randomstring = "".join([chr(random.randint(0, 31)) for _ in range(random.randint(1,3))])
            ball_points.append(point + " " + randomstring)
        print('BALL: ', '\t'.join(ball_points))
        return ball_points
                
    def ChatGPT_submit(self, question):
        url = 'https://api.openai.com/v1/chat/completions'
        data = {"model": "gpt-3.5-turbo", "temperature": self.temperture, 
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


    def ADA_embedding(self, content):
        url = 'https://api.openai.com/v1/embeddings'
        data = {"input": content, "model":"text-embedding-ada-002"}
        headers = {'content-type': 'application/json', 
                   'Authorization': self.api_key}
        payload = {'data': data, 'headers': headers}
        while(1):
            try:
                r = requests.post(url, data=json.dumps(data), headers=headers)
                content = json.loads(r.content.decode('utf8'))
                embedding = content['data'][0]['embedding']
                return  np.array(embedding)
            except Exception as e:
                print("ADA EXCEPTION: ", e, content)
                time.sleep(60)

        
def main():

    API_key = ""  # replace with own
    curr_tester = ChatGPTTester(API_key, radius=10)
    in_text = "Answer simply yes or no: did jesus_christ really walk on water?"
    print(f"Anharmoniticity: {curr_tester.anharmoniticity(in_text)}")





if __name__ == "__main__":
    main()
