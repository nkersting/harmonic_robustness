#!/uAsr/bin/pythonAA

import time
import json
import requests
import random
import os
import numpy as np
from LLM_tester import LLMTester
from harmonic_tester import Point



class LLM_model_tester(LLMTester):
    def __init__(self, model_submitter, radius=0, ord_limit=31, ord_size=3):
        """                                                                                                                      
        Args:
        API_key: string for OpenAI access
        radius: denotes number of random string insertions to include in ball
        temperature: GPT4o parameter
        """
        super().__init__(model_submitter, radius, embedding=self.ADA_embedding)
        self.ord_limit = ord_limit
        self.ord_size = ord_size
        self.api_key = os.getenv("OPENAI_API_KEY")


    def ball(self, point:Point, radius) -> list[Point]:
        """
        Here we generate strings 'close' to the original string by appending random control characters (ASCII 0-31)

        Args:
        radius: the number of strings on the 'ball' to generate
        """

        ball_points = []
        for _ in range(radius):
            randomstring = "".join([chr(random.randint(0, self.ord_limit)) for _ in range(random.randint(1,self.ord_size))])
            ball_points.append(point + " " + randomstring)
        ball_nums = [[ord(x) for x in y] for y in ball_points]
        return ball_points

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
                time.sleep(1)

     