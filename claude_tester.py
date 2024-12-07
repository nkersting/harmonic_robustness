#!/uAsr/bin/pythonA

import scipy
import time
import json
import requests
import random
import numpy as np
import os
from LLM_tester import LLMTester
from harmonic_tester import Point
from utils import normalize
from anthropic import Anthropic
from utils import sin_theta

class ClaudeTester(LLMTester):
    def __init__(self, radius=0, ord_limit=31, ord_size=3, temperature=0):
        """                                                                                                                      
        Specific to Anthropic's Claude
        Args:
        radius: denotes number of random string insertions to include in ball
        ord_limit: highest UTF8 value to assign to random chars
        ord_size: highest size of random strings to append
        temperature: ChatGPT parameter
        """
        super().__init__(self.claude_submit, radius, embedding=self.ADA_embedding)
        self.api_key = os.environ.get("OPENAI_API_KEY")  # export OPENAI_API_KEY=`cat openai.txt`
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")  # export ANTHROPIC_API_KEY=`cat claude.txt`
        self.temperature = temperature
        self.ord_limit = ord_limit
        self.ord_size = ord_size
        self.claude_client = Anthropic(api_key=self.anthropic_key)

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
        #print('BALL_NUMS: ', ball_nums)
        #print('BALL: ', '\t'.join(ball_points))
        return ball_points

        
    def claude_submit(self, question):
        
        while(1):
            message = None
            try:
                message = self.claude_client.messages.create(
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": question
                        }
                    ],
                    #model="claude-2.1",
                    model="claude-3-haiku-20240307",
                    temperature=self.temperature
                )
                return message.content[0].text
            except Exception as e:
                print("LLM EXCEPTION: ", e, message)
                time.sleep(10)


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

    curr_tester = ClaudeTester(radius=10, ord_limit=31, ord_size=3, temperature=0.0000001)
    #in_text = "Solve for x=sqrt(1+sqrt(7+sqrt(7+...)))"
    #in_text = "Solve for x=1/(7+1/(7+1/(7+...)))"
    #in_text = "Solve for x=1/(7*1/(7*1/(7*...)))"
    #in_text = "Who is my son's father's son's father's son's father's son's father?"
    #in_text = "Describe a particle physics theory with a 7th quark consistent with all known experimental signatures"
    #in_text = "Describe the electronic configuration of a stable element with atomic number 148"
    #in_text = "who does the united states export the most to?"
    #in_text = "who does the united states export the most to?"
    #in_text = "what school did sir isaac newton go to?"
    #in_text = "who was mary's mother?"
    #in_text = "who is rob kardashian dating now 2012?"
    #in_text = "who plays nana in the royal family?"
    #in_text = "where do logan browning live?"
    #in_text = "who played todd manning on one life to live?"
    #print(f"Anharmoniticity: {curr_tester.anharmoniticity(in_text)}")
    #exit()

    #f = open("/Users/lordkersting/neuro/Downloads/qa/webqa.tsv", "r")
    f = open("program_questions.tsv", "r")
    #f = open("truthful.tsv", "r")
    
    qas = []
    lines = f.readlines()
    for line in lines:
        qas.append(line.split("\t"))



    for i, qapair in enumerate(qas):
        if len(qapair) < 3:
            continue
        if i > 1000:
            break
        question = qapair[0]
        print(f"--------------------------------------------{i}")
        print(f"QUESTION={question}\tEXPECTED={qapair[1]}")
        anhar = curr_tester.anharmoniticity(question)
        print(f"{i}:\tANHAR={anhar}")

        




if __name__ == "__main__":
    main()
