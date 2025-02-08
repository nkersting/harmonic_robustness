#!/uAsr/bin/python

import scipy
import time
import json
import requests
import random
import numpy as np
from LLM_tester import LLMTester
from harmonic_tester import Point
from utils import normalize


def find_companion_words(word, lexicon):
    word = word.lower()
    if word not in lexicon:
        return word, word
    comp1, v1 = find_closest([word], lexicon[word], lexicon)
    comp2, v2 = find_closest([comp1, word], normalize(2*lexicon[word] - lexicon[comp1]), lexicon)
    return comp1, comp2

def find_closest(wordlist, vec, lexicon):
    all_vecs = [(w, lexicon[w]) for w in lexicon if w not in wordlist]
    tree = scipy.spatial.KDTree([v[1] for v in all_vecs])
    dist, pos =  tree.query(vec)
    closest_word = all_vecs[pos][0]
    return closest_word, lexicon[closest_word] 

def get_GloVe_lexicon(fileloc="glove/glove.6B.300d.txt"):
    # map english words to embeddings:
    english_vectors = {}
    
    with open(fileloc, "r") as f:
        lines = f.readlines()
        for line in lines:
            parsed = line.split()
            english_vectors[parsed[0]] = np.array([float(x) for x in parsed[1:]])
    return english_vectors


class ChatGPTWordTester(LLMTester):
    def __init__(self, lexicon, API_key, radius=0, temperature=0):
        """                                                                                                                      
        Specific to OpenAI's ChatGPT model
        Args:
        lexicon: dictionary with key=word, value=vector
        API_key: string for OpenAI access
        radius: denotes number of random string insertions to include in ball
        temperature: ChatGPT parameter
        """
        super().__init__(self.ChatGPT_submit, radius, embedding=self.ADA_embedding)
        self.api_key = API_key
        self.temperture = temperature
        self.lexicon = lexicon
            
    def ball(self, point:Point, radius) -> list[Point]:
        """
        Since we don't have access to the layer vectorization we approximate the ball by
        replacing each word with pairs of oppositely aligned words.

        Args:
        radius: denotes the number of random words to insert as additional perturbations
        """
        words = point.split()
        ball_points = []
        for i,w in enumerate(words):
            comp1, comp2 = find_companion_words(w, self.lexicon)
            for c in comp1, comp2:
                if c != w:
                    ball_points.append(" ".join(words[:i] + [c] + words[i+1:]))
        for _ in range(radius):
            rand_idx = random.randint(0,len(words))
            randomstring = "".join([chr(random.randint(1, 3200)) for _ in range(random.randint(1,8))])
            ball_points.append(" ".join(words[:rand_idx] + [randomstring] + words[rand_idx:]))

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
    #english_vectors = get_GloVe_lexicon("glove/glove.6B.300d.txt") # https://nlp.stanford.edu/data/glove.6B.zip
    english_vectors = get_GloVe_lexicon("/Users/lordkersting/neuro/Downloads/glove/glove.6B.300d.txt")
    n_english_vectors = {x:normalize(english_vectors[x]) for x in english_vectors}
    API_key = "Bearer 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"  # replace with own
    
    curr_tester = ChatGPTWordTester(n_english_vectors, API_key)

    in_text = "Answer simply yes or no: did jesus_christ really walk on water?"
    print(f"Anharmoniticity: {curr_tester.anharmoniticity(in_text)}")





if __name__ == "__main__":
    main()
