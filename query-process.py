#!/usr/bin/env python3

import cgi
import cgitb
import random
import requests
import json 
import time
from LLM_model_tester import LLM_model_tester

cgitb.enable()

def generate_perturbed_queries(query, n=10):
    return [f"{query} variant {i+1}" for i in range(n)]

def compute_gamma(query, model):
    # Placeholder for actual gamma computation
    return random.random()

def compute_perturbed_output(query, model):
    # Placeholder for actual output computation
    return f"Output for {query} with {model}"


def GPT_submit(self, question, model):
    url = 'https://api.openai.com/v1/chat/completions'
    data = {"model": model, "temperature": self.temperture,
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

def Claude_submit(self, question, model):
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
                model=model,
                temperature=self.temperature
    )
            return message.content[0].text
        except Exception as e:
            print("LLM EXCEPTION: ", e, message)
            time.sleep(10)    

print("Content-Type: text/html\n")

form = cgi.FieldStorage()
query = form.getvalue("query")
model = form.getvalue("model")
N = form.getvalue("n-value")

perturbed_queries = generate_perturbed_queries(query)
gamma = compute_gamma(query, model)
perturbed_outputs = [compute_perturbed_output(q, model) for q in perturbed_queries]

print(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Query Result</title>
    <style>
        .result-box {{
            border: 1px solid black;
            padding: 10px;
            margin: 10px;
        }}
        .side-window {{
            float: left;
            width: 45%;
            margin: 10px;
        }}
    </style>
</head>
<body>
    <h1>Query Result</h1>
    <div class="result-box">
        <h2>Gamma: {gamma}</h2>
    </div>
    <div class="side-window">
        <h3>Perturbed Queries</h3>
        <ul>
""")

for query in perturbed_queries:
    print(f"<li>{query}</li>")

print("""
        </ul>
    </div>
    <div class="side-window">
        <h3>Perturbed Outputs</h3>
        <ul>
""")

for output in perturbed_outputs:
    print(f"<li>{output}</li>")

print("""
        </ul>
    </div>
</body>
</html>
""")