#!/usr/bin/env python3

import cgi
import cgitb
from ClaudeTester import ClaudeTester
from GPT4Tester import GPT4Tester

cgitb.enable()

print("Content-Type: text/html\n")

form = cgi.FieldStorage()
query = form.getvalue("query")
model = form.getvalue("model")
N = int(form.getvalue("n-value"))

tester = None
if "claude" in model:
    tester = ClaudeTester(model, radius=N, ord_limit=31, ord_size=3, temperature=0)
elif "gpt-4" in model:
    tester = GPT4Tester(model, radius=N, ord_limit=31, ord_size=3, temperature=0)

if tester is not None:
    results = tester.anharmoniticity(query)
else:
    results = {
        "gamma": 0,
        "answer": "",
        "queries": [],
        "outputs": []
    }

gamma = results['gamma']
answer = results['answer']
perturbed_queries = results['queries']
perturbed_outputs = results['outputs']

print(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Query Result</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #f0f0f0, #c0c0c0);
        }}
        .result-box {{
            background-color: #fff;
            padding: 20px;
            border: 2px solid #ccc;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 20px;
        }}
        .link-box {{
            margin-top: 20px;
        }}
        .queries-outputs {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .query-output-pair {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }}
        .query, .output {{
            background-color: #fff;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin: 0 10px;
            width: 300px;
            word-wrap: break-word;
        }}
        .line {{
            width: 2px;
            height: 50px;
            background-color: #007bff;
        }}
    </style>
</head>
<body>
    <div class="result-box">
        <h1>Gamma Result</h1>
        <p>Gamma: {gamma:.3f}</p>
    </div>
    <div class="queries-outputs">
        <h2>Perturbed Queries and Outputs</h2>
        {"".join(f'<div class="query-output-pair"><div class="query">{pq}</div><div class="line"></div><div class="output">{po}</div></div>' for pq, po in zip(perturbed_queries, perturbed_outputs))}
    </div>
    <div class="link-box">
        <a href="/hallucination.html">Go back to the main page</a>
    </div>
</body>
</html>
""")
