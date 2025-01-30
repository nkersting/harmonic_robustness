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
elif "gpt-4" in model or 'o1' in model:
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

def escape_control_chars(text):
    return ''.join(f'<span style="color:red;">/x{ord(c):02x}</span>' if ord(c) < 32 else c for c in text)

escaped_queries = [escape_control_chars(pq) for pq in perturbed_queries]

# Determine the gamma message and color
if gamma < 0.1:
    gamma_message = "Probably Trustworthy"
    gamma_color = "green"
elif 0.1 <= gamma <= 0.2:
    gamma_message = "Possible Hallucination"
    gamma_color = "yellow"
else:
    gamma_message = "Likely Hallucination"
    gamma_color = "red"

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
        .gamma-box {{
            background-color: #39ff14; /* Electric green background */
            padding: 10px;
            border: 2px solid #ccc;
            border-radius: 10px;
            box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.1);
            font-size: 24px;
            font-weight: bold;
            color: {gamma_color};
            margin-top: 10px;
        }}
        .link-box {{
            margin-top: 20px;
        }}
        .queries-outputs {{
            display: flex;
            flex-direction: column;
            align-items: center;
            max-height: 50vh;
            overflow-y: auto;
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
            width: 400px; /* Increased width */
            word-wrap: break-word;
        }}
        .query {{
            background-color: #e0faef;
        }}
        .output {{
            background-color: #85edc7;
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
        <h1>Gamma</h1>
        <div class="gamma-box">{gamma:.3f}</div>
        <p>{gamma_message}</p>
    </div>
    <div class="queries-outputs">
        <h2>Perturbed Queries and Outputs</h2>
        {"".join(f'<div class="query-output-pair"><div class="query">{pq}</div><div class="line"></div><div class="output">{po}</div></div>' for pq, po in zip(escaped_queries, perturbed_outputs))}
    </div>
    <div class="link-box">
        <a href="/hallucination.html">Go back to measure another query</a>
    </div>
</body>
</html>
""")
