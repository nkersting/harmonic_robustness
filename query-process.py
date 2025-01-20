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
N = form.getvalue("n-value")

tester = None
if "claude" in model:
    tester = ClaudeTester(model, radius=N, ord_limit=31, ord_size=3, temperature=0)
elif "gpt-4" in model:
    tester = GPT4Tester(model, radius=N, ord_limit=31, ord_size=3, temperature=0)

if tester is not None:
    gamma, perturbed_queries, perturbed_outputs = tester.anharmoniticity(query)
else:
    gamma, perturbed_queries, perturbed_outputs = 0, [], []


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