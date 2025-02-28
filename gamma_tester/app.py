from flask import Flask, render_template, request, jsonify
import requests
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ClaudeTester import ClaudeTester
from GPT4Tester import GPT4Tester
from DeepSeekTester import DeepSeekTester

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_query', methods=['POST'])
def process_query():
    model = request.form['model']
    openaiapi_key = request.form['openaiapiKey']
    anthropicapi_key = request.form['anthropicapiKey']
    query = request.form['query']
    N = int(request.form['numVariants'])

    # set environment variables for api keys
    os.environ["OPENAI_API_KEY"] = openaiapi_key
    os.environ["ANTHROPIC_API_KEY"] = anthropicapi_key

    tester = None

    results = {
        "gamma": 0,
        "answer": "",
        "queries": [],
        "outputs": [],
        "N": N,
    }
    if "claude" in model:
        tester = ClaudeTester(model, radius=N, ord_limit=31, ord_size=3, temperature=0)
    elif "gpt" in model or "o1" in model:
        tester = GPT4Tester(model, radius=N, ord_limit=31, ord_size=3, temperature=0)
    elif "deepseek" in model:
        tester = DeepSeekTester(model, radius=N, ord_limit=31, ord_size=3, temperature=0)

    if tester is not None:
        results = tester.anharmoniticity(query)

    gamma = results['gamma']
    answer = results['answer']
    perturbed_queries = results['queries']
    perturbed_outputs = results['outputs']
   


    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
