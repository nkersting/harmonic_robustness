from flask import Flask, render_template, request, jsonify
import requests
import os

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
    
    # Placeholder for actual processing logic
    # Here you would call the respective LLM API with the provided query and API key
    # For demonstration, we will return a mock response
    
    response = {
        "original_query": query,
        "model": model,
        "api_key": api_key,
        "output": "This is a mock response from the model.",
        "perturbed_outputs": [
            "Mock response 1",
            "Mock response 2",
            "Mock response 3"
        ],
        "hallucination_metric": 0.5
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
