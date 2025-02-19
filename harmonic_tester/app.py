from flask import Flask, render_template, request, jsonify
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('harmonic.html')

@app.route('/process_parameters', methods=['POST'])
def process_parameters():
    data = request.get_json()
    n_estimators = data.get('n_estimators')
    learning_rate = data.get('learning_rate')
    max_depth = data.get('max_depth')
    min_samples_split = data.get('min_samples_split')

    # Do something with the parameters
    result = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
