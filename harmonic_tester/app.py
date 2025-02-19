from flask import Flask, render_template, request, jsonify
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('harmonic.html')



if __name__ == '__main__':
    app.run(debug=True)
