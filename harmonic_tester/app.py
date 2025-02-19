from flask import Flask, render_template, request, jsonify
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np # linear algebra
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score



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

    df=pd.read_csv('../Wine.csv')
    X=df.drop(['class'],axis=1)
    y=df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)
    # The two strongest features are Proline and Magnesium; however for pedagogical purposes we somewhat arbitrarily choose 2 of the weaker features 
    chosen_features = ['Flavanoids', 'OD280/OD315 of diluted wines']
    X_train_reduced= X_train[chosen_features]
    X_test_reduced = X_test[chosen_features]

    # Initialize the Gradient Boosting model
    gbdt_1 = GradientBoostingClassifier(n_estimators=5, learning_rate=0.1, max_depth=1, min_samples_split=2, random_state=42)

    # Train the model using the reduced feature set
    gbdt_1.fit(X_train_reduced, y_train)

    # Measure accuracy
    print(f"Train Accuracy: {accuracy_score(y_train, gbdt_1.predict(X_train_reduced)):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, gbdt_1.predict(X_test_reduced)):.4f}")

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
