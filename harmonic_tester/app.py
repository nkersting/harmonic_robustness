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
    n_estimators = int(data.get('n_estimators'))
    learning_rate = float(data.get('learning_rate'))
    max_depth = int(data.get('max_depth'))
    min_samples_split = int(data.get('min_samples_split'))

    df = pd.read_csv('../Wine.csv')
    X = df.drop(['class'], axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # The two strongest features are Proline and Magnesium; however for pedagogical purposes we somewhat arbitrarily choose 2 of the weaker features 
    chosen_features = ['Flavanoids', 'OD280/OD315 of diluted wines']
    X_train_reduced = X_train[chosen_features]
    X_test_reduced = X_test[chosen_features]

    # Initialize the Gradient Boosting model
    gbdt_1 = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)

    # Train the model using the reduced feature set
    gbdt_1.fit(X_train_reduced, y_train)

    # Measure accuracy
    train_accuracy = accuracy_score(y_train, gbdt_1.predict(X_train_reduced))
    test_accuracy = accuracy_score(y_test, gbdt_1.predict(X_test_reduced))
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Create a mesh grid over the feature space
    x_min, x_max = X_train_reduced[chosen_features[0]].min() - 1, X_train_reduced[chosen_features[0]].max() + 1
    y_min, y_max = X_train_reduced[chosen_features[1]].min() - 1, X_train_reduced[chosen_features[1]].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), 
                         np.linspace(y_min, y_max, 500))

    # Predict on the grid points
    Z = gbdt_1.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # scale all Z values between 0 and 1
    Z = (Z - Z.min()) / (Z.max() - Z.min())

    # Convert the mesh grid and prediction results to lists
    xx_list = xx.ravel().tolist()
    yy_list = yy.ravel().tolist()
    
    # flatten Z to just a 1-D list
    Z_list = Z.ravel().tolist()

    # scale train_y values between 0 and 1
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())

    return jsonify({
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'xx': xx_list,
        'yy': yy_list,
        'Z': Z_list,
        'train_x': X_train_reduced[chosen_features[0]].tolist(),
        'train_y': X_train_reduced[chosen_features[1]].tolist(),
        'train_z': y_train.tolist(),
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    })

if __name__ == '__main__':
    app.run(debug=True)
