# app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model (ensure 'logistic_regression_iris_model.pkl' is in the same directory)
model = joblib.load('logistic_regression_iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data from the POST request
    features = np.array(data['features']).reshape(1, -1)  # Adjust the shape to match your model's input

    # Predict with the model
    prediction = model.predict(features)

    # Return the prediction as JSON
    return jsonify({'prediction': int(prediction[0])})  # Iris classes are 0, 1, 2

if __name__ == '__main__':
    app.run(debug=True)