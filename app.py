from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
import numpy as np

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Define the acceptable ranges for each feature
ranges = {
    'ph': (0, 14),
    'hardness': (0, 400),
    'solids': (0, 50000),
    'chloramines': (0, 20),
    'sulfate': (0, 500),
    'conductivity': (0, 1000),
    'organic_carbon': (0, 30),
    'trihalomethanes': (0, 200),
    'turbidity': (0, 10)
}

@app.route('/')
def index():
    return render_template('index.html', feature_ranges=ranges)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and validate input data from form
        ph = float(request.form['ph'])
        hardness = float(request.form['hardness'])
        solids = float(request.form['solids'])
        chloramines = float(request.form['chloramines'])
        sulfate = float(request.form['sulfate'])
        conductivity = float(request.form['conductivity'])
        organic_carbon = float(request.form['organic_carbon'])
        trihalomethanes = float(request.form['trihalomethanes'])
        turbidity = float(request.form['turbidity'])

        # Creating an array for prediction
        input_features = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])

        # Perform prediction
        prediction = model.predict(input_features)

        # Determine if water is potable
        Output = "Drinkable" if prediction[0] == 1 else "Not Drinkable"

        # Send back the input data and prediction result to the frontend
        result = {
            'output': Output,
            'pH': ph,
            'HARDNESS': hardness,
            'SOLIDS': solids,
            'CHLORAMINES': chloramines,
            'SULFATE': sulfate,
            'CONDUCTIVITY': conductivity,
            'ORGANIC_CARBON': organic_carbon,
            'TRIHALOMETHANES': trihalomethanes,
            'TURBIDITY': turbidity
        }

        return jsonify(result)

    except ValueError:
        return jsonify({'error': 'Invalid input: Please enter numeric values'})

if __name__ == '__main__':
    app.run(debug=True)


