from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app) 

model = joblib.load('gbModel.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "Welcome to the Housing Price Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    location = data.get('location', '')
    
    location_mapping = {
        "São Paulo": [0, 0, 0, 0, 1],
        "Campinas": [0, 1, 0, 0, 0],
        "Porto Alegre": [0, 0, 1, 0, 0],
        "Rio de Janeiro": [0, 0, 0, 1, 0],
        "Belo Horizonte": [1, 0, 0, 0, 0]
    }
    
    location_values = location_mapping.get(location, [0, 0, 0, 0, 0])
    size = float(data.get('size', 0))
    rooms = int(data.get('rooms', 0))
    bathrooms = int(data.get('bathrooms', 0))
    park = int(data.get('park', 0))
    hoa = float(data.get('hoa', 0))
    tax = float(data.get('tax', 0))
    fi = float(data.get('fi', 0))
    animals = data.get('animals', '')
    furniture = data.get('furniture', '')

    # Convert animals to numerical values (example)
    animals_mapping = {
        'Yes': [1, 0],
        'No': [0, 1]
    }
    print(animals)
    animals_value = animals_mapping.get(animals, 0)
    print(animals_value)
    
    furniture_mapping = {
        'Yes': [1, 0],
        'No': [0, 1]
    }
    furniture_value = furniture_mapping.get(furniture, [0, 1])

    numeric_features = np.array([size, rooms, bathrooms, park, hoa, tax, fi]).reshape(1, -1)
    scaled_numeric_features = scaler.transform(numeric_features)[0]
       

    features = np.concatenate([scaled_numeric_features, location_values, animals_value, furniture_value]).reshape(1, -1)

    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)