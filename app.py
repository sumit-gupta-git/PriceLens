# flask_backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model = None

def load_model():
    global model
    
    try:
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        logger.info("Model loaded successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        create_dummy_model()

#dummy model in case trained model is not loaded successfully
def create_dummy_model():
    global model
    
    from sklearn.ensemble import RandomForestRegressor
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    np.random.seed(42)
    dummy_data = pd.DataFrame({
        'brand_encoded': np.random.randint(0, 10, 1000),  # 10 different brands
        'vehicle_age': np.random.randint(1, 15, 1000),
        'km_driven': np.random.randint(5000, 200000, 1000),
        'seller_type_encoded': np.random.randint(0, 3, 1000),  # 3 seller types
        'fuel_type_encoded': np.random.randint(0, 5, 1000),  # 5 fuel types
        'transmission_type_encoded': np.random.randint(0, 2, 1000),  # 2 transmission types
        'mileage': np.random.uniform(10, 25, 1000),
        'engine': np.random.randint(800, 3000, 1000),
        'max_power': np.random.uniform(60, 300, 1000),
        'seats': np.random.choice([4, 5, 7, 8], 1000)
    })
    
    # Create realistic price based on features
    dummy_target = (
        50000 +  # base price
        dummy_data['brand_encoded'] * 20000 +  # brand effect
        (15 - dummy_data['vehicle_age']) * 15000 +  # age depreciation
        dummy_data['engine'] * 50 +  # engine effect
        dummy_data['max_power'] * 1000 +  # power effect
        (25 - dummy_data['mileage']) * 5000 +  # mileage effect
        np.random.normal(0, 50000, 1000)  # random noise
    )
    
    # Ensure positive prices
    dummy_target = np.maximum(dummy_target, 50000)
    
    model.fit(dummy_data, dummy_target)
    
    logger.info("Dummy model created for demonstration")

def encode_categorical_features(data):
    processed_data = data.copy()
    
    # Brand encoding
    brand_mapping = {
        'Maruti': 0, 'Hyundai': 1, 'Honda': 2, 'Toyota': 3, 'Ford': 4,
        'Mahindra': 5, 'Renault': 6, 'Nissan': 7, 'Tata': 8, 'Volkswagen': 9,
        'BMW': 10, 'Mercedes': 11, 'Audi': 12, 'Other': 13
    }
    processed_data['brand_encoded'] = processed_data['brand'].map(brand_mapping).fillna(13)
    
    # Seller type encoding
    seller_mapping = {'Dealer': 0, 'Individual': 1, 'TrustMark': 2}
    processed_data['seller_type_encoded'] = processed_data['seller_type'].map(seller_mapping).fillna(1)
    
    # Fuel type encoding
    fuel_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4}
    processed_data['fuel_type_encoded'] = processed_data['fuel_type'].map(fuel_mapping).fillna(0)
    
    # Transmission encoding
    transmission_mapping = {'Manual': 0, 'Automatic': 1}
    processed_data['transmission_type_encoded'] = processed_data['transmission_type'].map(transmission_mapping).fillna(0)
    
    return processed_data

def prepare_features(data):
    # Encode categorical variables
    processed_data = encode_categorical_features(data)
    
    feature_columns = ['brand_encoded', 'vehicle_age', 'km_driven', 'seller_type_encoded', 
                      'fuel_type_encoded', 'transmission_type_encoded', 'mileage', 'engine', 
                      'max_power', 'seats']
    
    X = processed_data[feature_columns]
    return X

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'PriceLens API is running',
        'version': '1.0',
        'model_loaded': model is not None,
        'timestamp': pd.Timestamp.now().isoformat()
    })

@app.route('/predict', methods=['GET','POST'])
def predict_price():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        required_fields = ['brand', 'vehicle_age', 'km_driven', 'seller_type', 
                          'fuel_type', 'transmission_type', 'mileage', 'engine', 
                          'max_power', 'seats']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        input_df = pd.DataFrame([data])
        X = prepare_features(input_df)

        prediction = model.predict(X)[0]
        
        # Format prediction (round to 2 decimal places)
        predicted_price = round(float(prediction), 2)

        return jsonify({
            'status': 'success',
            'predicted_price': predicted_price,
            'currency': 'INR',
            'input_data': data
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': pd.Timestamp.now().isoformat()
    })

@app.route('/sample-data', methods=['GET'])
def get_sample_data():
    try:
        # Generate sample market data
        np.random.seed(42)
        
        brands = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Mahindra', 'Tata', 'Renault']
        fuel_types = ['Petrol', 'Diesel', 'CNG', 'Electric']
        
        sample_data = {
            'brand_stats': [
                {'brand': brand, 
                 'avg_price': round(np.random.uniform(300000, 1200000), 2),
                 'avg_km_driven': round(np.random.uniform(20000, 80000), 0),
                 'count': np.random.randint(50, 200)}
                for brand in brands
            ],
            'fuel_type_stats': [
                {'fuel_type': fuel, 
                 'avg_price': round(np.random.uniform(400000, 1000000), 2),
                 'median_price': round(np.random.uniform(350000, 900000), 2),
                 'count': np.random.randint(100, 500)}
                for fuel in fuel_types
            ],
            'age_price_data': [
                {'vehicle_age': age,
                 'avg_price': round(1200000 - (age * 60000) + np.random.normal(0, 50000), 2),
                 'count': np.random.randint(20, 100)}
                for age in range(1, 16)
            ]
        }
        
        return jsonify({
            'status': 'success',
            'data': sample_data
        })
        
    except Exception as e:
        logger.error(f"Sample data generation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    load_model()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)