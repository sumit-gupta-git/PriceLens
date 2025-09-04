from flask import Flask, render_template, request, jsonify
from src.prediction_service import PredictionService
import logging
app = Flask(__name__)
app.config.from_object('config.Config')
# Initialize prediction service
prediction_service = PredictionService()
@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        prediction = prediction_service.predict(data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)