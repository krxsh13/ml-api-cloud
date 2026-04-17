from flask import Flask, request, jsonify
import joblib
import os

application = Flask(__name__)

# Load model
model = joblib.load('sentiment_model.joblib')

# ✅ Home route (for AWS health check)
@application.route('/')
def home():
    return "API is running"

# ✅ Prediction route
@application.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    prediction = model.predict([text])[0]

    return jsonify({
        'input_text': text,
        'sentiment_prediction': prediction
    })

# ✅ IMPORTANT: dynamic port for AWS
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    application.run(host='0.0.0.0', port=port)