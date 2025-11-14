from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from collections import Counter
import os

app = Flask(__name__)
CORS(app)

# Load models at startup
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model_artifacts')
model = None
encoder = None
known_apps = []

def load_models():
    global model, encoder, known_apps
    model_path = os.path.join(MODEL_DIR, 'next_app_model', 'next_app_model.pkl')
    encoder_path = os.path.join(MODEL_DIR, 'next_app_model', 'app_encoder.pkl')
    
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    known_apps = list(encoder.classes_)
    print(f"✅ Models loaded! Known apps: {len(known_apps)}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': model is not None,
        'known_apps': len(known_apps)
    })

@app.route('/api/predict-next-app', methods=['POST'])
def predict_next_app():
    try:
        data = request.json.get('data', [])
        
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Expected "data" field with array of app access records'}), 400
        
        df = pd.DataFrame(data)
        
        # Filter out Login and Logout
        df = df[~df['appDisplayName'].isin(['Login', 'Logout'])]
        
        if len(df) == 0:
            return jsonify({'error': 'No valid app records after filtering Login/Logout'}), 400
        
        # Extract features
        most_recent_app = df['appDisplayName'].iloc[-1]
        previous_app = df['appDisplayName'].iloc[-2] if len(df) >= 2 else None
        previous_previous_app = df['appDisplayName'].iloc[-3] if len(df) >= 3 else None
        session_position = len(df) - 1
        avg_hour = int(round(df['hour'].mean()))
        most_common_weekday = Counter(df['weekday']).most_common(1)[0][0]
        
        # Safe encode function
        def safe_encode_app(app_name):
            if app_name and app_name in known_apps:
                return encoder.transform([app_name])[0]
            return -1
        
        most_recent_app_encoded = safe_encode_app(most_recent_app)
        previous_app_encoded = safe_encode_app(previous_app)
        previous_previous_app_encoded = safe_encode_app(previous_previous_app)
        
        # Build feature vector
        X = pd.DataFrame({
            'appDisplayName_encoded': [most_recent_app_encoded],
            'previous_app_encoded': [previous_app_encoded],
            'previous_previous_app_encoded': [previous_previous_app_encoded],
            'session_position': [session_position],
            'hour': [avg_hour],
            'weekday': [most_common_weekday]
        })
        
        # Predict top 5 apps
        pred_proba = model.predict_proba(X)[0]
        top_5_indices = pred_proba.argsort()[-5:][::-1]
        top_5_apps = encoder.inverse_transform(top_5_indices)
        top_5_probabilities = pred_proba[top_5_indices]
        
        predictions = [
            {
                "app": app,
                "confidence": float(prob),
                "rank": i + 1
            }
            for i, (app, prob) in enumerate(zip(top_5_apps, top_5_probabilities))
        ]
        
        return jsonify({
            "predictions": predictions,
            "top_prediction": predictions[0]["app"]
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🐍 Starting Python ML Server...")
    load_models()
    app.run(host='0.0.0.0', port=5001, debug=True)
