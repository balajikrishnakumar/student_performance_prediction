from flask import Flask, request, jsonify
import os, sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.append(SRC)

from predict import load_model, predict_from_dict

app = Flask(__name__)
model = load_model()

@app.route('/')
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    if not payload:
        return jsonify({'error': 'Invalid JSON payload'}), 400
    try:
        pred = predict_from_dict(model, payload)
        return jsonify({'predicted_G3': pred})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
