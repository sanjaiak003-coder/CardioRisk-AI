"""
Heart Attack Risk Prediction - Web Application (Flask)
Interactive clinical diagnostic interface, dual-model comparison, and risk analytics.
"""

import os
import sys
import io
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, url_for

# Ensure src is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.predictor import HeartAttackPredictor
from src.data_loader import FEATURE_NAMES, CLINICAL_FEATURE_DESCRIPTIONS, load_heart_data

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB upload limit

# Initialize predictor
predictor = HeartAttackPredictor(models_dir=os.path.join(BASE_DIR, 'models'))


@app.route('/')
def index():
    """Main diagnostic workstation & dashboard."""
    metadata_path = os.path.join(BASE_DIR, 'models', 'model_metadata.json')
    model_metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)

    return render_template(
        'index.html',
        metadata=model_metadata,
        feature_descriptions=CLINICAL_FEATURE_DESCRIPTIONS
    )


@app.route('/api/predict', methods=['POST'])
def predict():
    """Real-time clinical prediction endpoint for a single patient."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Validate numeric types
        parsed_data = {}
        for col in FEATURE_NAMES:
            if col not in data:
                return jsonify({'error': f'Missing feature: {col}'}), 400
            parsed_data[col] = float(data[col])

        # Run inference
        result = predictor.predict_patient(parsed_data)
        return jsonify({'status': 'success', 'data': result})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for CSV files or pre-filled dataset."""
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            # If no file uploaded, run batch on first 20 records of benchmark dataset
            df = load_heart_data().head(25)
        else:
            file = request.files['file']
            df = pd.read_csv(file)

        # Ensure required features exist
        for col in FEATURE_NAMES:
            if col not in df.columns:
                return jsonify({'status': 'error', 'message': f'Uploaded CSV is missing required column: {col}'}), 400

        results_df = predictor.predict_batch(df)
        records = results_df.to_dict(orient='records')

        return jsonify({
            'status': 'success',
            'total_patients': len(results_df),
            'high_risk_count': int(results_df['High_Risk_Flag'].sum()),
            'records': records
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/export-batch-csv', methods=['POST'])
def export_batch_csv():
    """Exports batch predictions as a downloadable CSV."""
    try:
        data = request.get_json()
        records = data.get('records', [])
        if not records:
            return jsonify({'error': 'No records provided'}), 400

        df = pd.DataFrame(records)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='heart_attack_predictions_batch.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Returns training metrics, ROC curves, and feature importance data."""
    metadata_path = os.path.join(BASE_DIR, 'models', 'model_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'status': 'not_trained', 'message': 'Models not yet trained.'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting Heart Attack Prediction Server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
