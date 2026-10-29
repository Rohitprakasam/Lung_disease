import numpy as np
import os
import time
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename

from config import Config
from Classes.Web_Model import Web_Model
from Classes.ImageProcessing import ImageProcessing
from Classes.ReportGenerator import ReportGenerator

# Ensure storage directories exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.REPORT_FOLDER, exist_ok=True)

# Create Flask App
app = Flask(__name__, template_folder='Templates', static_folder='static')
app.config.from_object(Config)

# Enable CORS with restricted origins and credentials / headers
CORS(
    app,
    resources={r"/api/*": {"origins": Config.CORS_ORIGINS}},
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=True
)

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

# Model Initialization
model_path = os.path.join(os.path.dirname(__file__), "runs", "best_weight.h5")
model = Web_Model(model_path, ['COVID-19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS'])

# Placeholder for HealthyTwin model (loaded in Phase 1)
healthy_twin = None
if Config.HEALTHY_TWIN_ENABLED:
    try:
        from Classes.HealthyTwin import HealthyTwin
        twin_model_path = os.path.join(os.path.dirname(__file__), "runs", "healthy_twin.keras")
        twin_meta_path = os.path.join(os.path.dirname(__file__), "runs", "healthy_twin_meta.json")
        if os.path.exists(twin_model_path) and os.path.exists(twin_meta_path):
            healthy_twin = HealthyTwin(twin_model_path, twin_meta_path)
            print("HealthyTwin model loaded successfully.")
    except Exception as e:
        print(f"HealthyTwin not initialized: {e}")

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File size exceeds the 15 MB limit."}), 413

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": "Invalid file"}), 400

    # Validate file extension
    original_name = secure_filename(file.filename)
    _, ext = os.path.splitext(original_name.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{ext}'. Allowed: png, jpg, jpeg"}), 400

    # Get Patient Details
    patient_data = {
        "id": request.form.get("id", "N/A"),
        "name": request.form.get("name", "Unknown"),
        "age": request.form.get("age", "Unknown"),
        "history": request.form.get("history", "None")
    }

    try:
        # 1. Save Original Image using UUID
        file_uuid = uuid.uuid4().hex
        filename = f"{file_uuid}{ext}"
        original_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(original_path)

        # 2. Check Image Validity (if HealthyTwin is enabled)
        pil_image = Image.open(original_path)
        twin_result = None
        if healthy_twin is not None:
            ok, reason = healthy_twin.pre_gate(pil_image)
            if not ok:
                return jsonify({"error": f"Invalid X-Ray: {reason}"}), 422
            
            twin_analysis = healthy_twin.analyze(pil_image)
            if not twin_analysis.get("valid", True):
                return jsonify({"error": f"Invalid chest X-Ray: {twin_analysis.get('invalid_reason', 'Reconstruction error exceeded bounds.')}"}), 422

        # 3. Enhance Image for Display/PDF (CLAHE + Turbo heatmap blend)
        enhanced_filename = f"enhanced_{file_uuid}.png"
        enhanced_path = os.path.join(Config.UPLOAD_FOLDER, enhanced_filename)
        ImageProcessing.enhance_image(original_path, enhanced_path)

        # 4. Predict using Real Probabilities (Using original RGB image)
        rgb_image = pil_image.convert("RGB")
        prediction, confidence, probabilities = model.predict_with_probs(rgb_image.copy())

        # 5. Process Healthy Twin Fusion (if available)
        twin_payload = None
        if healthy_twin is not None and 'twin_analysis' in locals():
            try:
                from Classes.HealthyTwin import fuse
                fusion = fuse(prediction, confidence, twin_analysis)
                rendered = healthy_twin.render_panels(twin_analysis, Config.UPLOAD_FOLDER, file_uuid)
                
                twin_payload = {
                    "abnormality_index": int(twin_analysis.get("abnormality_index", 0)),
                    "is_abnormal": bool(twin_analysis.get("is_abnormal", False)),
                    "status": fusion.get("status"),
                    "message": fusion.get("message"),
                    "images": {
                        "original": request.host_url + f"static/uploads/{filename}",
                        "twin": request.host_url + f"static/uploads/{rendered.get('twin_filename')}",
                        "diff": request.host_url + f"static/uploads/{rendered.get('diff_filename')}",
                        "triptych": request.host_url + f"static/uploads/{rendered.get('triptych_filename')}"
                    }
                }
            except Exception as e:
                print(f"HealthyTwin rendering error: {e}")
                twin_payload = None

        # 6. Generate Probability Graph
        graph_filename = f"graph_{file_uuid}.png"
        graph_path = os.path.join(Config.REPORT_FOLDER, graph_filename)
        ReportGenerator.generate_graph(probabilities, graph_path)

        # 7. Generate PDF Report
        report_filename = f"report_{file_uuid}.pdf"
        report_path = os.path.join(Config.REPORT_FOLDER, report_filename)
        ReportGenerator.generate_pdf(
            patient_data, 
            prediction, 
            confidence, 
            probabilities, 
            enhanced_path, 
            graph_path, 
            report_path,
            twin=twin_payload
        )
        
        report_url = request.host_url + f"static/reports/{report_filename}"

        response_data = {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "report_url": report_url,
            "twin": twin_payload
        }
        return jsonify(response_data)

    except Exception as e:
        print(f"Error processing the image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "Flask API is running. Use frontend at port 3000."

if __name__ == "__main__":
    app.run(debug=True, port=5000)
