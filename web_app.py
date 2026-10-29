import io
import os
import time
import uuid
import json
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager, get_jwt_identity, verify_jwt_in_request
from PIL import Image
from werkzeug.utils import secure_filename

from config import Config
from db import db, migrate
from db.models import Center, Doctor, Patient, Scan, ScanComparison
from db.uid import validate_patient_uid
from Classes.Web_Model import Web_Model
from Classes.ImageProcessing import ImageProcessing
from Classes.ReportGenerator import ReportGenerator
from Classes.HealthyTwin import HealthyTwin, fuse
from Classes.Storage import Storage
from Classes.ScanCompare import ScanCompare
from api.access import can_view_scan, write_audit_log

# Import Blueprints
from api.auth import auth_bp
from api.patients import patients_bp
from api.scans import scans_bp
from api.consent import consent_bp
from api.demo import demo_bp
from api.audit import audit_bp

def create_app(config_class=Config):
    app = Flask(__name__, template_folder='Templates', static_folder='static')
    app.config.from_object(config_class)

    # Init Extensions
    db.init_app(app)
    migrate.init_app(app, db)
    JWTManager(app)

    # Enable CORS
    CORS(
        app,
        resources={r"/api/*": {"origins": app.config['CORS_ORIGINS']}},
        allow_headers=["Content-Type", "Authorization"],
        supports_credentials=True
    )

    # Ensure storage directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)
    os.makedirs(app.config['STORAGE_DIR'], exist_ok=True)

    # Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(patients_bp)
    app.register_blueprint(scans_bp)
    app.register_blueprint(consent_bp)
    app.register_blueprint(demo_bp)
    app.register_blueprint(audit_bp)

    return app

app = create_app()

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

# Initialize Storage & Classification Model
storage = Storage()
model_path = os.path.join(os.path.dirname(__file__), "runs", "best_weight.h5")
model = Web_Model(model_path, ['COVID-19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS'])

# Initialize HealthyTwin Model
healthy_twin = None
if Config.HEALTHY_TWIN_ENABLED:
    try:
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

    original_name = secure_filename(file.filename)
    _, ext = os.path.splitext(original_name.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{ext}'. Allowed: png, jpg, jpeg"}), 400

    # Optional Auth & Patient UID check
    current_doctor = None
    try:
        verify_jwt_in_request(optional=True)
        doctor_id = get_jwt_identity()
        if doctor_id:
            current_doctor = Doctor.query.get(int(doctor_id))
    except Exception:
        pass

    patient_uid = request.form.get("patient_uid", request.form.get("id", "")).strip().upper()
    patient = None

    if patient_uid:
        patient = Patient.query.filter_by(uid=patient_uid).first()
        if not patient and not Config.ALLOW_ANONYMOUS_PREDICT:
            return jsonify({"error": f"Patient with UID {patient_uid} not found."}), 404

    if not current_doctor and not Config.ALLOW_ANONYMOUS_PREDICT:
        return jsonify({"error": "Authentication required to upload clinical scan records."}), 401

    try:
        scan_uuid = uuid.uuid4().hex
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        # 1. HealthyTwin Pre-gate Verification
        twin_analysis = None
        if healthy_twin is not None:
            ok, reason = healthy_twin.pre_gate(pil_image)
            if not ok:
                return jsonify({"error": f"Invalid X-Ray: {reason}"}), 422
            
            twin_analysis = healthy_twin.analyze(pil_image)
            if not twin_analysis.get("valid", True):
                return jsonify({"error": f"Invalid chest X-Ray: {twin_analysis.get('invalid_reason', 'Reconstruction error exceeded bounds.')}"}), 422

        # 2. Classifier Prediction (RGB)
        rgb_image = pil_image.convert("RGB")
        prediction, confidence, probabilities = model.predict_with_probs(rgb_image.copy())

        # 3. HealthyTwin Rendering & Clinical Fusion
        twin_payload = None
        triptych_bytes = None
        twin_bytes = None
        diff_bytes = None

        if healthy_twin is not None and twin_analysis is not None:
            try:
                fusion = fuse(prediction, confidence, twin_analysis)
                rendered = healthy_twin.render_panels(twin_analysis, Config.UPLOAD_FOLDER, scan_uuid)
                
                twin_payload = {
                    "abnormality_index": int(twin_analysis.get("abnormality_index", 0)),
                    "is_abnormal": bool(twin_analysis.get("is_abnormal", False)),
                    "status": fusion.get("status"),
                    "message": fusion.get("message"),
                    "images": {
                        "original": request.host_url + f"static/uploads/{scan_uuid}{ext}",
                        "twin": request.host_url + f"static/uploads/{rendered.get('twin_filename')}",
                        "diff": request.host_url + f"static/uploads/{rendered.get('diff_filename')}",
                        "triptych": request.host_url + f"static/uploads/{rendered.get('triptych_filename')}"
                    }
                }
            except Exception as e:
                print(f"Twin fusion rendering error: {e}")

        # 4. Image Enhancement for PDF Report
        enhanced_filename = f"enhanced_{scan_uuid}.png"
        enhanced_path = os.path.join(Config.UPLOAD_FOLDER, enhanced_filename)
        temp_orig_path = os.path.join(Config.UPLOAD_FOLDER, f"{scan_uuid}{ext}")
        with open(temp_orig_path, "wb") as f:
            f.write(image_bytes)
        ImageProcessing.enhance_image(temp_orig_path, enhanced_path)

        # 5. Probability Graph
        graph_filename = f"graph_{scan_uuid}.png"
        graph_path = os.path.join(Config.REPORT_FOLDER, graph_filename)
        ReportGenerator.generate_graph(probabilities, graph_path)

        # 6. Auto-compare against previous scan (if patient and earlier scans exist)
        comparison_payload = None
        if patient and current_doctor:
            earlier_scans = [s for s in patient.scans if s.is_deleted is False and can_view_scan(current_doctor, s)]
            if earlier_scans:
                prev_scan = earlier_scans[0] # Most recent previous permitted scan
                try:
                    prev_raw = storage.load_bytes(prev_scan.storage_key, 'original')
                    prev_pil = Image.open(io.BytesIO(prev_raw))
                    
                    prev_err = np.zeros((256, 256), dtype=np.float32)
                    if healthy_twin:
                        p_res = healthy_twin.analyze(prev_pil)
                        if p_res.get("valid") and p_res.get("error_map") is not None:
                            prev_err = p_res["error_map"]

                    curr_err = twin_analysis["error_map"] if twin_analysis and twin_analysis.get("error_map") is not None else np.zeros((256, 256), dtype=np.float32)

                    compare_res = ScanCompare.compare_scans(
                        prev_img=prev_pil,
                        new_img=pil_image,
                        prev_twin_index=prev_scan.twin_index or 0,
                        new_twin_index=twin_analysis.get("abnormality_index", 0) if twin_analysis else 0,
                        prev_error_map=prev_err,
                        new_error_map=curr_err,
                        prev_date_str=prev_scan.taken_at.strftime("%d %b %Y"),
                        prev_center_name=prev_scan.center.name
                    )

                    comparison_payload = {
                        "previous_scan_id": prev_scan.public_id,
                        "trend": compare_res["trend"],
                        "message": compare_res["message"],
                        "alignment_method": compare_res["alignment_method"],
                        "alignment_quality": compare_res["alignment_quality"],
                        "index_delta": compare_res["index_delta"],
                        "area_delta_pct": compare_res["area_delta_pct"]
                    }
                except Exception as e:
                    print(f"Comparison calculation error: {e}")

        # 7. Generate PDF Report
        patient_data = {
            "id": patient.uid if patient else request.form.get("id", "N/A"),
            "name": patient.name if patient else request.form.get("name", "Unknown"),
            "age": str(datetime.now().year - patient.birth_year) if patient and patient.birth_year else request.form.get("age", "Unknown"),
            "history": request.form.get("history", "None")
        }

        report_filename = f"report_{scan_uuid}.pdf"
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

        # 8. Persist to Secure Storage & Database if logged in and patient exists
        scan_record_id = None
        if patient and current_doctor:
            storage.save_bytes(scan_uuid, 'original', image_bytes)
            with open(enhanced_path, "rb") as f:
                storage.save_bytes(scan_uuid, 'enhanced', f.read())
            with open(graph_path, "rb") as f:
                storage.save_bytes(scan_uuid, 'graph', f.read())
            with open(report_path, "rb") as f:
                storage.save_bytes(scan_uuid, 'report', f.read())

            if twin_payload and 'rendered' in locals() and rendered:
                with open(rendered['twin_path'], "rb") as f:
                    storage.save_bytes(scan_uuid, 'twin', f.read())
                with open(rendered['diff_path'], "rb") as f:
                    storage.save_bytes(scan_uuid, 'diff', f.read())
                with open(rendered['triptych_path'], "rb") as f:
                    storage.save_bytes(scan_uuid, 'triptych', f.read())

            new_scan = Scan(
                public_id=scan_uuid,
                patient_id=patient.id,
                center_id=current_doctor.center_id,
                doctor_id=current_doctor.id,
                taken_at=datetime.now(timezone.utc),
                uploaded_at=datetime.now(timezone.utc),
                storage_key=scan_uuid,
                classifier_label=prediction,
                classifier_confidence=confidence,
                probabilities_json=json.dumps({k: round(v, 4) for k, v in probabilities.items()}),
                twin_raw_score=twin_analysis.get("raw_score") if twin_analysis else None,
                twin_index=twin_analysis.get("abnormality_index") if twin_analysis else None,
                twin_status=twin_payload.get("status") if twin_payload else None,
                doctor_notes=request.form.get("notes", "")
            )
            db.session.add(new_scan)
            db.session.commit()
            scan_record_id = new_scan.public_id

            write_audit_log('scan.create', doctor=current_doctor, patient_id=patient.id, scan_id=new_scan.id)

            # Update image URLs to authenticated endpoints
            if twin_payload:
                twin_payload["images"] = {
                    "original": f"/api/scans/{scan_uuid}/image/original",
                    "twin": f"/api/scans/{scan_uuid}/image/twin",
                    "diff": f"/api/scans/{scan_uuid}/image/diff",
                    "triptych": f"/api/scans/{scan_uuid}/image/triptych"
                }
            report_url = f"/api/scans/{scan_uuid}/report"

        return jsonify({
            "scan_id": scan_record_id,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "report_url": report_url,
            "twin": twin_payload,
            "comparison": comparison_payload
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "Flask API is running. Universal Scan Memory & Healthy Twin Active. Use Next.js frontend at port 3000."

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)
