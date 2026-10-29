import io
from flask import Blueprint, request, jsonify, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from db import db
from db.models import Scan, Doctor
from Classes.Storage import Storage
from api.access import can_view_scan, write_audit_log

scans_bp = Blueprint('scans_bp', __name__)
storage = Storage()

ALLOWED_IMAGE_KINDS = {'original', 'enhanced', 'twin', 'diff', 'triptych', 'graph'}

@scans_bp.route('/api/scans/<public_id>', methods=['GET'])
@jwt_required()
def get_scan_details(public_id):
    doctor_id = get_jwt_identity()
    doctor = db.session.get(Doctor, int(doctor_id))
    scan = Scan.query.filter_by(public_id=public_id, is_deleted=False).first()
    if not scan:
        return jsonify({'error': 'Scan not found'}), 404

    if not can_view_scan(doctor, scan):
        return jsonify({'error': 'Access denied: Patient consent required for external center scans.'}), 403

    write_audit_log('scan.view', doctor=doctor, patient_id=scan.patient_id, scan_id=scan.id)

    return jsonify({
        'id': scan.id,
        'public_id': scan.public_id,
        'taken_at': scan.taken_at.isoformat(),
        'center': {
            'id': scan.center.id,
            'name': scan.center.name,
            'code': scan.center.code
        },
        'classifier_label': scan.classifier_label,
        'classifier_confidence': round(scan.classifier_confidence, 4),
        'probabilities': scan.probabilities,
        'twin_index': scan.twin_index,
        'twin_status': scan.twin_status,
        'doctor_notes': scan.doctor_notes,
        'doctor_confirmed_label': scan.doctor_confirmed_label
    })

@scans_bp.route('/api/scans/<public_id>/image/<kind>', methods=['GET'])
@jwt_required()
def get_scan_image(public_id, kind):
    if kind not in ALLOWED_IMAGE_KINDS:
        return jsonify({'error': f'Invalid image kind. Allowed: {list(ALLOWED_IMAGE_KINDS)}'}), 400

    doctor_id = get_jwt_identity()
    doctor = db.session.get(Doctor, int(doctor_id))
    scan = Scan.query.filter_by(public_id=public_id, is_deleted=False).first()
    if not scan:
        return jsonify({'error': 'Scan not found'}), 404

    if not can_view_scan(doctor, scan):
        return jsonify({'error': 'Access denied: Patient consent required for external center scans.'}), 403

    try:
        data = storage.load_bytes(scan.storage_key, kind)
        write_audit_log('scan.image.view', doctor=doctor, patient_id=scan.patient_id, scan_id=scan.id, details={'kind': kind})
        return send_file(io.BytesIO(data), mimetype='image/png')
    except FileNotFoundError:
        return jsonify({'error': f'Image {kind} not available for this scan'}), 404

@scans_bp.route('/api/scans/<public_id>/report', methods=['GET'])
@jwt_required()
def get_scan_report(public_id):
    doctor_id = get_jwt_identity()
    doctor = db.session.get(Doctor, int(doctor_id))
    scan = Scan.query.filter_by(public_id=public_id, is_deleted=False).first()
    if not scan:
        return jsonify({'error': 'Scan not found'}), 404

    if not can_view_scan(doctor, scan):
        return jsonify({'error': 'Access denied: Patient consent required for external center scans.'}), 403

    try:
        data = storage.load_bytes(scan.storage_key, 'report')
        write_audit_log('report.download', doctor=doctor, patient_id=scan.patient_id, scan_id=scan.id)
        return send_file(
            io.BytesIO(data),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"Report_{scan.public_id}.pdf"
        )
    except FileNotFoundError:
        return jsonify({'error': 'PDF report not found'}), 404

@scans_bp.route('/api/scans/<public_id>', methods=['PATCH'])
@jwt_required()
def update_scan(public_id):
    doctor_id = get_jwt_identity()
    doctor = db.session.get(Doctor, int(doctor_id))
    scan = Scan.query.filter_by(public_id=public_id, is_deleted=False).first()
    if not scan:
        return jsonify({'error': 'Scan not found'}), 404

    # Modifications strictly permitted only to doctors in the originating center
    if scan.center_id != doctor.center_id:
        return jsonify({'error': 'Only doctors at the originating center can modify notes or confirmed labels.'}), 403

    data = request.get_json() or {}
    if 'doctor_notes' in data:
        scan.doctor_notes = data['doctor_notes']
    if 'doctor_confirmed_label' in data:
        scan.doctor_confirmed_label = data['doctor_confirmed_label']

    db.session.commit()
    write_audit_log('scan.update', doctor=doctor, patient_id=scan.patient_id, scan_id=scan.id)

    return jsonify({
        'message': 'Scan record updated successfully',
        'doctor_notes': scan.doctor_notes,
        'doctor_confirmed_label': scan.doctor_confirmed_label
    })

