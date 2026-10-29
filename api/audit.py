from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from db.models import Doctor, AuditLog, Patient

audit_bp = Blueprint('audit_bp', __name__)

@audit_bp.route('/api/audit', methods=['GET'])
@jwt_required()
def get_audit_logs():
    doctor_id = get_jwt_identity()
    doctor = db.session.get(Doctor, int(doctor_id))
    if not doctor or doctor.role != 'admin':
        return jsonify({'error': 'Administrative privileges required to access audit trail.'}), 403

    query = AuditLog.query.filter_by(actor_center_id=doctor.center_id)

    action = request.args.get('action')
    if action:
        query = query.filter_by(action=action)

    patient_uid = request.args.get('patient_uid')
    if patient_uid:
        p = Patient.query.filter_by(uid=patient_uid.strip().upper()).first()
        if p:
            query = query.filter_by(patient_id=p.id)
        else:
            return jsonify({'logs': []})

    logs = query.order_by(AuditLog.at.desc()).limit(100).all()

    return jsonify({'logs': [{
        'id': l.id,
        'at': l.at.isoformat(),
        'action': l.action,
        'actor_doctor_id': l.actor_doctor_id,
        'patient_id': l.patient_id,
        'scan_id': l.scan_id,
        'ip': l.ip,
        'details': l.details_json
    } for l in logs]})

