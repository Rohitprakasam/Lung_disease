from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from db import db
from db.models import Patient, Scan, Doctor, Consent
from db.uid import generate_patient_uid, validate_patient_uid
from Classes.Storage import Storage
from api.access import can_view_scan, write_audit_log, has_active_consent

patients_bp = Blueprint('patients_bp', __name__)
storage = Storage()

@patients_bp.route('/api/patients', methods=['POST'])
@jwt_required()
def create_patient():
    doctor_id = get_jwt_identity()
    doctor = db.session.get(Doctor, int(doctor_id))
    if not doctor:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json() or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Patient name is required'}), 400

    # Generate unique UID with check character
    uid = generate_patient_uid()
    while Patient.query.filter_by(uid=uid).first():
        uid = generate_patient_uid()

    patient = Patient(
        uid=uid,
        name=name,
        birth_year=data.get('birth_year'),
        sex=data.get('sex', 'Unknown'),
        phone=data.get('phone', '').strip(),
        abha_id=data.get('abha_id', '').strip() or None,
        registered_center_id=doctor.center_id
    )
    db.session.add(patient)
    db.session.commit()

    write_audit_log('patient.create', doctor=doctor, patient_id=patient.id, details={'uid': uid})

    return jsonify({
        'id': patient.id,
        'uid': patient.uid,
        'name': patient.name,
        'birth_year': patient.birth_year,
        'sex': patient.sex,
        'phone': patient.phone,
        'abha_id': patient.abha_id,
        'registered_center_id': patient.registered_center_id
    }), 201

@patients_bp.route('/api/patients', methods=['GET'])
@jwt_required()
def search_patients():
    doctor_id = get_jwt_identity()
    doctor = db.session.get(Doctor, int(doctor_id))
    if not doctor:
        return jsonify({'error': 'Unauthorized'}), 401

    uid_query = request.args.get('uid', '').strip()
    name_query = request.args.get('name', '').strip()

    if uid_query:
        # Cross-center lookup strictly allowed by exact UID with check char validation
        patient = Patient.query.filter_by(uid=uid_query.upper()).first()
        if not patient:
            # Check without formatting hyphens
            for p in Patient.query.all():
                if p.uid.replace('-', '').upper() == uid_query.replace('-', '').upper():
                    patient = p
                    break
        if not patient:
            return jsonify({'patients': []})
        return jsonify({'patients': [{
            'id': patient.id,
            'uid': patient.uid,
            'name': patient.name,
            'birth_year': patient.birth_year,
            'sex': patient.sex,
            'registered_center': patient.registered_center_id
        }]})

    # Name search is strictly scoped to doctor's own registered center (prevents data fishing)
    if name_query:
        patients = Patient.query.filter(
            Patient.registered_center_id == doctor.center_id,
            Patient.name.ilike(f"%{name_query}%")
        ).limit(20).all()
    else:
        # Recent patients for this center
        patients = Patient.query.filter_by(registered_center_id=doctor.center_id).order_by(Patient.created_at.desc()).limit(20).all()

    return jsonify({'patients': [{
        'id': p.id,
        'uid': p.uid,
        'name': p.name,
        'birth_year': p.birth_year,
        'sex': p.sex
    } for p in patients]})

@patients_bp.route('/api/patients/<uid>/timeline', methods=['GET'])
@jwt_required()
def get_patient_timeline(uid):
    doctor_id = get_jwt_identity()
    doctor = db.session.get(Doctor, int(doctor_id))
    if not doctor:
        return jsonify({'error': 'Unauthorized'}), 401

    patient = Patient.query.filter_by(uid=uid.strip().upper()).first()
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    # Audit timeline view
    write_audit_log('timeline.view', doctor=doctor, patient_id=patient.id, details={'uid': uid})

    all_scans = Scan.query.filter_by(patient_id=patient.id, is_deleted=False).order_by(Scan.taken_at.desc()).all()
    
    permitted_scans = []
    hidden_other_center_scans = 0

    for scan in all_scans:
        if can_view_scan(doctor, scan):
            permitted_scans.append({
                'id': scan.id,
                'public_id': scan.public_id,
                'taken_at': scan.taken_at.isoformat(),
                'center_id': scan.center_id,
                'center_name': scan.center.name,
                'center_code': scan.center.code,
                'classifier_label': scan.classifier_label,
                'classifier_confidence': round(scan.classifier_confidence, 4),
                'probabilities': scan.probabilities,
                'twin_index': scan.twin_index,
                'twin_status': scan.twin_status,
                'doctor_confirmed_label': scan.doctor_confirmed_label,
                'doctor_notes': scan.doctor_notes
            })
        else:
            hidden_other_center_scans += 1

    active_consent = has_active_consent(patient.id, doctor.center_id)

    return jsonify({
        'patient': {
            'id': patient.id,
            'uid': patient.uid,
            'name': patient.name,
            'birth_year': patient.birth_year,
            'sex': patient.sex,
            'phone': patient.phone,
            'has_active_consent': active_consent
        },
        'scans': permitted_scans,
        'hidden_other_center_scans': hidden_other_center_scans
    })

@patients_bp.route('/api/patients/<uid>', methods=['DELETE'])
@jwt_required()
def delete_patient(uid):
    """
    DPDP Right to Erasure implementation.
    Soft-deletes all patient scan rows, permanently purges encrypted storage folders,
    blanks out identifiable patient fields, and preserves immutable audit entries.
    """
    doctor_id = get_jwt_identity()
    doctor = db.session.get(Doctor, int(doctor_id))
    if not doctor or doctor.role != 'admin':
        return jsonify({'error': 'Administrative privileges required for patient data erasure.'}), 403

    patient = Patient.query.filter_by(uid=uid.strip().upper()).first()
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    # Purge encrypted files on disk for all scans
    for s in patient.scans:
        storage.delete_scan(s.storage_key)
        s.is_deleted = True

    # Anonymize patient demographic record
    patient.name = "ERASED_PATIENT"
    patient.phone = None
    patient.abha_id = None
    patient.birth_year = None

    db.session.commit()
    write_audit_log('patient.erasure', doctor=doctor, patient_id=patient.id, details={'uid': uid, 'reason': 'DPDP Right to Erasure exercised'})

    return jsonify({'message': f'Patient data and associated scans for {uid} have been permanently erased.'})
