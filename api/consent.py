import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from db import db
from db.models import Patient, Consent, Doctor
from api.access import write_audit_log
from config import Config

consent_bp = Blueprint('consent_bp', __name__)

# In-memory simulator store for OTPs in DEMO_MODE: phone -> list of recent OTPs
DEMO_OTP_INBOX: dict[str, list[dict]] = {}

@consent_bp.route('/api/patients/<uid>/consent/request', methods=['POST'])
@jwt_required()
def request_consent(uid):
    doctor_id = get_jwt_identity()
    doctor = db.session.get(Doctor, int(doctor_id))
    patient = Patient.query.filter_by(uid=uid.strip().upper()).first()
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    # Generate 6-digit OTP
    otp_code = f"{secrets.randbelow(900000) + 100000:06d}"
    otp_salt = secrets.token_hex(16)
    otp_hash = hashlib.sha256((otp_code + otp_salt).encode('utf-8')).hexdigest()

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=10)

    consent = Consent(
        patient_id=patient.id,
        grantee_center_id=doctor.center_id,
        requested_by_doctor_id=doctor.id,
        scope='all_centers_read',
        status='pending',
        otp_hash=otp_hash,
        otp_salt=otp_salt,
        otp_expires_at=expires_at,
        otp_attempts=0,
        created_at=now
    )
    db.session.add(consent)
    db.session.commit()

    # Deliver OTP (In demo mode, store in in-memory inbox)
    if Config.DEMO_MODE:
        phone_key = patient.phone or "default"
        if phone_key not in DEMO_OTP_INBOX:
            DEMO_OTP_INBOX[phone_key] = []
        DEMO_OTP_INBOX[phone_key].append({
            'otp': otp_code,
            'consent_id': consent.id,
            'patient_uid': patient.uid,
            'doctor_name': doctor.name,
            'center_name': doctor.center.name,
            'requested_at': now.isoformat()
        })
        print(f"[DEMO SIMULATOR OTP] For Patient {patient.uid} ({patient.phone}): {otp_code}")

    write_audit_log('consent.request', doctor=doctor, patient_id=patient.id, details={'consent_id': consent.id})

    # Return DPDP notice metadata
    return jsonify({
        'consent_id': consent.id,
        'message': f"OTP sent to patient's registered mobile number ending in {patient.phone[-4:] if patient.phone and len(patient.phone) >= 4 else 'XXXX'}.",
        'notice': {
            'requesting_doctor': doctor.name,
            'requesting_center': doctor.center.name,
            'scope': 'All historical chest radiographs across connected centers',
            'duration': '30 Days',
            'expires_in_minutes': 10
        }
    }), 201

@consent_bp.route('/api/patients/<uid>/consent/verify', methods=['POST'])
@jwt_required()
def verify_consent(uid):
    doctor_id = get_jwt_identity()
    doctor = db.session.get(Doctor, int(doctor_id))
    patient = Patient.query.filter_by(uid=uid.strip().upper()).first()
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    data = request.get_json() or {}
    consent_id = data.get('consent_id')
    otp_submitted = str(data.get('otp', '')).strip()

    consent = Consent.query.filter_by(id=consent_id, patient_id=patient.id).first()
    if not consent or consent.status != 'pending':
        return jsonify({'error': 'No pending consent request found.'}), 400

    now = datetime.now(timezone.utc)
    if consent.otp_expires_at.replace(tzinfo=timezone.utc) < now:
        consent.status = 'expired'
        db.session.commit()
        return jsonify({'error': 'OTP has expired. Please request a new consent code.'}), 400

    if consent.otp_attempts >= 5:
        consent.status = 'denied'
        db.session.commit()
        write_audit_log('consent.denied', doctor=doctor, patient_id=patient.id, details={'consent_id': consent.id, 'reason': 'Too many wrong OTP attempts'})
        return jsonify({'error': 'Maximum attempts exceeded. Consent request denied.'}), 403

    # Check OTP hash with salt
    computed_hash = hashlib.sha256((otp_submitted + consent.otp_salt).encode('utf-8')).hexdigest()
    if computed_hash != consent.otp_hash:
        consent.otp_attempts += 1
        db.session.commit()
        return jsonify({'error': f'Incorrect OTP. {5 - consent.otp_attempts} attempts remaining.'}), 400

    # OTP is valid -> Grant 30 days cross-center access
    consent.status = 'granted'
    consent.granted_at = now
    consent.expires_at = now + timedelta(days=30)
    db.session.commit()

    write_audit_log('consent.grant', doctor=doctor, patient_id=patient.id, details={'consent_id': consent.id, 'duration_days': 30})

    return jsonify({
        'message': 'Consent granted successfully. Cross-center scans are now unlocked for 30 days.',
        'consent': {
            'id': consent.id,
            'status': consent.status,
            'granted_at': consent.granted_at.isoformat(),
            'expires_at': consent.expires_at.isoformat()
        }
    })

@consent_bp.route('/api/consents/<int:consent_id>/revoke', methods=['POST'])
def revoke_consent(consent_id):
    """
    Patient-initiated consent revocation (DPDP right to revoke consent).
    Available from patient portal / phone simulator.
    """
    consent = db.session.get(Consent, consent_id)
    if not consent:
        return jsonify({'error': 'Consent record not found'}), 404

    now = datetime.now(timezone.utc)
    consent.status = 'revoked'
    consent.revoked_at = now
    db.session.commit()

    write_audit_log('consent.revoke', patient_id=consent.patient_id, details={'consent_id': consent.id})

    return jsonify({
        'message': 'Consent has been revoked immediately. Access to external center records is terminated.',
        'consent_id': consent.id,
        'status': 'revoked',
        'revoked_at': now.isoformat()
    })

