import time
from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from datetime import timedelta
from db.models import Doctor
from api.access import write_audit_log

auth_bp = Blueprint('auth_bp', __name__)

# Simple in-memory rate limiting for failed logins: email -> [timestamps]
_FAILED_ATTEMPTS: dict[str, list[float]] = {}
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_WINDOW_SECONDS = 600 # 10 minutes

@auth_bp.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    now = time.time()
    recent_failures = [t for t in _FAILED_ATTEMPTS.get(email, []) if now - t < LOCKOUT_WINDOW_SECONDS]
    _FAILED_ATTEMPTS[email] = recent_failures

    if len(recent_failures) >= MAX_FAILED_ATTEMPTS:
        write_audit_log('login.locked', details={'email': email, 'reason': 'Too many failed login attempts'})
        return jsonify({'error': 'Account temporarily locked due to excessive failed attempts. Please retry later.'}), 429

    doctor = Doctor.query.filter_by(email=email, is_active=True).first()
    if not doctor or not doctor.check_password(password):
        _FAILED_ATTEMPTS[email].append(now)
        write_audit_log('login.failed', details={'email': email})
        return jsonify({'error': 'Invalid email or password.'}), 401

    # Clear failed attempts on successful login
    _FAILED_ATTEMPTS.pop(email, None)

    token = create_access_token(
        identity=str(doctor.id),
        additional_claims={
            'doctor_id': doctor.id,
            'center_id': doctor.center_id,
            'role': doctor.role,
            'name': doctor.name
        },
        expires_delta=timedelta(hours=8)
    )

    write_audit_log('login', doctor=doctor, details={'email': email})

    return jsonify({
        'token': token,
        'doctor': {
            'id': doctor.id,
            'name': doctor.name,
            'email': doctor.email,
            'role': doctor.role,
            'center': {
                'id': doctor.center.id,
                'name': doctor.center.name,
                'code': doctor.center.code,
                'city': doctor.center.city
            }
        }
    })

@auth_bp.route('/api/auth/me', methods=['GET'])
@jwt_required()
def get_current_user():
    doctor_id = get_jwt_identity()
    doctor = db.session.get(Doctor, int(doctor_id))
    if not doctor:
        return jsonify({'error': 'Doctor not found'}), 404

    return jsonify({
        'id': doctor.id,
        'name': doctor.name,
        'email': doctor.email,
        'role': doctor.role,
        'center': {
            'id': doctor.center.id,
            'name': doctor.center.name,
            'code': doctor.center.code,
            'city': doctor.center.city
        }
    })

