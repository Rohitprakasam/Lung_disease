from datetime import datetime, timezone
import json
from werkzeug.security import generate_password_hash, check_password_hash
from db import db

def utc_now():
    return datetime.now(timezone.utc)

class Center(db.Model):
    __tablename__ = 'centers'

    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(10), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    city = db.Column(db.String(80), nullable=False)
    created_at = db.Column(db.DateTime, default=utc_now)

    doctors = db.relationship('Doctor', backref='center', lazy=True)
    scans = db.relationship('Scan', backref='center', lazy=True)

class Doctor(db.Model):
    __tablename__ = 'doctors'

    id = db.Column(db.Integer, primary_key=True)
    center_id = db.Column(db.Integer, db.ForeignKey('centers.id'), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='doctor') # 'doctor' | 'admin'
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=utc_now)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    __tablename__ = 'patients'

    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column(db.String(20), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    birth_year = db.Column(db.Integer, nullable=True)
    sex = db.Column(db.String(10), default='Unknown')
    phone = db.Column(db.String(20), nullable=True)
    abha_id = db.Column(db.String(30), nullable=True) # Unverified ABHA mock
    registered_center_id = db.Column(db.Integer, db.ForeignKey('centers.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=utc_now)

    scans = db.relationship('Scan', backref='patient', lazy=True, order_by='Scan.taken_at.desc()')
    consents = db.relationship('Consent', backref='patient', lazy=True)

class Scan(db.Model):
    __tablename__ = 'scans'

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(36), unique=True, nullable=False, index=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False, index=True)
    center_id = db.Column(db.Integer, db.ForeignKey('centers.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    
    taken_at = db.Column(db.DateTime, default=utc_now, nullable=False)
    uploaded_at = db.Column(db.DateTime, default=utc_now)
    storage_key = db.Column(db.String(64), nullable=False)
    
    classifier_label = db.Column(db.String(50), nullable=False)
    classifier_confidence = db.Column(db.Float, nullable=False)
    probabilities_json = db.Column(db.Text, nullable=False)
    
    twin_raw_score = db.Column(db.Float, nullable=True)
    twin_index = db.Column(db.Integer, nullable=True)
    twin_status = db.Column(db.String(50), nullable=True)
    
    model_version = db.Column(db.String(50), default='1.0')
    twin_version = db.Column(db.String(50), default='1.0')
    doctor_notes = db.Column(db.Text, nullable=True)
    doctor_confirmed_label = db.Column(db.String(50), nullable=True)
    report_storage_key = db.Column(db.String(64), nullable=True)
    is_deleted = db.Column(db.Boolean, default=False)

    __table_args__ = (
        db.Index('idx_patient_taken_at', 'patient_id', 'taken_at'),
    )

    @property
    def probabilities(self) -> dict:
        try:
            return json.loads(self.probabilities_json)
        except Exception:
            return {}

class Consent(db.Model):
    __tablename__ = 'consents'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False, index=True)
    grantee_center_id = db.Column(db.Integer, db.ForeignKey('centers.id'), nullable=False)
    requested_by_doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    scope = db.Column(db.String(50), default='all_centers_read')
    status = db.Column(db.String(20), default='pending') # 'pending'|'granted'|'denied'|'expired'|'revoked'
    
    otp_hash = db.Column(db.String(64), nullable=True)
    otp_salt = db.Column(db.String(32), nullable=True)
    otp_expires_at = db.Column(db.DateTime, nullable=True)
    otp_attempts = db.Column(db.Integer, default=0)
    
    granted_at = db.Column(db.DateTime, nullable=True)
    expires_at = db.Column(db.DateTime, nullable=True)
    revoked_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=utc_now)

class AuditLog(db.Model):
    __tablename__ = 'audit_log'

    id = db.Column(db.Integer, primary_key=True)
    at = db.Column(db.DateTime, default=utc_now, nullable=False, index=True)
    actor_doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=True)
    actor_center_id = db.Column(db.Integer, nullable=True)
    action = db.Column(db.String(50), nullable=False, index=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=True)
    scan_id = db.Column(db.Integer, db.ForeignKey('scans.id'), nullable=True)
    ip = db.Column(db.String(50), nullable=True)
    user_agent = db.Column(db.String(256), nullable=True)
    details_json = db.Column(db.Text, nullable=True)

class ScanComparison(db.Model):
    __tablename__ = 'scan_comparisons'

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(db.Integer, db.ForeignKey('scans.id'), nullable=False)
    previous_scan_id = db.Column(db.Integer, db.ForeignKey('scans.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=utc_now)
    alignment_method = db.Column(db.String(20), default='none') # 'ecc' | 'orb' | 'none'
    alignment_quality = db.Column(db.Float, default=0.0)
    index_delta = db.Column(db.Integer, default=0)
    area_delta_pct = db.Column(db.Float, default=0.0)
    trend = db.Column(db.String(20), default='stable') # 'improved'|'worsened'|'stable'|'unreliable'
    message = db.Column(db.Text, nullable=True)
    diff_storage_key = db.Column(db.String(64), nullable=True)

