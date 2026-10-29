import json
from datetime import datetime, timezone
from flask import request
from db import db
from db.models import Consent, AuditLog, Scan, Doctor

def write_audit_log(action: str, doctor: Doctor = None, patient_id: int = None, scan_id: int = None, details: dict = None):
    try:
        actor_doctor_id = doctor.id if doctor else None
        actor_center_id = doctor.center_id if doctor else None
        ip = request.remote_addr if request else None
        user_agent = request.user_agent.string if request and request.user_agent else None
        details_str = json.dumps(details) if details else None

        entry = AuditLog(
            at=datetime.now(timezone.utc),
            actor_doctor_id=actor_doctor_id,
            actor_center_id=actor_center_id,
            action=action,
            patient_id=patient_id,
            scan_id=scan_id,
            ip=ip,
            user_agent=user_agent,
            details_json=details_str
        )
        db.session.add(entry)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Failed to record audit log: {e}")

def has_active_consent(patient_id: int, center_id: int) -> bool:
    """Checks whether grantee center has active, unexpired, non-revoked consent for patient."""
    now = datetime.now(timezone.utc)
    consent = Consent.query.filter_by(
        patient_id=patient_id,
        grantee_center_id=center_id,
        status='granted'
    ).filter(
        Consent.expires_at > now,
        Consent.revoked_at.is_(None)
    ).first()
    return consent is not None

def can_view_scan(doctor: Doctor, scan: Scan) -> bool:
    """
    Core authorization rule:
    1. Scans originating from the doctor's own center are immediately accessible.
    2. Scans from external centers strictly require an active, unexpired, non-revoked patient consent.
    Every denied attempt writes 'access.denied' to the append-only audit log.
    """
    if not doctor or not scan:
        return False

    # Own center check
    if scan.center_id == doctor.center_id:
        return True

    # External center consent check
    if has_active_consent(scan.patient_id, doctor.center_id):
        return True

    # Access Denied -> Audit violation
    write_audit_log(
        action='access.denied',
        doctor=doctor,
        patient_id=scan.patient_id,
        scan_id=scan.id,
        details={'reason': 'No active patient consent for external center', 'scan_center_id': scan.center_id}
    )
    return False

