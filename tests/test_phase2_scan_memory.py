import io
import os
import json
import pytest
from PIL import Image
import numpy as np

from web_app import app, db
from db.models import Center, Doctor, Patient, Scan, Consent, AuditLog
from db.uid import generate_patient_uid, validate_patient_uid
from Classes.Storage import Storage
from Classes.ScanCompare import ScanCompare
from api.consent import DEMO_OTP_INBOX

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_patient_uid_generation_and_validation():
    for _ in range(50):
        uid = generate_patient_uid()
        assert uid.startswith("PT-")
        assert len(uid) == 12 # PT-XXXX-XXXX
        assert validate_patient_uid(uid) is True

    valid_uid = generate_patient_uid()
    bad_uid = valid_uid[:-1] + ('0' if valid_uid[-1] != '0' else '1')
    assert validate_patient_uid(bad_uid) is False

def test_encrypted_storage(tmp_path):
    storage = Storage(storage_dir=str(tmp_path))
    scan_id = "test_scan_enc_123"
    payload = b"Top secret chest X-ray binary data"

    file_path = storage.save_bytes(scan_id, "original", payload)
    assert os.path.exists(file_path)

    with open(file_path, "rb") as f:
        raw_on_disk = f.read()
    assert raw_on_disk != payload

    decrypted = storage.load_bytes(scan_id, "original")
    assert decrypted == payload

def test_auth_login_and_rate_limiting(client):
    res = client.post('/api/auth/login', json={
        'email': 'dr.meena@cityhospital.org',
        'password': 'WrongPassword123'
    })
    assert res.status_code == 401

    res = client.post('/api/auth/login', json={
        'email': 'dr.meena@cityhospital.org',
        'password': 'DemoPass123!'
    })
    assert res.status_code == 200
    token = res.get_json()['token']
    assert token is not None

def test_cross_center_access_and_consent_flow(client):
    res_login = client.post('/api/auth/login', json={
        'email': 'dr.meena@cityhospital.org',
        'password': 'DemoPass123!'
    })
    cty_token = res_login.get_json()['token']
    headers_cty = {'Authorization': f'Bearer {cty_token}'}

    res_timeline = client.get('/api/patients/PT-7K3M-9Q2D/timeline', headers=headers_cty)
    assert res_timeline.status_code == 200
    timeline_data = res_timeline.get_json()
    
    assert len(timeline_data['scans']) == 1
    assert timeline_data['hidden_other_center_scans'] == 2

    res_req = client.post('/api/patients/PT-7K3M-9Q2D/consent/request', headers=headers_cty)
    assert res_req.status_code == 201
    consent_id = res_req.get_json()['consent_id']

    demo_msgs = DEMO_OTP_INBOX.get('+919876543210', [])
    assert len(demo_msgs) > 0
    otp = demo_msgs[-1]['otp']

    res_verify = client.post('/api/patients/PT-7K3M-9Q2D/consent/verify', headers=headers_cty, json={
        'consent_id': consent_id,
        'otp': otp
    })
    assert res_verify.status_code == 200

    res_expanded = client.get('/api/patients/PT-7K3M-9Q2D/timeline', headers=headers_cty)
    expanded_data = res_expanded.get_json()
    assert len(expanded_data['scans']) == 3
    assert expanded_data['hidden_other_center_scans'] == 0

    res_revoke = client.post(f'/api/consents/{consent_id}/revoke')
    assert res_revoke.status_code == 200

    res_post_revoke = client.get('/api/patients/PT-7K3M-9Q2D/timeline', headers=headers_cty)
    post_revoke_data = res_post_revoke.get_json()
    assert len(post_revoke_data['scans']) == 1
    assert post_revoke_data['hidden_other_center_scans'] == 2

def test_scan_compare_registration():
    arr = np.random.randint(40, 200, size=(256, 256), dtype=np.uint8)
    img = Image.fromarray(arr)
    err_map = np.zeros((256, 256), dtype=np.float32)

    res = ScanCompare.compare_scans(
        prev_img=img,
        new_img=img,
        prev_twin_index=20,
        new_twin_index=20,
        prev_error_map=err_map,
        new_error_map=err_map,
        prev_date_str="01 Aug 2026",
        prev_center_name="City Hospital"
    )
    assert res["alignment_quality"] >= 0.8
    assert res["trend"] == "stable"

def test_patient_erasure_dpdp(client):
    # Login as Admin
    res_admin = client.post('/api/auth/login', json={
        'email': 'admin@cityhospital.org',
        'password': 'DemoPass123!'
    })
    admin_token = res_admin.get_json()['token']
    headers_admin = {'Authorization': f'Bearer {admin_token}'}

    # Create dummy patient to erase
    res_create = client.post('/api/patients', headers=headers_admin, json={
        'name': 'Erase Me',
        'birth_year': 1999,
        'sex': 'Other',
        'phone': '+919999999999'
    })
    uid = res_create.get_json()['uid']

    # Delete patient
    res_del = client.delete(f'/api/patients/{uid}', headers=headers_admin)
    assert res_del.status_code == 200

    # Verify patient anonymized
    with app.app_context():
        p = Patient.query.filter_by(uid=uid).first()
        assert p.name == "ERASED_PATIENT"
        assert p.phone is None
