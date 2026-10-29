import io
import os
import uuid
import json
from datetime import datetime, timezone, timedelta
import numpy as np
from PIL import Image

from config import Config
from db import db
from db.models import Center, Doctor, Patient, Scan, Consent, AuditLog
from db.uid import generate_patient_uid
from Classes.Storage import Storage
from Classes.HealthyTwin import HealthyTwin
from web_app import app

def seed():
    if not Config.DEMO_MODE:
        print("Refusing to seed: DEMO_MODE is false.")
        return

    with app.app_context():
        db.create_all()
        storage = Storage()
        password = os.getenv("SEED_PASSWORD", "DemoPass123!")

        print("--- Seeding Medical Centers ---")
        centers_data = [
            {"code": "CTY", "name": "City Hospital", "city": "Metro Central"},
            {"code": "GVC", "name": "Green Valley Clinic", "city": "North Valley"},
            {"code": "LKV", "name": "Lakeview Diagnostics", "city": "Lake District"}
        ]
        centers = {}
        for c_data in centers_data:
            c = Center.query.filter_by(code=c_data["code"]).first()
            if not c:
                c = Center(code=c_data["code"], name=c_data["name"], city=c_data["city"])
                db.session.add(c)
                db.session.flush()
            centers[c_data["code"]] = c

        print("--- Seeding Clinical Staff & Admins ---")
        doctors_data = [
            # City Hospital
            {"center": "CTY", "name": "Dr. Meena Iyer", "email": "dr.meena@cityhospital.org", "role": "doctor"},
            {"center": "CTY", "name": "Dr. Arun Varma", "email": "dr.arun@cityhospital.org", "role": "doctor"},
            {"center": "CTY", "name": "Admin CTY", "email": "admin@cityhospital.org", "role": "admin"},
            # Green Valley Clinic
            {"center": "GVC", "name": "Dr. Sarah Jenkins", "email": "dr.sarah@greenvalley.org", "role": "doctor"},
            {"center": "GVC", "name": "Dr. Rajesh Gupta", "email": "dr.rajesh@greenvalley.org", "role": "doctor"},
            {"center": "GVC", "name": "Admin GVC", "email": "admin@greenvalley.org", "role": "admin"},
            # Lakeview Diagnostics
            {"center": "LKV", "name": "Dr. Chen Wei", "email": "dr.chen@lakeview.org", "role": "doctor"},
            {"center": "LKV", "name": "Dr. Vikram Seth", "email": "dr.vikram@lakeview.org", "role": "doctor"},
            {"center": "LKV", "name": "Admin LKV", "email": "admin@lakeview.org", "role": "admin"},
        ]
        doctors = {}
        for d_data in doctors_data:
            d = Doctor.query.filter_by(email=d_data["email"]).first()
            if not d:
                d = Doctor(
                    center_id=centers[d_data["center"]].id,
                    name=d_data["name"],
                    email=d_data["email"],
                    role=d_data["role"]
                )
                d.set_password(password)
                db.session.add(d)
                db.session.flush()
            doctors[d_data["email"]] = d

        print("--- Seeding Patients ---")
        patients_data = [
            {"uid": "PT-7K3M-9Q2D", "name": "Ramesh Kumar", "birth_year": 1982, "sex": "Male", "phone": "+919876543210", "reg_center": "CTY"},
            {"uid": "PT-4H8N-2K9B", "name": "Anita Sharma", "birth_year": 1990, "sex": "Female", "phone": "+919876543211", "reg_center": "GVC"},
            {"uid": "PT-9W1Z-5M4P", "name": "David Wilson", "birth_year": 1975, "sex": "Male", "phone": "+919876543212", "reg_center": "LKV"},
            {"uid": "PT-3T6R-8Y7C", "name": "Priya Patel", "birth_year": 1995, "sex": "Female", "phone": "+919876543213", "reg_center": "CTY"},
            {"uid": "PT-2X5V-7J3F", "name": "Mohammed Ali", "birth_year": 1988, "sex": "Male", "phone": "+919876543214", "reg_center": "GVC"}
        ]
        patients = {}
        for p_data in patients_data:
            p = Patient.query.filter_by(uid=p_data["uid"]).first()
            if not p:
                p = Patient(
                    uid=p_data["uid"],
                    name=p_data["name"],
                    birth_year=p_data["birth_year"],
                    sex=p_data["sex"],
                    phone=p_data["phone"],
                    registered_center_id=centers[p_data["reg_center"]].id
                )
                db.session.add(p)
                db.session.flush()
            patients[p_data["uid"]] = p

        print("--- Seeding Longitudinal Scans & Encrypted Storage ---")
        # Patient 1 Ramesh: 3 scans across centers
        # 1. 2026-08-12 at GVC (Pneumonia)
        # 2. 2026-09-18 at LKV (Pneumonia moderate)
        # 3. 2026-10-25 at CTY (Normal / Recovered)
        p1 = patients["PT-7K3M-9Q2D"]
        if Scan.query.filter_by(patient_id=p1.id).count() == 0:
            scans_config = [
                {
                    "center": "GVC", "doctor": "dr.sarah@greenvalley.org",
                    "days_ago": 78, "label": "PNEUMONIA", "conf": 0.91,
                    "twin_idx": 84, "twin_status": "consistent_disease",
                    "notes": "Bilateral infiltrates detected in lower lobes."
                },
                {
                    "center": "LKV", "doctor": "dr.chen@lakeview.org",
                    "days_ago": 42, "label": "PNEUMONIA", "conf": 0.79,
                    "twin_idx": 68, "twin_status": "consistent_disease",
                    "notes": "Follow-up scan. Gradual clearing of pulmonary consolidation."
                },
                {
                    "center": "CTY", "doctor": "dr.meena@cityhospital.org",
                    "days_ago": 5, "label": "NORMAL", "conf": 0.94,
                    "twin_idx": 18, "twin_status": "consistent_normal",
                    "notes": "Full resolution. Normal lung parenchymal markings.",
                    "confirmed": "NORMAL"
                }
            ]

            now = datetime.now(timezone.utc)
            for s_cfg in scans_config:
                scan_id = uuid.uuid4().hex
                # Generate a synthetic radiographic image for the demo scan
                base_val = 140 if s_cfg["label"] == "NORMAL" else 80
                arr = np.random.randint(base_val - 30, base_val + 50, size=(256, 256), dtype=np.uint8)
                img = Image.fromarray(arr)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                raw_bytes = buf.getvalue()

                # Encrypt and save in private storage
                storage.save_bytes(scan_id, 'original', raw_bytes)
                storage.save_bytes(scan_id, 'enhanced', raw_bytes)
                storage.save_bytes(scan_id, 'twin', raw_bytes)
                storage.save_bytes(scan_id, 'diff', raw_bytes)
                storage.save_bytes(scan_id, 'triptych', raw_bytes)
                storage.save_bytes(scan_id, 'graph', raw_bytes)

                taken_time = now - timedelta(days=s_cfg["days_ago"])
                scan_row = Scan(
                    public_id=scan_id,
                    patient_id=p1.id,
                    center_id=centers[s_cfg["center"]].id,
                    doctor_id=doctors[s_cfg["doctor"]].id,
                    taken_at=taken_time,
                    uploaded_at=taken_time,
                    storage_key=scan_id,
                    classifier_label=s_cfg["label"],
                    classifier_confidence=s_cfg["conf"],
                    probabilities_json=json.dumps({s_cfg["label"]: s_cfg["conf"], "NORMAL": 0.05 if s_cfg["label"] != "NORMAL" else 0.94}),
                    twin_index=s_cfg["twin_idx"],
                    twin_status=s_cfg["twin_status"],
                    doctor_notes=s_cfg.get("notes"),
                    doctor_confirmed_label=s_cfg.get("confirmed")
                )
                db.session.add(scan_row)

        db.session.commit()
        print("Demo seed completed successfully!")
        print(f"Default login password for all seeded doctors and admins: {password}")

if __name__ == "__main__":
    seed()

