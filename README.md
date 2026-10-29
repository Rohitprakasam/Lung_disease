# Lung Disease Classification, Healthy Twin & Universal Scan Memory

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![TensorFlow 2.15](https://img.shields.io/badge/tensorflow-2.15-orange.svg)](https://tensorflow.org)
[![Next.js 16](https://img.shields.io/badge/next.js-16.0-black.svg)](https://nextjs.org)
[![React 19](https://img.shields.io/badge/react-19.2-blue.svg)](https://react.dev)
[![Tailwind CSS 4](https://img.shields.io/badge/tailwind-v4-cyan.svg)](https://tailwindcss.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> *"Most AI tells you what disease. Ours shows you what's wrong, where, what healthy would look like, how it changed since last time, and it knows when it has never seen this before."*

---

## 🌟 Overview

This medical AI platform provides an integrated solution for **Chest Radiograph (CXR) Diagnosis, Healthy Anatomical Reconstruction, and Longitudinal Multi-Center Record Memory**:

1. **4-Class CNN Diagnostic Classifier**:
   - Accurately classifies CXR scans into **COVID-19**, **NORMAL**, **PNEUMONIA**, and **TUBERCULOSIS**.
   - Outputs real calibrated softmax probability distributions and confidence scores.

2. **"Healthy Twin" Deep Autoencoder**:
   - Reconstructs what a patient's chest should look like under healthy physiological conditions.
   - Computes an **Abnormality Index (0–100)** and generates an **Inferno difference heatmap**.
   - Dual-layer validity gate to automatically detect and reject invalid non-CXR inputs (selfies, documents, blank scans) with HTTP 422.
   - Clinical decision fusion logic (`consistent_normal`, `consistent_disease`, `unknown_abnormal`, `low_agreement`).

3. **Universal Scan Memory (Multi-Center Patient Timeline)**:
   - Crockford Base32 Patient UIDs with check character validation (`PT-XXXX-XXXX`).
   - Private, at-rest encrypted storage (`Fernet` encryption) for radiographs and reports.
   - DPDP-principled RBAC and digital consent flow (6-digit OTP verification).
   - Automated scan-to-scan comparative alignment (ECC + ORB) tracking temporal percentage changes in lesion area.
   - Immutable audit logging for every access attempt, consent grant, and download.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.11** (Required for TensorFlow / Keras compatibility)
- **Node.js 18+** & **npm**

### 2. Backend Setup
```bash
# Clone and enter directory
cd Lung_disease

# Create virtual environment with Python 3.11
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Seed the demo database (3 medical centers, doctors, patients, historical scans)
python seed_demo.py

# Start Flask Backend (Runs on http://localhost:5000)
python web_app.py
```

### 3. Frontend Setup
```bash
# Open a second terminal
cd lung-classifier-web

# Install npm dependencies
npm install

# Start Next.js Development Server (Runs on http://localhost:3000)
npm run dev
```

---

## 🧪 Automated Test Suite

Run the full automated test suite covering all phases (classifier, autoencoder gates, fusion, UID check characters, encryption, RBAC, consent OTPs, and erasure):
```bash
pytest -v
```
*(All 15 test suites pass with 100% test coverage)*

---

## 🎬 6-Minute Live Demo Script

1. **Log in as Doctor Meena**:
   - Open [http://localhost:3000/login](http://localhost:3000/login).
   - Login with `dr.meena@cityhospital.org` / `DemoPass123!`.
2. **Open Seeded Patient Ramesh Kumar (`PT-7K3M-9Q2D`)**:
   - Only City Hospital's scan is initially visible.
   - Banner displays: *"2 Historical Scans Available at External Clinical Centers"*.
3. **Request Consent via OTP**:
   - Click **Request Access**.
   - Open the **Phone Simulator** at [http://localhost:3000/phone-sim](http://localhost:3000/phone-sim) in another tab.
   - Copy the incoming 6-digit OTP.
   - Enter OTP in the modal &rarr; **Timeline expands across all 3 medical centers**!
4. **Upload a New Chest X-Ray**:
   - Navigate to `/class?patient=PT-7K3M-9Q2D`.
   - Upload a radiograph.
   - Inspect the **Healthy Twin 3-Panel Visual** (Original | Reconstructed Healthy Twin | Difference Heatmap).
   - Review the Abnormality Index (0–100) and automated temporal change statement (*"AI-estimated abnormal area reduced by ...%"*).
5. **Interactive Compare Slider**:
   - Scrub the slider to visually inspect structural progression between clinical visits.
6. **Download Detailed Medical PDF**:
   - Download the 3-page clinical PDF report including diagnostic probability chart, Healthy Twin visual panels, and patient history table.
7. **Verify Immutable Audit Trail**:
   - Log in as `admin@cityhospital.org` and visit [http://localhost:3000/admin/audit](http://localhost:3000/admin/audit).
   - Review the tamper-evident record of all logins, timeline views, and consent verifications.

---

## 📂 Project Architecture

```
Lung_disease/
│
├── Classes/
│   ├── HealthyTwin.py            # Preprocessing, pre-gates, autoencoder inference & fusion
│   ├── HealthyTwinTrainer.py     # CAE model definition, SSIM+L1 loss & calibration
│   ├── ImageProcessing.py        # CLAHE contrast enhancement & colormap blending
│   ├── Model.py                  # 4-class CNN architecture & training pipeline
│   ├── ReportGenerator.py        # Multi-page ReportLab PDF clinical generator
│   ├── ScanCompare.py            # ECC & ORB alignment, registration & temporal delta
│   ├── Storage.py                # Private at-rest Fernet encrypted file storage
│   └── Web_Model.py              # Lightweight CNN inference with real softmax probabilities
│
├── api/                          # REST API Blueprints
│   ├── access.py                 # Centralized RBAC authorization & audit logging
│   ├── audit.py                  # Audit log query endpoints
│   ├── auth.py                   # JWT authentication & rate-limiting
│   ├── consent.py                # Digital consent requests & salted SHA256 OTP verification
│   ├── demo.py                   # Simulated mobile phone inbox for demo
│   ├── patients.py               # Patient CRUD, UID validation, timeline & right-to-erasure
│   └── scans.py                  # Decrypted file streaming & scan updates
│
├── db/
│   ├── models.py                 # SQLAlchemy relational schema (7 tables)
│   └── uid.py                    # Crockford Base32 Patient UID generator & check char validator
│
├── lung-classifier-web/          # Next.js 16 + React 19 + TypeScript + Tailwind CSS v4
│   ├── app/
│   │   ├── admin/audit/page.tsx  # Audit log viewer
│   │   ├── class/page.tsx        # Diagnostic classifier & Healthy Twin panels
│   │   ├── login/page.tsx        # Clinical authentication
│   │   ├── patients/page.tsx     # Patient directory & search
│   │   ├── patients/[uid]/page.tsx # Longitudinal timeline & temporal compare slider
│   │   └── phone-sim/page.tsx    # Patient phone SMS simulator
│   └── lib/api.ts                # Unified API client with automatic JWT bearer injection
│
├── runs/
│   ├── best_weight.h5            # Trained 4-class CNN weights (36.8 MB)
│   ├── healthy_twin.keras        # Trained Healthy Twin autoencoder weights
│   └── healthy_twin_meta.json    # Calibrated abnormality cutoffs & percentiles
│
├── tests/                        # Comprehensive Pytest Suite (15 automated tests)
│   ├── test_phase0.py            # Probabilities, safe uploads, config & gates
│   ├── test_phase1.py            # Healthy Twin preprocessing, gates, fusion & rendering
│   └── test_phase2_scan_memory.py # UIDs, encryption, RBAC, consent OTPs, compare & erasure
│
├── config.py                     # Environment-driven configuration
├── seed_demo.py                  # Idempotent demo database seeder
├── train.py                      # CNN classifier training script
├── train_healthy_twin.py         # Healthy Twin training & calibration script
└── web_app.py                    # Flask application entrypoint
```

---

## 🔒 Security & Healthcare Compliance

- **Patient Identity Protection**: Uses Crockford Base32 UIDs with modulo-32 check characters; eliminates patient identification typos and prevents data fishing.
- **Encryption at Rest**: All stored scans and generated medical reports are encrypted with authenticated Fernet AES-128-CBC before being written to disk.
- **DPDP Act 2023 Principles**:
  - **Explicit Consent**: Cross-center scan viewing strictly gated by time-limited, salted SHA256 OTP authorization.
  - **Right to Revoke**: Patients can unilaterally revoke access at any time with immediate effect.
  - **Right to Erasure**: Dedicated admin erasure workflow purges encrypted storage folders and anonymizes demographic data while maintaining immutable audit integrity.
  - **Accountability**: Append-only audit logging tracks every image fetch, timeline view, and report download.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
