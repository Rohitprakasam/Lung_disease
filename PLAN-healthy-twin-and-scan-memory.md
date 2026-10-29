# PLAN: "Healthy Twin" + "Universal Scan Memory"

> **Audience:** the worker who implements this. Read top to bottom once, then work phase by phase.
> **Author role:** analysis/guide only. No code in this repo was changed while writing this plan.
> **Repo:** `Lung_disease/` (git, branch `main`, remote `Rohitprakasam/Lung_disease`)
> **Date:** 2026-10-30

---

## 0. The story in one page (end-to-end narrative)

Today a doctor opens the Next.js page `/class`, types a free-text Patient ID, name, age and history, uploads a chest X-ray, and gets back **one word** (`PNEUMONIA`) plus a PDF. The PDF claims "95% confidence" for every scan because the confidence is hard-coded. The scan is saved in a public folder (`static/uploads/`) and nothing links it to the patient's earlier scans.

After this plan, the same visit looks like this:

1. **Dr. Meena logs in** at *City Hospital* (center A). She is a doctor account tied to that center.
2. She searches for the patient by **Patient UID** `PT-7K3M-9Q2D` (or registers a new patient, who gets a fresh UID).
3. The patient's own scans from *City Hospital* are visible straight away. The system says *"2 more scans exist at other centers. Request access?"*
4. She clicks **Request access**. The patient gets a **6-digit OTP** on their phone (in the demo, it shows on a "patient phone" simulator page). The patient reads it out, she enters it, and a **consent** record is written: *center A may view this patient's scans from all centers for 30 days*.
5. She uploads today's X-ray. The backend:
   - checks that the image is a plausible chest X-ray (**validity gate**),
   - runs the existing 4-class CNN and now returns **real probabilities**,
   - runs the **Healthy Twin** autoencoder, which reconstructs what this chest "should" look like if it were healthy, and builds a **difference heatmap** plus an **abnormality index (0–100)**,
   - fuses both into a clear message (e.g. *"Classifier: PNEUMONIA 88%. Abnormality index 81/100. Findings consistent."*), or flags *"Abnormal, pattern not in known classes, refer to a specialist"* when the CNN says NORMAL but the twin disagrees,
   - stores the scan under the patient UID in private storage, never in a public folder,
   - **auto-compares** with the patient's most recent previous scan: aligns the two images, compares the abnormality maps, and writes *"AI-estimated abnormal area reduced by 34% since 12 Aug 2026 (Green Valley Clinic)"*.
6. The results page shows **three panels** (Original | Healthy Twin | Difference), the classifier bar chart, the abnormality index, and a **timeline** with thumbnails, a trend line and a **compare slider** against any earlier scan.
7. The **PDF** now has three pages: (1) the existing diagnosis page with real numbers, (2) "AI Visual Analysis" with the three panels and the index, (3) "Scan History" with the trend table and chart. Every AI image is labelled *"AI-generated reference, not a diagnosis."*
8. Every view of the timeline, every image fetch and every PDF download writes an **audit_log** row. The demo shows the audit log and the consent record, which is what makes the project look credible.

Pitch line: *"Most AI tells you what disease. Ours shows you what's wrong, where, what healthy would look like, how it changed since last time, and it knows when it has never seen this before."*

---

## 1. What exists today (analysis of current code)

| Area | File | Facts that matter for this plan |
| :--- | :--- | :--- |
| Classifier training | `Classes/Model.py`, `train.py` | Keras `Sequential` CNN, input **224×224×3 RGB**, `Rescaling(1/255)`, classes `['COVID-19','NORMAL','PNEUMONIA','TUBERCULOSIS']`, data loaded with `utils.image_dataset_from_directory(<data_dir>/{train,val,test})`. Paths hard-coded to `LungDiseaseClassification/...` (wrong for this repo). **`test_dataset` is never normalized** (only train/val are mapped through `Rescaling`), so any evaluation on it is wrong. |
| Classifier inference | `Classes/Web_Model.py` | `preprocess_image`: PIL resize to 224×224, `/255.0`, add batch dim. `predict` computes softmax but **returns only the class name**. |
| API | `web_app.py` | One route `POST /api/predict`. Saves upload as `static/uploads/<timestamp>_<file.filename>` (**user-controlled filename, path-traversal risk**). Calls `ImageProcessing.enhance_image` (CLAHE + Turbo blend, for display only). **Confidence and probabilities are mocked** (`0.95` / `0.05`). Builds graph + PDF in `static/reports/`, returns `{prediction, report_url}`. `CORS(app)` open to all. No auth, no DB. |
| Image processing | `Classes/ImageProcessing.py` | CLAHE (`clipLimit=2.0`, `tileGridSize=(8,8)`) then Turbo colormap 30% blend. Reusable for normalizing images before comparison (CLAHE part only). |
| PDF | `Classes/ReportGenerator.py` | ReportLab `canvas`, **single letter page with hard-coded y-coordinates**. Images row at `y_pos -= 250`, precautions below. There is **no room** for more images on page 1. `determine_stage` maps confidence to "stage", which is meaningless with mocked confidence. |
| Frontend | `lung-classifier-web/app/class/page.tsx` | Single client component. Fetches `http://localhost:5000/api/predict` (hard-coded). Shows prediction text + PDF link. Patient ID is free text. Next 16.0.8, React 19.2.1, Tailwind 4, no other deps. |
| Legacy UI | `Templates/*.html` | Old Jinja pages. Not used by `/api/predict`. Leave alone. |
| Weights | `runs/best_weight.h5` (~37 MB) | Legacy HDF5. |
| Environment | — | No `requirements.txt`. `__pycache__` shows **Python 3.9** was used. The machine's default `python` is 3.14 with **no TensorFlow**; TF does not support 3.14, so a dedicated venv is required. |
| Dataset | not in repo | Expected layout `Data/{train,val,test}/{COVID-19,NORMAL,PNEUMONIA,TUBERCULOSIS}`. README says 4,244 / 1,818 / 1,070 images. Worker must obtain it locally. |

**Root problems this plan must fix along the way** (they block the new features):
1. Mocked probabilities: fusion logic and trend charts need real numbers.
2. Public, unauthenticated storage of patient images and PDFs: incompatible with a "scan memory" across centers.
3. Hard-coded API URL in the frontend.
4. Unsafe upload filenames.
5. Single-page PDF with fixed coordinates.

---

## 2. Decisions (and alternatives considered)

### 2.1 Healthy Twin model type

| Option | Verdict | Why |
| :--- | :--- | :--- |
| **Convolutional autoencoder with a narrow bottleneck, no skip connections** | **Chosen** | Forced to compress, so it can only rebuild what it learned (healthy anatomy). Abnormal regions reconstruct poorly, which is exactly the signal we want. Small, CPU-friendly. |
| U-Net with skip connections | Rejected (as-is) | Skip connections let the network **copy the input straight through**, including the disease. The "twin" would look like the original and the difference map would be empty. If the worker wants U-Net quality later, use it only with skips removed or heavily dropped out at the top levels. |
| VAE | Optional later | Smoother latent space, slightly blurrier output. Not needed for v1. |
| GAN / diffusion inpainting | Rejected for now | Better-looking twins, but training is unstable, needs GPU, and is far beyond project scope. Mention as "future work" in the report. |

### 2.2 Abnormality score

- **Not** the plain mean error over the whole image (borders, text markers and shoulders dominate it).
- **Chosen:** mean of the **top 5% pixel errors inside a central chest region of interest (ROI)**, computed on a smoothed error map. Converted to an **Abnormality Index 0–100** = percentile of this raw score against the distribution of NORMAL validation scans (so 50 = typical healthy, 95+ = more abnormal than 95% of healthy scans). Doctors understand percentiles.

### 2.3 Invalid image detection

The idea doc says a selfie "fails to reconstruct and gets rejected." **This is only partly true.** Autoencoders are known to reconstruct simple images (blank pages, flat colours, dark images) very well, so reconstruction error alone is **not** a reliable invalid-image detector. Use **two layers**:
1. **Cheap pre-gate** (before any model): colour saturation (X-rays are grayscale, so channel difference is near zero), minimum size, aspect ratio, and intensity spread.
2. **Reconstruction gate**: raw score above an `invalid` threshold calibrated on a small "junk" set.
Either gate failing → HTTP 422 *"This does not look like a frontal chest X-ray."* No report is generated.

### 2.4 Database

| Option | Verdict |
| :--- | :--- |
| **SQLAlchemy ORM + SQLite for demo, same code on PostgreSQL for "production"** | **Chosen.** Switching is a single `DATABASE_URL` change. No server to install for the demo. |
| Raw `sqlite3` | Rejected: no migrations, harder to move to PostgreSQL. |
| PostgreSQL from day 1 | Allowed if the team already has it (Docker). Not required. |

Migrations: **Flask-Migrate (Alembic)**. Never edit tables by hand.

### 2.5 Patient identity (answers the open question "ABHA vs generated UID")

**Chosen: generated Patient UID now, with a nullable `abha_id` column for later.**
- Format: `PT-XXXX-XXXX` using Crockford base32 (no `I`, `L`, `O`, `U`) + 1 check character, e.g. `PT-7K3M-9Q2D`. Check character catches typos, which matters because **wrong patient match is the most dangerous failure** in this feature.
- Real ABHA integration requires ABDM sandbox registration and HIP/HIU certification; out of scope for a student project. Keep the column, show it in the UI as "ABHA ID (optional, not verified)".
- **Do not store Aadhaar numbers.** The Aadhaar Act restricts storing them. Phone number + OTP is enough for the demo.

### 2.6 Auth

- Doctor login with **email + password** (hashed with `werkzeug.security.generate_password_hash`, already installed with Flask), issuing a **JWT** (`flask-jwt-extended`) that carries `doctor_id`, `center_id`, `role`.
- Token lifetime 8 h. Frontend keeps it in memory + `sessionStorage` (simple for demo). Note in the report that production should use httpOnly cookies.
- Roles: `doctor`, `admin` (admin can view audit log for their own center).

### 2.7 Image storage

- **Chosen:** private folder `storage/` (outside `static/`), one folder per scan, files **encrypted at rest with Fernet** (`cryptography` package, key from env `STORAGE_KEY`). Served only through authenticated endpoints that check consent and write the audit log.
- Alternative: MinIO/S3. Design the storage code behind a tiny interface (`save_bytes`, `load_bytes`, `delete`) so MinIO can be swapped in later. Not required for demo.

### 2.8 Scan-to-scan comparison

Raw pixel subtraction of two X-rays taken months apart, on different machines, mostly shows **positioning and exposure differences, not disease change**. So:
- **Primary trend signal:** compare the **Healthy Twin anomaly maps** of the two scans (each already normalized against "healthy"), after aligning them.
- **Secondary:** classifier probabilities over time (trend chart).
- **Alignment:** OpenCV **ECC** (`cv2.findTransformECC`, affine) on CLAHE-normalized downscaled images; fallback to **ORB + RANSAC** if ECC does not converge; if both fail or alignment quality is poor, show the side-by-side slider but **no** "reduced/increased" message ("Alignment not reliable, compare visually").

---

## 3. Target architecture

```
Next.js (lung-classifier-web)
  /login            -> POST /api/auth/login
  /patients         -> GET  /api/patients?query=   , POST /api/patients
  /patients/[uid]   -> GET  /api/patients/<uid>/timeline
                       POST /api/patients/<uid>/consent/request|verify
                       GET  /api/scans/<id>/image/<kind>
                       GET  /api/scans/<id>/compare/<other_id>
  /class            -> POST /api/predict   (now needs token + patient_uid)
  /phone-sim        -> GET  /api/demo/otp-inbox   (DEMO ONLY)
  /admin/audit      -> GET  /api/audit

Flask (web_app.py + blueprints)
  config.py         env-driven settings, feature flags
  Classes/
    Web_Model.py          (+ real probabilities)
    HealthyTwin.py        NEW  inference: reconstruct, maps, score, gates
    HealthyTwinTrainer.py NEW  model definition + training + calibration
    ScanCompare.py        NEW  registration + map comparison + message
    Storage.py            NEW  encrypted file store
    ReportGenerator.py    (+ page 2 visual analysis, page 3 history)
    ImageProcessing.py    (+ normalize_for_compare helper)
  db/
    models.py             NEW  SQLAlchemy tables
    __init__.py           NEW  db, migrate objects
  api/
    auth.py, patients.py, scans.py, consent.py, audit.py, demo.py   NEW blueprints
  train_healthy_twin.py   NEW  root script, like train.py
  seed_demo.py            NEW  3 centers, doctors, patients, historical scans
  migrations/             NEW  Alembic
  storage/                NEW  (gitignored) encrypted scans
  runs/healthy_twin.keras         NEW weights
  runs/healthy_twin_meta.json     NEW thresholds + preprocessing version
```

---

## 4. Implementation steps

Work in **phases**. Each phase is its own git branch and its own PR/merge, and each ends with its verification checklist (section 7). Do not start a phase before the previous one passes.

### Phase 0 — Foundation fixes (prerequisite, ~1 day)

**Branch:** `feat/p0-foundation`

0.1 **Create a venv with a TensorFlow-supported Python.**
   - Use Python **3.10 or 3.11** (3.9 also works, it is what the repo was built with).
   - `py -3.11 -m venv venv` → activate → install: `tensorflow` (2.15.x if you want guaranteed `.h5` + `keras` 2 behaviour; 2.16+ ships Keras 3, which usually still loads legacy `.h5` but test it), `flask`, `flask-cors`, `pillow`, `opencv-python`, `matplotlib`, `reportlab`, `numpy`.
   - Confirm `runs/best_weight.h5` loads with the chosen version **before** anything else. If it fails on Keras 3, pin `tensorflow==2.15.*`.
   - Freeze into **`requirements.txt`** at repo root (pinned versions). Add `venv/` is already in `.gitignore`.

0.2 **Return real probabilities from `Web_Model`.**
   - Add a new method `predict_with_probs(image)` returning `(predicted_class, confidence, probabilities_dict)` where `probabilities_dict = {class_name: float(p)}` from the softmax row.
   - Keep `predict(image)` working (call the new method and return only the class) so nothing else breaks.
   - In `web_app.py`, delete the mocked block (`confidence = 0.95`, the `probabilities` dict) and use the real values. `ReportGenerator.generate_graph/generate_pdf` already accept these shapes; no signature change needed.

0.3 **Safe upload names.**
   - Replace `f"{timestamp}_{file.filename}"` with a UUID-based name and keep only a whitelisted extension (`.png`, `.jpg`, `.jpeg`). Use `werkzeug.utils.secure_filename` only for logging the original name.
   - Reject files larger than e.g. 15 MB (`app.config['MAX_CONTENT_LENGTH']`).

0.4 **Config module.** Create `config.py` reading env vars with defaults:
   `DATABASE_URL` (default `sqlite:///lung.db`), `JWT_SECRET_KEY`, `STORAGE_DIR` (`storage`), `STORAGE_KEY`, `HEALTHY_TWIN_ENABLED` (`true`), `SCAN_MEMORY_ENABLED` (`false` until Phase 2 is done), `ALLOW_ANONYMOUS_PREDICT` (`true` until Phase 3), `CORS_ORIGINS` (`http://localhost:3000`), `DEMO_MODE` (`true`).
   Use `python-dotenv` to load `.env` (already gitignored). Add `.env.example` with every key and dummy values.

0.5 **Frontend API URL.** In `app/class/page.tsx` replace `'http://localhost:5000'` with `process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:5000'`. Add `lung-classifier-web/.env.local.example`. Put the base URL in one helper file `lung-classifier-web/lib/api.ts` that every page will use later.

0.6 **Fix training script paths** (`train.py`, `Model.py`): data dir from CLI arg / env, default `Data`; `os.makedirs("runs", exist_ok=True)`; also map `test_dataset` through the same `Rescaling`. (Needed because Phase 1 reuses the same data layout.)

0.7 **Restrict CORS** to `CORS_ORIGINS`, and allow the `Authorization` header (needed from Phase 2).

---

### Phase 1 — Healthy Twin (~4–6 days)

**Branch:** `feat/p1-healthy-twin`

#### 1A. Data preparation (decisions the worker must follow exactly)

1.1 **Preprocessing function — one single function used by training AND inference.** Put it in `Classes/HealthyTwin.py` as a static method `preprocess(pil_or_path) -> np.ndarray (H, W, 1) float32 in [0,1]`:
   1. Load → convert to **grayscale** (`L`).
   2. Apply **CLAHE** (`clipLimit=2.0, tileGridSize=(8,8)`, same as `ImageProcessing`) — this is the brightness/contrast normalization across machines.
   3. Resize to **256×256** (`cv2.INTER_AREA`). (256 divides cleanly by 2⁴ = 16 for the encoder; 224 would also work with 4 stride-2 layers → 14×14. Pick 256 and record it in meta.)
   4. Scale to `[0,1]` float32.
   Record `"preprocess_version": 1, "size": 256, "clahe": [2.0, 8]` in `runs/healthy_twin_meta.json`. Inference must refuse to run if meta's `preprocess_version` differs from code's constant.

1.2 **Training data = NORMAL only**: `Data/train/NORMAL`. Validation for early stopping: `Data/val/NORMAL`. Do **not** let any non-NORMAL image into training (check folder counts and print them).

1.3 **Calibration sets** (used only after training):
   - `Data/val/NORMAL` → healthy score distribution.
   - `Data/val/{COVID-19,PNEUMONIA,TUBERCULOSIS}` → abnormal distribution (for reporting AUROC).
   - **Junk set**: create `Data/junk/` with ~50–100 non-CXR images (selfies, documents, CT slices, lateral X-rays, hand X-rays, blank/black images). Do not commit them if license unclear; keep local. Used to set the `invalid` threshold.

1.4 **Augmentation** (training only, light — we want a model of *healthy anatomy*, not of distortions): random shift ±5%, rotation ±5°, zoom ±5%, brightness ±10%. No horizontal flips (heart is on the left; flipping teaches anatomically wrong "healthy").

#### 1B. Model (`Classes/HealthyTwinTrainer.py`)

1.5 Architecture (functional Keras, grayscale 256×256×1):
   - Encoder: `Conv2D(32,3,stride 2) → BN → LeakyReLU` → 64 → 128 → 256 (each stride 2) → output 16×16×256.
   - **Bottleneck:** `Conv2D(16, 1)` → **16×16×16 = 4,096 values** (≈1/16 of input). This narrowness is the whole point; if twins come out too sharp and copy disease, reduce to 8 channels; if too blurry to be useful, raise to 32.
   - Decoder: mirror with `Conv2DTranspose(…, stride 2)` 256 → 128 → 64 → 32 → `Conv2D(1, 3, activation='sigmoid')`.
   - **No skip connections.**
1.6 Loss: `0.84 * (1 - SSIM) + 0.16 * L1` (standard mix; SSIM via `tf.image.ssim`, max_val=1.0). Optimizer Adam 1e-3, ReduceLROnPlateau, EarlyStopping(patience=8, restore_best_weights=True), up to 100 epochs, batch 32.
1.7 Save as **`runs/healthy_twin.keras`** (native Keras format, not `.h5`).
1.8 Root script **`train_healthy_twin.py`** (mirrors `train.py` style): args `--data Data --epochs 100 --out runs`. Runs training, then calibration (1.9), then writes meta JSON, then saves a `runs/healthy_twin_samples.png` grid (8 NORMAL + 8 diseased: original / twin / diff) for eyeballing.

#### 1C. Score + calibration

1.9 **ROI mask:** a fixed elliptical/rectangular mask covering the central chest (e.g. x 10–90%, y 10–92% of the 256 image), stored as a constant. Excludes corners where hospital text markers and edges live. (Future upgrade: a real lung segmentation model; out of scope.)
1.10 **Error map:** `err = |x − x̂|` ⊕ `(1 − SSIM_map)` averaged 50/50 → Gaussian blur (σ ≈ 4 px) → multiply by ROI mask.
1.11 **Raw score:** mean of the top 5% values of `err` inside ROI.
1.12 Calibration writes into meta JSON:
   - `normal_scores_sorted`: the sorted list (or 101 percentiles) of raw scores on val NORMAL → used to convert any raw score into the **Abnormality Index** (percentile, 0–100).
   - `thr_abnormal`: 95th percentile of NORMAL val scores.
   - `thr_invalid`: chosen so ≥ 90% of junk set is above it **and** ≤ 1% of all val CXRs (all 4 classes) are above it. If no such value exists, say so in the meta (`"invalid_gate_reliable": false`) and rely only on the pre-gate.
   - Report metrics: **AUROC** NORMAL-vs-abnormal on val, AUROC CXR-vs-junk. Print and store them.
   - **Acceptance target:** NORMAL-vs-abnormal AUROC ≥ 0.80. If below, tune bottleneck width / epochs before integrating. Below 0.70 means the feature should be shown as "experimental" only.

#### 1D. Inference class (`Classes/HealthyTwin.py`)

1.13 `class HealthyTwin:` loaded once at app start (like `Web_Model`):
   - `__init__(model_path, meta_path)` – load model + meta, check `preprocess_version`.
   - `pre_gate(pil_image) -> (ok: bool, reason: str)` – checks: min side ≥ 200 px; aspect ratio 0.6–1.6; mean |R−G|+|G−B| < small threshold (grayscale); grayscale std dev > threshold (not blank). Thresholds as class constants.
   - `analyze(pil_image) -> dict` returns:
     ```
     {
       "valid": bool, "invalid_reason": str|None,
       "raw_score": float, "abnormality_index": int (0-100),
       "is_abnormal": bool,             # raw_score > thr_abnormal
       "input_256": np.ndarray, "twin_256": np.ndarray, "error_map": np.ndarray
     }
     ```
   - `render_panels(result, out_dir, stem) -> dict of paths`: writes `twin.png` (grayscale), `diff.png` (error map → `COLORMAP_INFERNO` blended 50% over the CLAHE input, masked to ROI), and `triptych.png` (Original | Healthy Twin | Difference, labelled, with a footer line *"AI-generated reference, not a diagnosis"*). Upscale outputs to 512 for display.

#### 1E. Fusion logic (in `HealthyTwin.py` as a pure function `fuse(classifier_label, classifier_conf, twin_result) -> dict`)

| Classifier says | Twin says | Status code | Message shown |
| :--- | :--- | :--- | :--- |
| NORMAL | not abnormal | `consistent_normal` | "No abnormality detected by either model." |
| Disease X | abnormal | `consistent_disease` | "Findings consistent: X, abnormal regions highlighted." |
| NORMAL | abnormal | `unknown_abnormal` | "Abnormal pattern not matching known classes. Refer to a specialist." |
| Disease X | not abnormal | `low_agreement` | "Classifier suggests X but the image is close to healthy anatomy. Clinical review advised." |
| any | invalid | `invalid_image` | "This does not look like a frontal chest X-ray." (HTTP 422) |

Also: if `classifier_conf < 0.5`, append "Low classifier confidence."

#### 1F. Integrate in `web_app.py`

1.14 At startup: if `HEALTHY_TWIN_ENABLED` and weights exist → `twin = HealthyTwin(...)`; else `twin = None` and log a warning (app must still run without the twin).
1.15 In `/api/predict`, **order**:
   1. Save upload (Phase 0 naming).
   2. `twin.pre_gate` → 422 if fails.
   3. Classifier `predict_with_probs`.
   4. `twin.analyze` → 422 if `valid == False`.
   5. `fuse(...)`.
   6. `render_panels(...)` into the same per-request folder.
   7. Graph + PDF (1.17).
   8. Response JSON adds:
      ```
      "confidence": 0.88, "probabilities": {...},
      "twin": { "abnormality_index": 81, "is_abnormal": true,
                "status": "consistent_disease", "message": "...",
                "images": { "original": url, "twin": url, "diff": url, "triptych": url } }
      ```
      (In Phase 1 the URLs are still under `static/` for simplicity; Phase 2 moves them behind auth.)
1.16 Wrap the twin step in its own `try/except`: a twin failure must **not** fail the classification; return `"twin": null` and a warning.

#### 1G. PDF (`ReportGenerator.py`)

1.17 Add an optional parameter `twin=None` at the **end** of `generate_pdf(...)` signature (keeps old calls valid). If present, after page 1 call `c.showPage()` and draw **page 2 "AI Visual Analysis"**:
   - Triptych image full width.
   - Abnormality Index as a big number + a horizontal 0–100 bar with the 95th-percentile marker.
   - Fusion status message (colour: green/amber/red).
   - Box: *"The Healthy Twin is an AI-generated reconstruction of normal anatomy. It is a visual aid, not a clinical ground truth or diagnosis."*
   - Same footer disclaimer.
1.18 Fix `determine_stage`: now that confidence is real, rename the output label from "Stage N" to **"Model certainty: low / moderate / high"**. Classifier confidence is not disease severity; calling it "Stage 4 Critical" is clinically misleading. (This is a wording change; keep colours.)

#### 1H. Frontend (`app/class/page.tsx`)

1.19 Show: prediction + **confidence %**, probability bars (simple Tailwind divs, no chart lib), the **three panels** side by side (stack on mobile), Abnormality Index badge, and the fusion message with colour. Handle HTTP 422 by showing the invalid-image message in the error area. Under the panels, small grey caption: "AI-generated reference, not a diagnosis."

---

### Phase 2 — Universal Scan Memory: data, auth, storage (~5–7 days)

**Branch:** `feat/p2-scan-memory-core`

2.1 Install: `flask-sqlalchemy`, `flask-migrate`, `flask-jwt-extended`, `cryptography`, `python-dotenv`. Add to `requirements.txt`.

2.2 **Restructure Flask into blueprints** (`api/` folder). `web_app.py` becomes: create app, load config, init db/jwt/cors, load models, register blueprints. Keep `/api/predict` URL unchanged.

2.3 **Schema** (`db/models.py`). All IDs: integer PK internally + public UUID/UID where exposed. Timestamps UTC.

```
centers
  id PK, code (unique, e.g. "CTY"), name, city, created_at

doctors
  id PK, center_id FK→centers, name, email (unique), password_hash,
  role ('doctor'|'admin'), is_active, created_at

patients
  id PK, uid (unique, 'PT-XXXX-XXXX'), name, birth_year, sex,
  phone (E.164, for OTP), abha_id (nullable, unverified),
  registered_center_id FK→centers, created_at

scans
  id PK, public_id (UUID, unique), patient_id FK, center_id FK, doctor_id FK,
  taken_at (date the X-ray was taken; default upload time), uploaded_at,
  storage_key (folder name in storage/),
  classifier_label, classifier_confidence, probabilities_json,
  twin_raw_score, twin_index, twin_status,
  model_version (classifier weights hash), twin_version (meta version),
  doctor_notes (text), doctor_confirmed_label (nullable; doctor's own label),
  report_storage_key, is_deleted (soft delete), 
  INDEX (patient_id, taken_at)

consents
  id PK, patient_id FK, grantee_center_id FK, requested_by_doctor_id FK,
  scope ('all_centers_read'), status ('pending'|'granted'|'denied'|'expired'|'revoked'),
  otp_hash, otp_expires_at, otp_attempts,
  granted_at, expires_at (granted_at + 30 days), revoked_at, created_at

audit_log   (append-only: no UPDATE/DELETE in code)
  id PK, at, actor_doctor_id FK (nullable for system), actor_center_id,
  action ('login','patient.create','scan.create','timeline.view','scan.image.view',
          'report.download','consent.request','consent.grant','consent.deny',
          'consent.revoke','compare.view','access.denied'),
  patient_id (nullable), scan_id (nullable), ip, user_agent, details_json

scan_comparisons   (cache)
  id PK, scan_id FK, previous_scan_id FK, created_at,
  alignment_method ('ecc'|'orb'|'none'), alignment_quality float,
  index_delta int, area_delta_pct float, trend ('improved'|'worsened'|'stable'|'unreliable'),
  message, diff_storage_key
```

2.4 `flask db init` → `flask db migrate -m "initial scan memory schema"` → review the generated file by eye → `flask db upgrade`. Commit `migrations/`.

2.5 **Patient UID generator** (`db/uid.py`): random 8 Crockford-base32 chars + 1 check char (mod-37 or Luhn-mod-N). `validate_uid(s)` normalizes case/hyphens and checks the check char. Unit-test it.

2.6 **Storage** (`Classes/Storage.py`): `save(scan_key, name, bytes)`, `load(scan_key, name) -> bytes`, `delete_scan(scan_key)`. Encrypt with Fernet using `STORAGE_KEY`. Files: `storage/<scan_public_id>/{original.png, enhanced.png, twin.png, diff.png, triptych.png, graph.png, report.pdf}`. Add `storage/` and `*.db` to `.gitignore`.

2.7 **Auth endpoints** (`api/auth.py`): `POST /api/auth/login` → JWT; `GET /api/auth/me`. Log `login` in audit. Rate-limit naïvely (5 failed attempts / 10 min per email) — in-memory dict is fine for demo.

2.8 **Access rule — one function used everywhere** (`api/access.py`):
   ```
   can_view_scan(doctor, scan):
       scan.center_id == doctor.center_id                      -> allowed (own center)
       active consent exists (patient, doctor.center_id, granted, not expired, not revoked) -> allowed
       else -> denied (write audit 'access.denied')
   ```
   Every endpoint that returns scan data or image bytes must call this. **No exceptions.** This is the most important line of code in Phase 2; review it twice.

2.9 **Patient endpoints** (`api/patients.py`):
   - `POST /api/patients` (create, returns uid).
   - `GET /api/patients?uid=` (exact UID lookup; **no fuzzy name search across centers**, to avoid wrong-patient matches and data fishing). Name search allowed only within the doctor's own center.
   - `GET /api/patients/<uid>/timeline` → list of scans the doctor may see + a count of hidden scans from other centers (`hidden_other_center_scans: 2`) so the UI can offer consent. Audit `timeline.view`.

2.10 **Predict endpoint change**: accepts `patient_uid` (form field). If JWT present → create a `scans` row, store files in `Storage`, return `scan_id`. If no JWT and `ALLOW_ANONYMOUS_PREDICT=true` → old behaviour (Phase 1). Validate `patient_uid` with the check char; unknown UID → 404 (never auto-create on predict).

2.11 **Image/report endpoints** (`api/scans.py`): `GET /api/scans/<public_id>` (metadata), `GET /api/scans/<public_id>/image/<kind>` (decrypt + stream, `kind` whitelisted), `GET /api/scans/<public_id>/report` (PDF), `PATCH /api/scans/<public_id>` (doctor notes / confirmed label; only same-center doctor). Each call checks `can_view_scan` and writes audit. Frontend fetches images with the token and turns them into blob URLs (plain `<img src>` cannot send the header).

2.12 **Migrate Phase 1 URLs**: when scan memory is on, the predict response returns API image URLs, not `static/` URLs. Stop writing patient files to `static/`.

---

### Phase 3 — Consent flow (~2–3 days)

**Branch:** `feat/p3-consent`

3.1 `POST /api/patients/<uid>/consent/request` → creates `consents` row `pending`, generates 6-digit OTP with `secrets.randbelow`, stores **hash only** (`hashlib.sha256(otp + per-row salt)`), expiry 10 min, attempts 0. "Sends" the OTP:
   - `DEMO_MODE=true`: put it in an in-memory "inbox" readable at `GET /api/demo/otp-inbox?phone=` (and print to console). Disable this route entirely when `DEMO_MODE=false`.
   - Real SMS (MSG91/Twilio) is future work; keep a `send_otp(phone, otp)` function as the single hook.
3.2 `POST /api/patients/<uid>/consent/verify {consent_id, otp}` → max 5 attempts; on success status `granted`, `granted_at`, `expires_at=+30 days`; audit `consent.grant`. On too many attempts → `denied`.
3.3 `POST /api/consents/<id>/revoke` (patient-side action; in demo, available on the phone-sim page) → `revoked`; access stops immediately.
3.4 The consent screen must show **who** is asking (doctor name, center name), **what** (all scans from all centers), **for how long** (30 days). This is the DPDP "notice" element; show it in the demo.
3.5 After this phase, set `ALLOW_ANONYMOUS_PREDICT=false` in `.env.example` (anonymous mode stays available only by explicit config).

---

### Phase 4 — Auto-compare (~3–4 days)

**Branch:** `feat/p4-compare`

4.1 `Classes/ScanCompare.py`:
   - `normalize_for_compare(img)`: grayscale → CLAHE → resize 512 (add as a helper in `ImageProcessing.py`, reusing the existing CLAHE parameters).
   - `register(fixed, moving) -> (warp_matrix, method, quality)`:
     1. ECC: `cv2.findTransformECC(fixed, moving, warp, cv2.MOTION_AFFINE, criteria=(EPS|COUNT, 200, 1e-5), None, 5)`; downscale to 256 first for speed, then scale the translation terms back up.
     2. If ECC throws / does not converge → ORB (2000 features) + BFMatcher Hamming + `cv2.estimateAffinePartial2D` with RANSAC.
     3. `quality` = normalized cross-correlation of fixed vs warped moving inside ROI. If `< 0.6` → `method='none'`, `trend='unreliable'`.
   - `compare(prev_scan, new_scan)`:
     - Re-run (or load cached) twin error maps for both, warp the **previous** map onto the new scan's frame with the same matrix.
     - `area_prev` / `area_new` = fraction of ROI pixels where error > `thr_pixel` (pixel-level threshold = 95th percentile of NORMAL val pixel errors; add this to meta in Phase 1 calibration, or compute here once and store).
     - `area_delta_pct = (area_new − area_prev) / max(area_prev, ε) × 100`; `index_delta = new.twin_index − prev.twin_index`.
     - Trend rules: improved if `area_delta_pct ≤ −15` **and** `index_delta ≤ −5`; worsened if `≥ +15` and `≥ +5`; else stable; unreliable if alignment failed.
     - Change map image: green where abnormality decreased, red where it increased, over the new scan.
     - Message template: *"AI-estimated abnormal area {reduced/increased} by {abs}% since {prev.taken_at:%d %b %Y} ({prev.center.name})."* or *"No meaningful change since …"* or *"Scans could not be aligned reliably; compare visually."*
4.2 When a new scan is created, compare automatically with the **most recent previous scan the current doctor is allowed to see**. If the only earlier scans are at other centers without consent, skip and say "Earlier scans at other centers are available with patient consent."
4.3 Store the result in `scan_comparisons` (cache). `GET /api/scans/<id>/compare/<other_id>` computes on demand for any pair the doctor can see both of (both must pass `can_view_scan`).
4.4 **Personal baseline (Healthy Twin goes personal):** if the patient has an earlier scan with `doctor_confirmed_label == 'NORMAL'` (doctor-confirmed, not just AI), also show "Difference vs patient's own healthy scan from {date}" using the registered raw images (CLAHE-normalized) as an extra panel. Only use doctor-confirmed scans as baselines; an AI-only NORMAL could be wrong.
4.5 Add to predict response: `"comparison": {previous_scan_id, trend, message, images: {change_map}}`.

---

### Phase 5 — Frontend timeline, PDF history, seed data (~4–5 days)

**Branch:** `feat/p5-timeline-ui`

5.1 **Pages** (App Router):
   - `app/login/page.tsx` – email/password → store token (`lib/auth.ts`: in-memory + sessionStorage; `apiFetch()` adds `Authorization`, on 401 redirect to `/login`).
   - `app/patients/page.tsx` – lookup by UID (input with live check-char validation), create patient form, list of own-center recent patients.
   - `app/patients/[uid]/page.tsx` – **timeline**:
     - header: patient UID, name, birth year, consent status chip ("Own center only" / "All centers until 29 Nov 2026"),
     - "N scans at other centers – Request access" button → consent modal (shows who/what/how long) → OTP input,
     - vertical timeline cards: date, center name, thumbnail, classifier label + %, abnormality index, trend arrow vs previous,
     - **trend chart**: abnormality index over time + classifier top-class probability (inline SVG polyline; no new dependency needed — or add `recharts` if preferred),
     - **compare slider**: pick any two scans → two aligned images stacked, a range input controls a CSS `clip-path` on the top one; toggle to show the change map.
     - "Upload new scan" button → goes to `/class?patient=<uid>`.
   - `app/class/page.tsx` – when `patient` query param is present, hide name/age/history inputs (they come from the patient record), show patient header, and after prediction show the Phase 1 panels + the comparison message + link to timeline.
   - `app/phone-sim/page.tsx` (DEMO only, hidden if `NEXT_PUBLIC_DEMO_MODE != 'true'`) – enter phone → see OTP inbox → "Revoke consent" button.
   - `app/admin/audit/page.tsx` – table of audit rows for the admin's center (filters: patient UID, action, date).
   - Keep the current dark/cyan look; reuse classes from the current page.
5.2 **PDF page 3 "Scan History"** (`ReportGenerator.generate_pdf(..., history=None)` – another optional trailing parameter):
   - Table: date | center | classifier label (%) | abnormality index | trend.
   - Trend chart PNG (matplotlib, same style as `generate_graph`) of index over time.
   - Comparison message + change-map thumbnail for the latest pair.
   - Only include scans the requesting doctor is allowed to see; write "N additional scans not shown (no consent)" if applicable.
   - PDF metadata footer: "Generated for {doctor} at {center} on {date}. Contains personal health data – handle per DPDP Act 2023."
5.3 **Seed script `seed_demo.py`** (idempotent, refuses to run if `DEMO_MODE=false`):
   - 3 centers: *City Hospital (CTY)*, *Green Valley Clinic (GVC)*, *Lakeview Diagnostics (LKV)*.
   - 2 doctors per center + 1 admin; passwords from env `SEED_PASSWORD` (default printed once to console; **record it in `.env.example` comment, not in chat or README**).
   - 5 patients with 2–4 scans each spread across centers and dates (e.g. Aug→Oct 2026), using **test-set images** from `Data/test/` so stories are plausible: e.g. patient 1 = PNEUMONIA → PNEUMONIA → NORMAL (improving); patient 2 = NORMAL → TUBERCULOSIS (worsening); patient 3 = NORMAL baseline confirmed by doctor, then COVID-19.
   - Runs the real models on each seeded image so stored scores are genuine.
   - Note: different images of the "same patient" are actually different people, so alignment quality will vary — expected for a demo; mention it in the demo script.

---

### Phase 6 — Hardening and docs (~2 days)

**Branch:** `feat/p6-hardening`

6.1 Retention: management command `flask purge-legacy-static` that deletes old `static/uploads` / `static/reports` files older than N days (legacy anonymous mode).
6.2 Patient erasure (DPDP right to erasure): admin endpoint `DELETE /api/patients/<uid>` → soft-delete scans, delete storage folders, keep audit rows (with `patient_id` retained but personal fields of `patients` blanked). Confirm with a typed UID in UI.
6.3 Security headers, `MAX_CONTENT_LENGTH`, JWT secret must not be default when `DEMO_MODE=false` (fail fast at startup).
6.4 Update `README.md`: setup (venv, requirements, `.env`), training both models, migrations, seeding, running both servers, demo script. Update `PROJECT_ANALYSIS.md` architecture diagram.
6.5 Add `tests/` with pytest (see section 7).

---

## 5. Migration notes

### 5.1 Ordering (must follow)
`P0 → P1 → P2 → P3 → P4 → P5 → P6`. P1 can be demoed alone. P2 depends on P0 (config, safe names). P4 depends on P1 (twin maps) **and** P2 (stored scans). P5 depends on all.

### 5.2 Breaking changes

| Change | Who breaks | Mitigation |
| :--- | :--- | :--- |
| `/api/predict` response gains fields | Nobody (additive) | Frontend ignores unknown fields. |
| `/api/predict` returns **422** for invalid images | Clients expecting 200/500 | Frontend handles 422 in P1 (step 1.19). |
| `/api/predict` requires JWT + `patient_uid` | Current `/class` page, legacy templates | `ALLOW_ANONYMOUS_PREDICT` flag; default `true` until P3, then `false`. |
| Files move from `static/` to encrypted `storage/` | Old PDF links | Old links keep working (legacy files untouched); new scans only in `storage/`. |
| `generate_pdf` signature | Internal callers | New params are optional and trailing. |
| "Stage 1–4" wording → "Model certainty" | Report readers | Intentional; note in changelog. |
| Python/TF environment pinned | Anyone using another env | `requirements.txt` + README. |

### 5.3 Existing data
- Old files in `static/uploads` / `static/reports` are **not** linked to any patient (free-text IDs, no verification). **Do not auto-import them** into `scans` — that would create unverified patient links, which is the exact danger this feature must avoid. Leave them, purge later with 6.1.

### 5.4 Rollback strategy
- Every phase is a separate branch/merge → `git revert` the merge commit to roll back a phase.
- Feature flags: `HEALTHY_TWIN_ENABLED=false` makes the app behave like P0; `SCAN_MEMORY_ENABLED=false` hides all new blueprints and the frontend shows only `/class` (frontend reads `NEXT_PUBLIC_SCAN_MEMORY_ENABLED`).
- DB: before every `flask db upgrade`, copy `lung.db` → `lung.db.bak-<date>`. Every migration must have a working `downgrade()`; test `flask db downgrade` once per migration on a copy.
- Model files: keep previous `healthy_twin.keras` + meta as `*.prev`; the meta carries `twin_version`, stored on each scan, so you always know which model produced which score.
- `STORAGE_KEY` loss = all stored images unreadable. Back it up outside the repo. Never rotate it without a re-encryption script.

---

## 6. Risks and how to handle them

| Risk | Impact | Handling |
| :--- | :--- | :--- |
| Autoencoder reconstructs disease too well (twin ≈ original) | Empty heatmaps, useless feature | No skip connections; narrow bottleneck; check `healthy_twin_samples.png`; AUROC gate ≥ 0.80. |
| Autoencoder reconstructs junk well (blank images) | Invalid images accepted | Pre-gate (2.3); junk calibration; `invalid_gate_reliable` flag. |
| Heatmap highlights text markers / edges / devices (pacemakers, ECG leads) | False "abnormal" | ROI mask; mention limitation in report; future: lung segmentation. |
| Dataset bias (NORMAL images from one source, disease from others) | Twin learns "source", not "health" | Check sample grid per class for brightness/border differences; CLAHE helps; state limitation. |
| Wrong patient match | Clinically dangerous | UID with check char; exact-UID lookup across centers; no auto-create; show name + birth year confirmation before upload. |
| Consent bypass by a missed endpoint | Privacy breach | Single `can_view_scan`; tests for every endpoint with a no-consent doctor (7.3). |
| Misleading trend messages from bad alignment | Wrong clinical impression | Alignment quality gate; "unreliable" state; cautious wording ("AI-estimated"). |
| Classifier "stage" wording | Misleading | Changed in 1.18. |
| DPDP compliance claims | Over-claiming | Say "designed with DPDP principles (consent, purpose, audit, erasure)", not "compliant". Not legal advice. |
| TF / Python version mismatch | App won't start | Phase 0.1 first. |
| CPU latency (classifier + twin + compare + PDF) | Slow demo | Twin at 256² is ~tens of ms on CPU; ECC at 256² is fast; measure total and target < 5 s per upload. Load models once at startup. |

---

## 7. Verification checklist

Tick every box before calling a phase done. Reviewer (me) will re-check these.

### 7.0 Phase 0
- [ ] `pip install -r requirements.txt` in a fresh venv works; `python web_app.py` starts; `best_weight.h5` loads.
- [ ] Upload the same image twice → identical probabilities; probabilities sum to ~1.0; confidence ≠ 0.95 for most images.
- [ ] PDF bar chart shows real values.
- [ ] Upload a file named `../../evil.png` → stored under a UUID name inside `static/uploads`.
- [ ] Frontend works with `NEXT_PUBLIC_API_URL` set to another port.
- [ ] `python train.py --data Data` runs at least 1 epoch without path errors.

### 7.1 Phase 1 – Healthy Twin
- [ ] Training log shows **only NORMAL** image counts.
- [ ] `runs/healthy_twin.keras` + `runs/healthy_twin_meta.json` exist; meta has size, preprocess_version, thresholds, AUROCs.
- [ ] AUROC NORMAL vs abnormal (val) ≥ 0.80 — number recorded in PR description.
- [ ] `healthy_twin_samples.png`: NORMAL diffs are mostly dark; diseased diffs light up in lung fields (eyeball and attach to PR).
- [ ] Selfie, document photo, black image, CT slice → HTTP 422 with message, no PDF created.
- [ ] Classifier NORMAL + high twin score case shows "unknown abnormal" message (find one in val set or create by testing a TB image the classifier misses).
- [ ] Killing the twin (rename weights) → app still classifies, `"twin": null`.
- [ ] PDF page 2 renders triptych, index bar, disclaimer; page 1 unchanged except "Model certainty".
- [ ] Frontend shows 3 panels, index, message; mobile width stacks panels.
- [ ] Unit tests: `preprocess` output shape/range; `fuse` returns the right status for all 5 rows of the table.

### 7.2 Phase 2 – Core
- [ ] `flask db upgrade` on empty DB creates all 7 tables; `flask db downgrade` removes them.
- [ ] UID generator: 10,000 UIDs unique; single-character typo always rejected by `validate_uid`.
- [ ] Files in `storage/` are not readable as PNG/PDF without decryption (open one in an image viewer → fails).
- [ ] Nothing new appears in `static/` when scan memory is on.
- [ ] Login wrong password ×6 → blocked; audit shows attempts.
- [ ] Doctor A (center CTY) can see CTY scans of patient P without consent.

### 7.3 Phase 3 – Consent (security tests, all automated in pytest)
- [ ] Doctor B (center GVC) without consent: timeline shows only GVC scans + hidden count; **every** scan endpoint (`/scans/<id>`, `/image/<kind>`, `/report`, `/compare/...`) for a CTY scan returns 403 and writes `access.denied`.
- [ ] Wrong OTP ×5 → consent `denied`; correct OTP after expiry → rejected.
- [ ] Correct OTP → B sees all scans; audit has `consent.request` + `consent.grant`.
- [ ] Revoke → B immediately gets 403 again.
- [ ] Consent expiry: set `expires_at` in the past in the DB → access denied.
- [ ] OTP never stored in plain text (inspect DB).
- [ ] `DEMO_MODE=false` → `/api/demo/otp-inbox` returns 404.

### 7.4 Phase 4 – Compare
- [ ] Same image vs itself shifted 20 px and rotated 3° → ECC aligns, quality > 0.9, trend `stable`.
- [ ] Two unrelated images (different people) → quality low → `unreliable` message (never "reduced/increased").
- [ ] Seeded improving patient → `improved` message with correct previous date + center name.
- [ ] Compare endpoint denies when doctor cannot see **either** scan.
- [ ] Personal baseline panel appears only when a previous scan has `doctor_confirmed_label == 'NORMAL'`.

### 7.5 Phase 5 – UI / PDF / Seed
- [ ] `python seed_demo.py` twice → no duplicates.
- [ ] Full demo script (section 8) runs start to finish without touching the DB manually.
- [ ] Timeline thumbnails load via authenticated blob URLs (no `static/` URLs in network tab).
- [ ] Compare slider works with mouse and touch.
- [ ] PDF page 3 lists only permitted scans and shows "N not shown" otherwise.
- [ ] `npm run build` and `npm run lint` pass.

### 7.6 Phase 6
- [ ] Patient erase → storage folders gone, audit rows remain.
- [ ] App refuses to start with default JWT secret when `DEMO_MODE=false`.
- [ ] README setup followed by a teammate on a clean machine works.

---

## 8. Demo script (what to show, ~6 minutes)

1. Log in as CTY doctor. Open seeded patient 1 → only CTY scan visible, "2 scans at other centers".
2. Request access → open phone-sim in a second tab → read OTP → enter → timeline expands to 3 centers.
3. Upload a new PNEUMONIA test image for patient 1 → show three panels, abnormality index, fusion message, comparison message "reduced by …% since … (Green Valley Clinic)".
4. Open the compare slider between oldest and newest scan; toggle change map.
5. Download PDF → show page 2 (twin) and page 3 (history).
6. Upload a selfie → rejected with 422 message.
7. Upload a NORMAL-classified but twin-abnormal image (pick one beforehand) → "pattern not in known classes, refer to a specialist."
8. Log in as admin → audit log shows every view you just made.
9. Phone-sim → revoke consent → back as doctor, other-center scans disappear.

---

## 9. Effort summary

| Phase | Est. days | Demo-able on its own? |
| :--- | :---: | :---: |
| P0 Foundation | 1 | – |
| P1 Healthy Twin | 4–6 | **Yes** |
| P2 Scan memory core | 5–7 | partly |
| P3 Consent | 2–3 | yes (with P2) |
| P4 Auto-compare | 3–4 | yes |
| P5 Timeline UI / PDF / seed | 4–5 | **Full demo** |
| P6 Hardening/docs | 2 | – |
| **Total** | **~21–28** | |

If time is short: do **P0 + P1** (strongest visual wow, smallest risk), then P2 + P3 + a minimal timeline list (skip slider and personal baseline).

---

## 10. Hand-off and review protocol

1. Worker implements one phase per branch, ticks section 7 for that phase, and reports back with: branch name, the AUROC numbers (P1), screenshots of panels/timeline, and pytest output.
2. Reviewer checks the work against this plan and checklist and returns gaps as instructions.
3. After all phases pass review, this file (`PLAN-healthy-twin-and-scan-memory.md`) is deleted.
