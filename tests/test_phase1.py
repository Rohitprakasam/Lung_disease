import os
import numpy as np
import pytest
from PIL import Image
from Classes.HealthyTwin import HealthyTwin, fuse

@pytest.fixture
def twin_instance():
    model_path = os.path.join(os.path.dirname(__file__), "..", "runs", "healthy_twin.keras")
    meta_path = os.path.join(os.path.dirname(__file__), "..", "runs", "healthy_twin_meta.json")
    return HealthyTwin(model_path, meta_path)

def test_preprocess_output():
    img = Image.new("L", (300, 400), color=120)
    processed = HealthyTwin.preprocess(img)
    assert processed.shape == (256, 256, 1)
    assert processed.dtype == np.float32
    assert 0.0 <= processed.min() <= processed.max() <= 1.0

def test_pre_gate_valid_and_invalid():
    # 1. Valid monochrome radiograph
    valid_cxr = Image.new("L", (256, 256), color=100)
    # Add some chest contrast
    arr = np.random.randint(40, 200, size=(256, 256), dtype=np.uint8)
    valid_cxr = Image.fromarray(arr)
    ok, reason = HealthyTwin.pre_gate(valid_cxr)
    assert ok is True
    assert reason is None

    # 2. Too small
    tiny_img = Image.new("L", (100, 100), color=128)
    ok, reason = HealthyTwin.pre_gate(tiny_img)
    assert ok is False
    assert "too low" in reason

    # 3. Vivid color selfie
    color_img = Image.new("RGB", (300, 300), color=(255, 20, 10))
    ok, reason = HealthyTwin.pre_gate(color_img)
    assert ok is False
    assert "saturation" in reason

    # 4. Blank/washed out image
    blank_img = Image.new("L", (256, 256), color=128)
    ok, reason = HealthyTwin.pre_gate(blank_img)
    assert ok is False
    assert "lacks contrast" in reason

def test_fusion_logic():
    # 1. Consistent Normal
    res = fuse("NORMAL", 0.92, {"valid": True, "is_abnormal": False, "abnormality_index": 20})
    assert res["status"] == "consistent_normal"

    # 2. Consistent Disease
    res = fuse("PNEUMONIA", 0.88, {"valid": True, "is_abnormal": True, "abnormality_index": 82})
    assert res["status"] == "consistent_disease"

    # 3. Unknown Abnormal (NORMAL classifier but Twin detects abnormal lesion)
    res = fuse("NORMAL", 0.85, {"valid": True, "is_abnormal": True, "abnormality_index": 92})
    assert res["status"] == "unknown_abnormal"
    assert "not matching known classes" in res["message"]

    # 4. Low Agreement (Classifier predicts disease but Twin says normal)
    res = fuse("COVID-19", 0.75, {"valid": True, "is_abnormal": False, "abnormality_index": 30})
    assert res["status"] == "low_agreement"

    # 5. Invalid Image
    res = fuse("NORMAL", 0.50, {"valid": False, "invalid_reason": "Not a CXR"})
    assert res["status"] == "invalid_image"

def test_analyze_and_render_panels(twin_instance, tmp_path):
    arr = np.random.randint(30, 220, size=(256, 256), dtype=np.uint8)
    cxr = Image.fromarray(arr)
    
    analysis = twin_instance.analyze(cxr)
    assert analysis["valid"] is True
    assert 0 <= analysis["abnormality_index"] <= 100
    assert analysis["error_map"] is not None

    panels = twin_instance.render_panels(analysis, str(tmp_path), "test_sample")
    assert os.path.exists(panels["twin_path"])
    assert os.path.exists(panels["diff_path"])
    assert os.path.exists(panels["triptych_path"])



def test_report_generator_omits_twin_page(tmp_path):
    import re
    from Classes.ReportGenerator import ReportGenerator

    img_path = str(tmp_path / "scan.png")
    Image.new("RGB", (100, 100), color="white").save(img_path)
    out_pdf = str(tmp_path / "report.pdf")

    patient = {"name": "Test Patient", "id": "PT-001", "age": "45", "gender": "F"}
    twin_data = {
        "valid": True,
        "abnormality_index": 75,
        "status": "consistent_disease",
        "message": "Abnormal scan",
        "images": {"triptych": img_path}
    }

    ReportGenerator.generate_pdf(patient, "COVID-19", 0.92, {"COVID-19": 0.92}, img_path, img_path, out_pdf, twin=twin_data)

    assert os.path.exists(out_pdf)
    with open(out_pdf, "rb") as f:
        data = f.read()
    page_count = len(re.findall(rb"/Type\s*/Page\b", data))
    assert page_count == 1
    assert b"AI Visual Analysis" not in data
