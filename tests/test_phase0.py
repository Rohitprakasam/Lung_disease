import os
import io
import pytest
import numpy as np
from PIL import Image
from config import Config
from Classes.Web_Model import Web_Model
from web_app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_config_loads():
    assert Config.MAX_CONTENT_LENGTH > 0
    assert "http://localhost:3000" in Config.CORS_ORIGINS or "http://127.0.0.1:3000" in Config.CORS_ORIGINS

def test_web_model_predict_with_probs():
    model_path = os.path.join(os.path.dirname(__file__), "..", "runs", "best_weight.h5")
    model = Web_Model(model_path, ['COVID-19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS'])
    
    # Create test image
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    pred1, conf1, probs1 = model.predict_with_probs(img)
    pred2, conf2, probs2 = model.predict_with_probs(img)

    # Deterministic output for same input
    assert pred1 == pred2
    assert abs(conf1 - conf2) < 1e-5
    
    # Probabilities sum to approximately 1.0
    prob_sum = sum(probs1.values())
    assert 0.99 <= prob_sum <= 1.01
    
    # Confidence is real model output (not hardcoded 0.95)
    assert 0.0 <= conf1 <= 1.0
    assert pred1 in ['COVID-19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

def test_safe_upload_filename(client):
    # Create realistic contrast grayscale image
    arr = np.random.randint(40, 210, size=(224, 224), dtype=np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Attempt path traversal in filename
    data = {
        'image': (img_bytes, '../../evil.png'),
        'id': 'TEST-1',
        'name': 'Test Patient',
        'age': '30',
        'history': 'None'
    }
    
    response = client.post('/api/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    res_json = response.get_json()
    assert 'prediction' in res_json
    assert 'confidence' in res_json
    assert 'report_url' in res_json
    
    # Confirm no file named 'evil.png' exists in parent directory
    assert not os.path.exists('../../evil.png')
    assert not os.path.exists('../evil.png')

def test_invalid_image_pre_gate_rejection(client):
    # Blank image with zero contrast
    blank_img = Image.new("RGB", (224, 224), color=(80, 80, 80))
    img_bytes = io.BytesIO()
    blank_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    data = {
        'image': (img_bytes, 'blank.png'),
        'id': 'TEST-INVALID'
    }
    response = client.post('/api/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 422
    assert "Invalid X-Ray" in response.get_json().get("error", "")

def test_disallowed_extension(client):
    data = {
        'image': (io.BytesIO(b'fake script content'), 'script.sh'),
        'id': 'TEST-2'
    }
    response = client.post('/api/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert "Unsupported file type" in response.get_json().get("error", "")
