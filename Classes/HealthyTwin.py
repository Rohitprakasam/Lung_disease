import os
import json
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

class HealthyTwin:
    PREPROCESS_VERSION = 1
    IMAGE_SIZE = 256
    CLAHE_CLIP = 2.0
    CLAHE_GRID = (8, 8)

    # Central chest Region of Interest (excluding edges & markers)
    ROI_X_MIN = 0.10
    ROI_X_MAX = 0.90
    ROI_Y_MIN = 0.10
    ROI_Y_MAX = 0.92

    def __init__(self, model_path: str, meta_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"HealthyTwin model not found at {model_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"HealthyTwin metadata not found at {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        meta_version = self.meta.get("preprocess_version")
        if meta_version != self.PREPROCESS_VERSION:
            raise ValueError(
                f"Preprocess version mismatch: code={self.PREPROCESS_VERSION}, meta={meta_version}"
            )

        # Load Keras native autoencoder model
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.thr_abnormal = float(self.meta.get("thr_abnormal", 0.18))
        self.thr_invalid = float(self.meta.get("thr_invalid", 0.45))
        self.percentiles = np.array(self.meta.get("normal_percentiles", list(np.linspace(0.05, 0.25, 101))))

        # Build reusable ROI mask
        h, w = self.IMAGE_SIZE, self.IMAGE_SIZE
        self.roi_mask = np.zeros((h, w), dtype=np.float32)
        x1, x2 = int(w * self.ROI_X_MIN), int(w * self.ROI_X_MAX)
        y1, y2 = int(h * self.ROI_Y_MIN), int(h * self.ROI_Y_MAX)
        self.roi_mask[y1:y2, x1:x2] = 1.0

    @staticmethod
    def preprocess(pil_or_path) -> np.ndarray:
        """
        Unified preprocessing pipeline for both training and inference.
        Returns float32 ndarray of shape (256, 256, 1) in [0.0, 1.0].
        """
        if isinstance(pil_or_path, str):
            img_pil = Image.open(pil_or_path)
        else:
            img_pil = pil_or_path

        # 1. Convert to grayscale
        gray_pil = img_pil.convert("L")
        gray_arr = np.array(gray_pil, dtype=np.uint8)

        # 2. Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=HealthyTwin.CLAHE_CLIP, tileGridSize=HealthyTwin.CLAHE_GRID)
        enhanced = clahe.apply(gray_arr)

        # 3. Resize to 256x256 using area interpolation
        resized = cv2.resize(enhanced, (HealthyTwin.IMAGE_SIZE, HealthyTwin.IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        # 4. Normalize to [0.0, 1.0] float32
        norm = resized.astype(np.float32) / 255.0
        return np.expand_dims(norm, axis=-1)

    @staticmethod
    def pre_gate(pil_image: Image.Image) -> tuple[bool, str | None]:
        """
        Cheap pre-inference heuristic gate: checks image dimensions, aspect ratio,
        grayscale saturation, and intensity contrast.
        """
        w, h = pil_image.size
        # Minimum dimension
        if min(w, h) < 180:
            return False, f"Image resolution too low ({w}x{h}). Minimum dimension is 180px."

        # Aspect ratio bounds (chest radiographs are typically 0.65 to 1.5)
        ratio = w / float(h)
        if ratio < 0.55 or ratio > 1.65:
            return False, f"Aspect ratio ({ratio:.2f}) is atypical for a frontal chest X-ray."

        # Check color saturation (X-rays should have near-zero color divergence)
        rgb_img = pil_image.convert("RGB")
        rgb_arr = np.array(rgb_img, dtype=np.float32)
        r, g, b = rgb_arr[:, :, 0], rgb_arr[:, :, 1], rgb_arr[:, :, 2]
        color_diff = np.mean(np.abs(r - g) + np.abs(g - b))
        if color_diff > 25.0:
            return False, f"Image contains substantial color saturation ({color_diff:.1f}); expected monochrome radiograph."

        # Intensity spread check (not blank or solid color)
        gray_arr = np.array(pil_image.convert("L"), dtype=np.float32)
        std_dev = np.std(gray_arr)
        if std_dev < 15.0:
            return False, f"Image lacks contrast (std={std_dev:.1f}). Appears blank, corrupted, or washed out."

        return True, None

    def analyze(self, pil_image: Image.Image) -> dict:
        """
        Runs autoencoder reconstruction, error map calculation, and abnormality scoring.
        """
        # Pre-gate check
        ok, reason = self.pre_gate(pil_image)
        if not ok:
            return {
                "valid": False,
                "invalid_reason": reason,
                "raw_score": 1.0,
                "abnormality_index": 100,
                "is_abnormal": True,
                "input_256": None,
                "twin_256": None,
                "error_map": None
            }

        inp_256 = self.preprocess(pil_image) # (256, 256, 1)
        batch_inp = np.expand_dims(inp_256, axis=0)

        # Run autoencoder reconstruction
        recon = self.model.predict(batch_inp, verbose=0)[0] # (256, 256, 1)
        x = inp_256[:, :, 0]
        x_hat = recon[:, :, 0]

        # Calculate Pixel-level absolute error
        l1_err = np.abs(x - x_hat)

        # Calculate local structural dissimilarity (approx SSIM diff)
        # Using local mean and variance via uniform/Gaussian filtering
        x_blur = cv2.GaussianBlur(x, (7, 7), 1.5)
        x_hat_blur = cv2.GaussianBlur(x_hat, (7, 7), 1.5)
        ssim_approx = np.abs(x_blur - x_hat_blur)

        # Combined smoothed error map
        combined_err = 0.5 * l1_err + 0.5 * ssim_approx
        blurred_err = cv2.GaussianBlur(combined_err, (9, 9), 4.0)

        # Mask to chest ROI
        masked_err = blurred_err * self.roi_mask

        # Raw score: mean of the top 5% pixel errors inside the ROI
        roi_pixels = masked_err[self.roi_mask > 0.5]
        if len(roi_pixels) > 0:
            top_5_cutoff = np.percentile(roi_pixels, 95)
            raw_score = float(np.mean(roi_pixels[roi_pixels >= top_5_cutoff]))
        else:
            raw_score = float(np.mean(masked_err))

        # Check reconstruction validity gate
        if raw_score > self.thr_invalid:
            return {
                "valid": False,
                "invalid_reason": f"Reconstruction anomaly score ({raw_score:.3f}) exceeds chest radiograph threshold ({self.thr_invalid:.3f}).",
                "raw_score": raw_score,
                "abnormality_index": 100,
                "is_abnormal": True,
                "input_256": inp_256,
                "twin_256": recon,
                "error_map": masked_err
            }

        # Compute Abnormality Index (0 to 100 percentile against healthy baseline)
        if len(self.percentiles) > 0:
            idx = int(np.searchsorted(self.percentiles, raw_score))
            abnormality_index = min(100, max(0, idx))
        else:
            abnormality_index = min(100, max(0, int((raw_score / self.thr_abnormal) * 50)))

        is_abnormal = raw_score > self.thr_abnormal

        return {
            "valid": True,
            "invalid_reason": None,
            "raw_score": round(raw_score, 4),
            "abnormality_index": abnormality_index,
            "is_abnormal": is_abnormal,
            "input_256": inp_256,
            "twin_256": recon,
            "error_map": masked_err
        }

    def render_panels(self, result: dict, out_dir: str, stem: str) -> dict:
        """
        Renders twin.png, diff.png, and a side-by-side triptych.png for visualization and PDF embedding.
        """
        os.makedirs(out_dir, exist_ok=True)
        inp_256 = (result["input_256"][:, :, 0] * 255.0).astype(np.uint8)
        twin_256 = (result["twin_256"][:, :, 0] * 255.0).astype(np.uint8)
        error_map = result["error_map"]

        # Upscale for clear display
        display_size = 512
        inp_disp = cv2.resize(inp_256, (display_size, display_size), interpolation=cv2.INTER_CUBIC)
        twin_disp = cv2.resize(twin_256, (display_size, display_size), interpolation=cv2.INTER_CUBIC)
        err_disp = cv2.resize(error_map, (display_size, display_size), interpolation=cv2.INTER_CUBIC)

        # Normalize error map to [0, 255]
        norm_err = np.clip(err_disp / (self.thr_abnormal * 1.5), 0, 1.0)
        norm_err_u8 = (norm_err * 255.0).astype(np.uint8)

        # Apply Inferno colormap to difference
        heatmap = cv2.applyColorMap(norm_err_u8, cv2.COLORMAP_INFERNO)
        inp_bgr = cv2.cvtColor(inp_disp, cv2.COLOR_GRAY2BGR)
        twin_bgr = cv2.cvtColor(twin_disp, cv2.COLOR_GRAY2BGR)

        # Blend: 50% grayscale detail + 50% heatmap
        diff_blend = cv2.addWeighted(inp_bgr, 0.5, heatmap, 0.5, 0)

        twin_filename = f"twin_{stem}.png"
        diff_filename = f"diff_{stem}.png"
        triptych_filename = f"triptych_{stem}.png"

        twin_path = os.path.join(out_dir, twin_filename)
        diff_path = os.path.join(out_dir, diff_filename)
        triptych_path = os.path.join(out_dir, triptych_filename)

        cv2.imwrite(twin_path, twin_bgr)
        cv2.imwrite(diff_path, diff_blend)

        # Construct labeled Triptych (Original | Healthy Twin | Anomaly Map)
        gutter = 16
        total_w = display_size * 3 + gutter * 2
        total_h = display_size + 60
        triptych = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        triptych[:] = (20, 20, 20) # Dark gray background

        # Place panels
        triptych[40:40 + display_size, 0:display_size] = inp_bgr
        triptych[40:40 + display_size, display_size + gutter : display_size * 2 + gutter] = twin_bgr
        triptych[40:40 + display_size, display_size * 2 + gutter * 2 :] = diff_blend

        # Add Titles
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(triptych, "Original Chest X-Ray", (20, 26), font, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(triptych, "Healthy Twin (Reconstructed)", (display_size + gutter + 20, 26), font, 0.7, (100, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(triptych, "Difference Heatmap", (display_size * 2 + gutter * 2 + 20, 26), font, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        # Footer disclaimer
        cv2.putText(triptych, "AI-generated reference, not a diagnosis.", (20, total_h - 10), font, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

        cv2.imwrite(triptych_path, triptych)

        return {
            "twin_filename": twin_filename,
            "diff_filename": diff_filename,
            "triptych_filename": triptych_filename,
            "twin_path": twin_path,
            "diff_path": diff_path,
            "triptych_path": triptych_path
        }


def fuse(classifier_label: str, classifier_conf: float, twin_result: dict) -> dict:
    """
    Pure decision fusion function joining CNN classification with Healthy Twin anomaly detection.
    Follows Section 1E of PLAN-healthy-twin-and-scan-memory.md.
    """
    if not twin_result.get("valid", True):
        return {
            "status": "invalid_image",
            "message": twin_result.get("invalid_reason", "This does not look like a frontal chest X-ray.")
        }

    is_abnormal = twin_result.get("is_abnormal", False)
    is_normal_class = (classifier_label == "NORMAL")

    if is_normal_class and not is_abnormal:
        status = "consistent_normal"
        message = "No abnormality detected by either model."
    elif not is_normal_class and is_abnormal:
        status = "consistent_disease"
        message = f"Findings consistent: {classifier_label}, abnormal regions highlighted."
    elif is_normal_class and is_abnormal:
        status = "unknown_abnormal"
        message = "Abnormal pattern not matching known classes. Refer to a specialist."
    else: # not is_normal_class and not is_abnormal
        status = "low_agreement"
        message = f"Classifier suggests {classifier_label} but the image is close to healthy anatomy. Clinical review advised."

    if classifier_conf < 0.50:
        message += " Low classifier confidence."

    return {
        "status": status,
        "message": message,
        "classifier_label": classifier_label,
        "classifier_conf": round(classifier_conf, 4),
        "abnormality_index": twin_result.get("abnormality_index", 0)
    }

