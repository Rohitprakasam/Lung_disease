import os
import cv2
import numpy as np
from PIL import Image
from Classes.HealthyTwin import HealthyTwin

class ScanCompare:
    COMPARE_SIZE = 512

    @staticmethod
    def normalize_for_compare(img_pil: Image.Image) -> np.ndarray:
        """Grayscale -> CLAHE -> 512x512 uint8 image."""
        gray = np.array(img_pil.convert("L"), dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.resize(enhanced, (ScanCompare.COMPARE_SIZE, ScanCompare.COMPARE_SIZE), interpolation=cv2.INTER_AREA)

    @staticmethod
    def register_images(fixed_img: np.ndarray, moving_img: np.ndarray) -> tuple[np.ndarray | None, str, float]:
        """
        Registers moving_img onto fixed_img using ECC (Enhanced Correlation Coefficient)
        with fallback to ORB + RANSAC affine estimation.
        Returns: (warp_matrix, method, quality_score)
        """
        # Downscale to 256 for fast ECC convergence
        scale = 256.0 / ScanCompare.COMPARE_SIZE
        fixed_small = cv2.resize(fixed_img, (256, 256), interpolation=cv2.INTER_AREA)
        moving_small = cv2.resize(moving_img, (256, 256), interpolation=cv2.INTER_AREA)

        warp_2x3 = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-4)

        method = 'none'
        quality = 0.0

        # Method 1: ECC
        try:
            _, warp_small = cv2.findTransformECC(
                fixed_small, moving_small, warp_2x3, cv2.MOTION_AFFINE, criteria, None, 5
            )
            # Scale translation parameters back up to full 512 size
            warp_matrix = warp_small.copy()
            warp_matrix[0, 2] /= scale
            warp_matrix[1, 2] /= scale
            method = 'ecc'
        except Exception:
            # Method 2: Fallback to ORB feature matching
            try:
                orb = cv2.ORB_create(2000)
                kp1, des1 = orb.detectAndCompute(fixed_img, None)
                kp2, des2 = orb.detectAndCompute(moving_img, None)

                if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)

                    if len(matches) >= 8:
                        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:60]]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:60]]).reshape(-1, 1, 2)
                        warp_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
                        if warp_matrix is not None:
                            method = 'orb'
                        else:
                            warp_matrix = np.eye(2, 3, dtype=np.float32)
                    else:
                        warp_matrix = np.eye(2, 3, dtype=np.float32)
                else:
                    warp_matrix = np.eye(2, 3, dtype=np.float32)
            except Exception:
                warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Compute Alignment Quality using Normalized Cross Correlation (NCC) inside ROI
        warped_moving = cv2.warpAffine(moving_img, warp_matrix, (ScanCompare.COMPARE_SIZE, ScanCompare.COMPARE_SIZE))
        
        # Central ROI for quality assessment
        h, w = ScanCompare.COMPARE_SIZE, ScanCompare.COMPARE_SIZE
        y1, y2 = int(h * 0.15), int(h * 0.85)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        roi_fixed = fixed_img[y1:y2, x1:x2].astype(np.float32)
        roi_warped = warped_moving[y1:y2, x1:x2].astype(np.float32)

        roi_fixed -= np.mean(roi_fixed)
        roi_warped -= np.mean(roi_warped)
        denom = (np.std(roi_fixed) * np.std(roi_warped))
        if denom > 1e-5:
            quality = float(np.mean(roi_fixed * roi_warped) / denom)
        else:
            quality = 0.0

        if quality < 0.55:
            method = 'none'

        return warp_matrix, method, round(quality, 3)

    @staticmethod
    def compare_scans(prev_img: Image.Image, new_img: Image.Image,
                      prev_twin_index: int, new_twin_index: int,
                      prev_error_map: np.ndarray, new_error_map: np.ndarray,
                      prev_date_str: str, prev_center_name: str) -> dict:
        """
        Executes longitudinal analysis between previous and new chest X-rays.
        """
        prev_norm = ScanCompare.normalize_for_compare(prev_img)
        new_norm = ScanCompare.normalize_for_compare(new_img)

        warp_matrix, method, quality = ScanCompare.register_images(new_norm, prev_norm)

        # Warp previous error map to align with current scan coordinates
        if method != 'none' and warp_matrix is not None:
            prev_err_512 = cv2.resize(prev_error_map, (ScanCompare.COMPARE_SIZE, ScanCompare.COMPARE_SIZE))
            warped_prev_err = cv2.warpAffine(prev_err_512, warp_matrix, (ScanCompare.COMPARE_SIZE, ScanCompare.COMPARE_SIZE))
        else:
            warped_prev_err = cv2.resize(prev_error_map, (ScanCompare.COMPARE_SIZE, ScanCompare.COMPARE_SIZE))

        new_err_512 = cv2.resize(new_error_map, (ScanCompare.COMPARE_SIZE, ScanCompare.COMPARE_SIZE))

        # Calculate abnormal area fractions (pixels exceeding threshold)
        thr_pixel = 0.12 # Error cutoff
        area_prev = float(np.mean(warped_prev_err > thr_pixel))
        area_new = float(np.mean(new_err_512 > thr_pixel))

        area_delta_pct = ((area_new - area_prev) / max(area_prev, 0.01)) * 100.0
        index_delta = new_twin_index - prev_twin_index

        # Clinical trend rules
        if method == 'none':
            trend = 'unreliable'
            message = "Scans could not be aligned reliably across clinical centers; compare visually."
        elif area_delta_pct <= -15.0 and index_delta <= -5:
            trend = 'improved'
            message = f"AI-estimated abnormal area reduced by {abs(round(area_delta_pct))}% since {prev_date_str} ({prev_center_name})."
        elif area_delta_pct >= 15.0 and index_delta >= 5:
            trend = 'worsened'
            message = f"AI-estimated abnormal area increased by {round(area_delta_pct)}% since {prev_date_str} ({prev_center_name})."
        else:
            trend = 'stable'
            message = f"No meaningful progression or regression detected since {prev_date_str} ({prev_center_name})."

        # Generate change map overlay (Green: Abnormality decreased | Red: Abnormality increased)
        change_diff = new_err_512 - warped_prev_err
        change_map = cv2.cvtColor(new_norm, cv2.COLOR_GRAY2BGR)

        # Areas where abnormality increased (Red)
        increased_mask = (change_diff > 0.05)
        change_map[increased_mask] = (0, 0, 240)

        # Areas where abnormality decreased / cleared (Green)
        decreased_mask = (change_diff < -0.05)
        change_map[decreased_mask] = (0, 240, 0)

        # Blend 60% base scan + 40% change color
        new_norm_bgr = cv2.cvtColor(new_norm, cv2.COLOR_GRAY2BGR)
        blended_change = cv2.addWeighted(new_norm_bgr, 0.6, change_map, 0.4, 0)

        return {
            "alignment_method": method,
            "alignment_quality": quality,
            "trend": trend,
            "message": message,
            "index_delta": index_delta,
            "area_delta_pct": round(area_delta_pct, 1),
            "change_map_image": blended_change
        }

