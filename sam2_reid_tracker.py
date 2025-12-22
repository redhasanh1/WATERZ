#!/usr/bin/env python3
"""
SAM2 ReID Tracker - Identity Verification Layer

Revolutionary single-object tracking that prevents drift by verifying
object identity using frozen embeddings (DINOv2 for general, ArcFace for faces).

Usage:
    from sam2_reid_tracker import ReIDVerifier

    verifier = ReIDVerifier(object_type="face")  # or "general"
    verifier.set_reference(first_frame, first_mask)

    for frame_idx, mask in tracking_loop:
        is_same, similarity = verifier.verify(frame, mask)
        if not is_same:
            print(f"DRIFT at frame {frame_idx}!")
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
USE_REID_VERIFICATION = True
VERIFY_EVERY_N_FRAMES = 10  # Check identity every N frames
SIMILARITY_THRESHOLD_FACE = 0.45  # ArcFace threshold
SIMILARITY_THRESHOLD_GENERAL = 0.35  # DINOv2 threshold
MIN_CROP_SIZE = 32  # Minimum crop size in pixels


class ReIDVerifier:
    """
    Identity verification using frozen embeddings.

    Supports:
    - DINOv2 (768-D) for general objects (cars, plates, any object)
    - ArcFace (512-D) for faces specifically
    """

    def __init__(self, object_type="general", device="cuda"):
        """
        Initialize ReID verifier.

        Args:
            object_type: "general" (DINOv2) or "face" (ArcFace)
            device: "cuda" or "cpu"
        """
        self.object_type = object_type
        self.device = device
        self.reference_embedding = None
        self.similarity_history = []

        # Set threshold based on object type
        if object_type == "face":
            self.threshold = SIMILARITY_THRESHOLD_FACE
            self._init_arcface()
        else:
            self.threshold = SIMILARITY_THRESHOLD_GENERAL
            self._init_dinov2()

        print(f"[ReID] Initialized {object_type} verifier (threshold={self.threshold})")

    def _init_dinov2(self):
        """Initialize DINOv2 for general object embeddings."""
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model = self.model.to(self.device).eval()
            self.embed_dim = 384  # ViT-S/14
            self.input_size = 224
            print("[ReID] DINOv2 ViT-S/14 loaded (384-D embeddings)")
        except Exception as e:
            print(f"[ReID] Failed to load DINOv2: {e}")
            print("[ReID] Falling back to no verification")
            self.model = None

    def _init_arcface(self):
        """Initialize ArcFace for face embeddings."""
        try:
            from insightface.app import FaceAnalysis
            self.face_app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            self.embed_dim = 512
            self.model = self.face_app  # For consistency
            print("[ReID] InsightFace ArcFace loaded (512-D embeddings)")
        except ImportError:
            print("[ReID] insightface not installed, falling back to DINOv2")
            self.object_type = "general"
            self._init_dinov2()
        except Exception as e:
            print(f"[ReID] Failed to load ArcFace: {e}")
            print("[ReID] Falling back to DINOv2")
            self.object_type = "general"
            self._init_dinov2()

    def _crop_by_mask(self, frame, mask):
        """
        Crop frame region covered by mask.

        Args:
            frame: BGR image (H, W, 3)
            mask: Binary mask (H, W) - uint8 or bool

        Returns:
            Cropped BGR image
        """
        if mask is None or frame is None:
            return None

        # Ensure mask is uint8
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        # Find bounding box
        coords = cv2.findNonZero(mask)
        if coords is None or len(coords) < MIN_CROP_SIZE:
            return None

        x, y, w, h = cv2.boundingRect(coords)

        # Add padding (10%)
        pad_x = max(int(w * 0.1), 5)
        pad_y = max(int(h * 0.1), 5)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)

        crop = frame[y1:y2, x1:x2].copy()

        if crop.shape[0] < MIN_CROP_SIZE or crop.shape[1] < MIN_CROP_SIZE:
            return None

        return crop

    def _preprocess_dinov2(self, crop):
        """Preprocess crop for DINOv2."""
        # Resize to 224x224
        crop_resized = cv2.resize(crop, (self.input_size, self.input_size))

        # BGR -> RGB, normalize
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        crop_tensor = torch.from_numpy(crop_rgb).float() / 255.0

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        crop_tensor = (crop_tensor - mean) / std

        # (H, W, C) -> (1, C, H, W)
        crop_tensor = crop_tensor.permute(2, 0, 1).unsqueeze(0)

        return crop_tensor.to(self.device)

    def _get_embedding(self, crop):
        """
        Extract embedding from cropped image.

        Args:
            crop: BGR image

        Returns:
            Normalized embedding tensor (embed_dim,)
        """
        if crop is None or self.model is None:
            return None

        try:
            if self.object_type == "face":
                # ArcFace: expects BGR numpy array
                faces = self.face_app.get(crop)
                if len(faces) == 0:
                    # No face detected, fall back to DINOv2-style center crop
                    return None
                # Use largest face
                face = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])
                embedding = torch.from_numpy(face.embedding).float()
            else:
                # DINOv2: extract CLS token
                with torch.no_grad():
                    crop_tensor = self._preprocess_dinov2(crop)
                    embedding = self.model(crop_tensor)
                    embedding = embedding.squeeze(0)  # (embed_dim,)

            # L2 normalize for cosine similarity
            embedding = F.normalize(embedding.float(), p=2, dim=0)
            return embedding.cpu()

        except Exception as e:
            print(f"[ReID] Embedding extraction failed: {e}")
            return None

    def set_reference(self, frame, mask):
        """
        Set reference embedding from first frame.

        Args:
            frame: BGR image of first frame
            mask: Binary mask of target object
        """
        crop = self._crop_by_mask(frame, mask)
        if crop is None:
            print("[ReID] Warning: Could not extract reference crop")
            return False

        self.reference_embedding = self._get_embedding(crop)
        if self.reference_embedding is None:
            print("[ReID] Warning: Could not extract reference embedding")
            return False

        print(f"[ReID] Reference embedding set ({self.embed_dim}-D)")
        return True

    def verify(self, frame, mask):
        """
        Verify if current mask contains same object as reference.

        Args:
            frame: BGR image of current frame
            mask: Binary mask of tracked object

        Returns:
            (is_same_object: bool, similarity: float)
        """
        if self.reference_embedding is None:
            return True, 1.0  # No reference, assume OK

        if self.model is None:
            return True, 1.0  # No model, skip verification

        crop = self._crop_by_mask(frame, mask)
        if crop is None:
            # Can't verify (empty mask or decode error) - return uncertain, not drift
            return True, -1.0  # -1.0 = uncertain (impossible cosine sim value)

        current_embedding = self._get_embedding(crop)
        if current_embedding is None:
            # Embedding extraction failed (decode error, empty crop, etc.)
            # Return uncertain instead of triggering false drift alarm
            return True, -1.0

        # Cosine similarity (embeddings are already L2 normalized)
        similarity = torch.dot(self.reference_embedding, current_embedding).item()

        # Track history for EMA-based confidence
        self.similarity_history.append(similarity)
        if len(self.similarity_history) > 30:
            self.similarity_history.pop(0)

        is_same = similarity >= self.threshold
        return is_same, similarity

    def get_confidence_trend(self):
        """Get EMA of similarity scores to detect gradual drift."""
        if len(self.similarity_history) < 3:
            return 1.0

        # Exponential moving average
        alpha = 0.3
        ema = self.similarity_history[0]
        for sim in self.similarity_history[1:]:
            ema = alpha * sim + (1 - alpha) * ema
        return ema

    def reset(self):
        """Reset verifier state."""
        self.reference_embedding = None
        self.similarity_history = []

    def find_object_in_frame(self, frame, mask_hint=None, grid_size=8, min_similarity=0.25):
        """
        Find where reference object is in frame using DINOv2 visual search.

        Uses two-phase search:
        1. If mask_hint provided, search around mask centroid at multiple scales
        2. Fall back to global grid search if local search fails

        Args:
            frame: BGR image
            mask_hint: Current (possibly drifted) mask to guide search
            grid_size: Grid density for global fallback (8 = 64 patches)
            min_similarity: Minimum similarity to consider a match

        Returns:
            (best_point, best_similarity) or (None, best_similarity) if not found
        """
        if self.reference_embedding is None or self.model is None:
            return None, 0.0

        H, W = frame.shape[:2]
        base_patch_size = min(H, W) // grid_size

        best_sim = -1.0
        best_point = None

        with torch.no_grad():
            # PHASE 1: Search around mask hint (if provided)
            if mask_hint is not None and np.sum(mask_hint) > 100:
                coords = np.argwhere(mask_hint > 127)
                if len(coords) > 0:
                    cy, cx = coords.mean(axis=0).astype(int)

                    # Search 3x3 grid around centroid at multiple scales
                    for scale in [0.5, 1.0, 1.5, 2.0]:
                        patch_size = int(base_patch_size * scale)
                        half = patch_size // 2

                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                py = int(cy + dy * patch_size // 2)
                                px = int(cx + dx * patch_size // 2)

                                # Extract patch
                                x1 = max(0, px - half)
                                y1 = max(0, py - half)
                                x2 = min(W, px + half)
                                y2 = min(H, py + half)

                                patch = frame[y1:y2, x1:x2]
                                if patch.shape[0] < MIN_CROP_SIZE or patch.shape[1] < MIN_CROP_SIZE:
                                    continue

                                embedding = self._get_embedding(patch)
                                if embedding is None:
                                    continue

                                sim = torch.dot(self.reference_embedding, embedding).item()
                                if sim > best_sim:
                                    best_sim = sim
                                    best_point = (px, py)

                    # If local search found good match, return early
                    if best_sim >= min_similarity:
                        return best_point, best_sim

            # PHASE 2: Global grid search (fallback)
            for i in range(grid_size):
                for j in range(grid_size):
                    cx = int((j + 0.5) * W / grid_size)
                    cy = int((i + 0.5) * H / grid_size)

                    half = base_patch_size // 2
                    x1 = max(0, cx - half)
                    y1 = max(0, cy - half)
                    x2 = min(W, cx + half)
                    y2 = min(H, cy + half)

                    patch = frame[y1:y2, x1:x2]
                    if patch.shape[0] < MIN_CROP_SIZE or patch.shape[1] < MIN_CROP_SIZE:
                        continue

                    embedding = self._get_embedding(patch)
                    if embedding is None:
                        continue

                    sim = torch.dot(self.reference_embedding, embedding).item()
                    if sim > best_sim:
                        best_sim = sim
                        best_point = (cx, cy)

        if best_sim < min_similarity:
            return None, best_sim

        return best_point, best_sim


def classify_tracking_state(mask, drift_sim, search_sim):
    """
    Classify tracking state to decide what action to take.

    Args:
        mask: Current mask (uint8)
        drift_sim: Similarity when drift was detected
        search_sim: Best similarity from visual search

    Returns:
        One of: "OK", "DRIFT", "LOST", "OCCLUDED"
    """
    mask_area = np.sum(mask > 127)

    if drift_sim >= 0.35:
        return "OK"
    elif search_sim >= 0.35 and mask_area > 5000:
        return "DRIFT"  # Found something, probably drifted to similar object
    elif search_sim < 0.25 or mask_area < 1000:
        return "LOST"   # Can't find object anywhere
    else:
        return "OCCLUDED"  # Keep tracking through partial occlusion


def get_user_click_for_recovery(frame_bgr, frame_idx, previous_mask=None):
    """
    Show frame to user and wait for click to re-identify object.

    Opens a cv2 window showing the frame with previous mask overlay.
    User clicks on the object's new location, or presses ESC to skip.

    Args:
        frame_bgr: BGR image of current frame
        frame_idx: Frame number (for display)
        previous_mask: Previous mask to show as faded overlay

    Returns:
        (x, y) tuple if user clicked, None if ESC/closed window or GUI unavailable
    """
    click_point = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal click_point
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point = (x, y)

    # Create display frame with overlay
    display = frame_bgr.copy()
    if previous_mask is not None and np.sum(previous_mask) > 0:
        # Resize mask to match frame if needed (mask may be SAM2's 720p, frame may be 1080p)
        mask_for_display = previous_mask
        if previous_mask.shape[:2] != frame_bgr.shape[:2]:
            mask_for_display = cv2.resize(
                previous_mask,
                (frame_bgr.shape[1], frame_bgr.shape[0]),  # (width, height)
                interpolation=cv2.INTER_NEAREST
            )
        # Show faded red overlay where mask WAS
        overlay = display.copy()
        overlay[mask_for_display > 127] = [0, 0, 255]  # Red
        display = cv2.addWeighted(overlay, 0.3, display, 0.7, 0)

    # Add text
    cv2.putText(display, f"Frame {frame_idx}: OBJECT LOST", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(display, "Click on object to re-identify, ESC to skip", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Try to create window - may fail in headless environments (WSL2, Docker, SSH)
    window_name = "Recovery - Click on Object"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error as e:
        print(f"[ReID] GUI unavailable ({e}), skipping user input")
        return None

    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, display)

    while click_point is None:
        key = cv2.waitKey(100)
        if key == 27:  # ESC
            break
        # Check if window was closed
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)  # Flush events
    return click_point


def integrate_reid_verification(
    frame_generator,
    output_masks_dir,
    verifier,
    verify_every=VERIFY_EVERY_N_FRAMES,
    on_drift_callback=None
):
    """
    Wrapper that adds ReID verification to any frame generator.

    Args:
        frame_generator: Iterator yielding (frame_idx, frame, mask)
        output_masks_dir: Directory to save masks
        verifier: ReIDVerifier instance
        verify_every: Check identity every N frames
        on_drift_callback: Function to call when drift detected

    Yields:
        (frame_idx, mask, is_verified, similarity)
    """
    reference_set = False
    drift_count = 0

    for frame_idx, frame, mask in frame_generator:
        similarity = 1.0
        is_verified = True

        # Set reference on first valid mask
        if not reference_set and mask is not None and np.sum(mask) > 100:
            if verifier.set_reference(frame, mask):
                reference_set = True
                print(f"[ReID] Reference set at frame {frame_idx}")

        # Verify identity periodically
        elif reference_set and frame_idx % verify_every == 0:
            is_verified, similarity = verifier.verify(frame, mask)

            if not is_verified:
                drift_count += 1
                print(f"[ReID] DRIFT DETECTED at frame {frame_idx} "
                      f"(sim={similarity:.3f}, count={drift_count})")

                if on_drift_callback:
                    mask = on_drift_callback(frame_idx, frame, mask)

        yield frame_idx, mask, is_verified, similarity

    if drift_count > 0:
        print(f"[ReID] Total drift events: {drift_count}")


# =============================================================================
# QUICK INTEGRATION HELPER
# =============================================================================

def create_verifier_for_object(first_frame, first_mask, auto_detect_face=True):
    """
    Create appropriate verifier based on object type.

    Args:
        first_frame: BGR image of first frame
        first_mask: Binary mask of target object
        auto_detect_face: If True, auto-detect if object is a face

    Returns:
        ReIDVerifier instance with reference set
    """
    object_type = "general"

    if auto_detect_face:
        # Try to detect if it's a face
        try:
            from insightface.app import FaceAnalysis
            temp_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            temp_app.prepare(ctx_id=-1, det_size=(320, 320))

            # Crop by mask and check for faces
            crop = ReIDVerifier(object_type="general")._crop_by_mask(first_frame, first_mask)
            if crop is not None:
                faces = temp_app.get(crop)
                if len(faces) > 0:
                    object_type = "face"
                    print("[ReID] Auto-detected: FACE")
                else:
                    print("[ReID] Auto-detected: General object")
            del temp_app
        except:
            print("[ReID] Face detection unavailable, using general")

    verifier = ReIDVerifier(object_type=object_type)
    verifier.set_reference(first_frame, first_mask)

    return verifier


if __name__ == "__main__":
    # Quick test
    print("SAM2 ReID Tracker Module")
    print("========================")
    print(f"USE_REID_VERIFICATION: {USE_REID_VERIFICATION}")
    print(f"VERIFY_EVERY_N_FRAMES: {VERIFY_EVERY_N_FRAMES}")
    print(f"SIMILARITY_THRESHOLD_FACE: {SIMILARITY_THRESHOLD_FACE}")
    print(f"SIMILARITY_THRESHOLD_GENERAL: {SIMILARITY_THRESHOLD_GENERAL}")

    # Test DINOv2 loading
    print("\nTesting DINOv2...")
    try:
        verifier = ReIDVerifier(object_type="general")
        print("DINOv2 ready!")
    except Exception as e:
        print(f"DINOv2 failed: {e}")

    # Test ArcFace loading
    print("\nTesting ArcFace...")
    try:
        verifier = ReIDVerifier(object_type="face")
        print("ArcFace ready!")
    except Exception as e:
        print(f"ArcFace failed: {e}")
