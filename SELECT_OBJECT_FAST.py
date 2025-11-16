"""
SAM2 Fast Object Selection - Just the clicking part!
Click on objects → See instant green overlay → That's it!
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog

# Add TensorRT predictor to path
sys.path.insert(0, str(Path(__file__).parent))

# Import TensorRT predictor
from sam2_trt_predictor import SAM2TensorRTPredictor

# Configuration
ENCODER_ENGINE = r"D:\watermarkz\sam2_trt_inference\engines\sam2_encoder_fp16.engine"
DECODER_ENGINE = r"D:\watermarkz\sam2_trt_inference\engines\sam2_decoder_fp16_dynamic.engine"
MAX_DISPLAY_WIDTH = 800

# Global state - multi-object selection
selections = []  # Array of {id, points, mask}
current_selection_id = 0
display_scale = 1.0
original_size = None


def select_video_file():
    """Open file picker dialog to select video"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("[SELECT] Please choose a video file...")
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        initialdir=os.getcwd(),
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("MP4 files", "*.mp4"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not file_path:
        print("[ERROR] No file selected!")
        sys.exit(1)

    print(f"[OK] Selected: {file_path}")
    return file_path


def load_sam2_model():
    """Load SAM2 TensorRT model"""
    print("[SAM2] Loading SAM2 TensorRT FP16 predictor...")

    if not os.path.exists(ENCODER_ENGINE):
        print(f"[ERROR] TensorRT encoder not found: {ENCODER_ENGINE}")
        sys.exit(1)

    if not os.path.exists(DECODER_ENGINE):
        print(f"[ERROR] TensorRT decoder not found: {DECODER_ENGINE}")
        sys.exit(1)

    predictor = SAM2TensorRTPredictor(ENCODER_ENGINE, DECODER_ENGINE)

    print(f"[OK] SAM2 TensorRT loaded!")
    return predictor


def load_first_frame(video_path):
    """Load first frame from video"""
    global original_size, display_scale

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Could not read video!")
        sys.exit(1)

    original_size = (frame.shape[1], frame.shape[0])

    # Calculate display scale
    if frame.shape[1] > MAX_DISPLAY_WIDTH:
        display_scale = MAX_DISPLAY_WIDTH / frame.shape[1]
        display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
    else:
        display_scale = 1.0
        display_frame = frame.copy()

    print(f"[VIDEO] Size: {original_size[0]}x{original_size[1]}")
    print(f"[VIDEO] Display scale: {display_scale:.2f}x")

    return frame, display_frame


def get_sam2_mask(predictor, frame, points_list):
    """Get SAM2 mask for current points"""
    if not points_list:
        return None

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(frame_rgb)

        points_array = np.array([[x, y] for x, y, _ in points_list], dtype=np.float32)
        labels_array = np.array([label for _, _, label in points_list], dtype=np.int32)

        mask, score = predictor.predict(points_array, labels_array)
        return mask
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return None


def draw_overlay(display_frame, selections_list):
    """Draw all masks and points"""
    overlay = display_frame.copy()

    # Draw all masks (green overlay)
    for selection in selections_list:
        mask = selection['mask']
        if mask is not None:
            mask_display = cv2.resize(
                mask.astype(np.uint8) * 255,
                (display_frame.shape[1], display_frame.shape[0])
            )
            mask_bool = mask_display > 127
            overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([0, 255, 0]) * 0.5

    # Draw all points
    for selection in selections_list:
        for x, y, label in selection['points']:
            x_display = int(x * display_scale)
            y_display = int(y * display_scale)
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(overlay, (x_display, y_display), 5, color, -1)
            cv2.circle(overlay, (x_display, y_display), 6, (255, 255, 255), 2)

    # Draw instructions
    total_objects = len(selections_list)
    cv2.putText(overlay, f"Objects: {total_objects} | Press 'r' to reset | 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return overlay


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks - each click creates a NEW object"""
    global selections, current_selection_id

    predictor, frame, display_frame = param

    if event == cv2.EVENT_LBUTTONDOWN:
        # Create new selection for this click
        x_orig = int(x / display_scale)
        y_orig = int(y / display_scale)
        points = [(x_orig, y_orig, 1)]

        print(f"[CLICK] Creating object #{current_selection_id} at ({x_orig}, {y_orig})")

        # Get mask for this single point
        mask = get_sam2_mask(predictor, frame, points)

        if mask is not None:
            selections.append({
                'id': current_selection_id,
                'points': points,
                'mask': mask
            })
            current_selection_id += 1
            print(f"[OK] Added object #{current_selection_id - 1}, total: {len(selections)}")

        update_display(predictor, frame, display_frame)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click - add negative point to refine LAST selection
        if selections:
            x_orig = int(x / display_scale)
            y_orig = int(y / display_scale)

            last_selection = selections[-1]
            last_selection['points'].append((x_orig, y_orig, 0))
            print(f"[REFINE] Added negative point to object #{last_selection['id']}")

            # Re-predict with all points
            last_selection['mask'] = get_sam2_mask(predictor, frame, last_selection['points'])
            update_display(predictor, frame, display_frame)


def update_display(predictor, frame, display_frame):
    """Update display window"""
    global selections

    overlay = draw_overlay(display_frame, selections)
    cv2.imshow("SAM2 Fast Object Selection", overlay)


def main():
    global selections, current_selection_id

    print("="*70)
    print("SAM2 FAST OBJECT SELECTION - MULTI-OBJECT MODE")
    print("="*70)

    # Select video
    video_path = select_video_file()

    # Load SAM2
    predictor = load_sam2_model()

    # Load first frame
    frame, display_frame = load_first_frame(video_path)

    # Create window
    cv2.namedWindow("SAM2 Fast Object Selection")
    cv2.setMouseCallback("SAM2 Fast Object Selection",
                         mouse_callback, (predictor, frame, display_frame))

    # Show initial frame
    update_display(predictor, frame, display_frame)

    print("\n[READY] Left click = New object | Right click = Refine last object")

    # Main loop
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[QUIT] Exiting")
            break

        elif key == ord('r'):
            print("[RESET] Clearing all objects")
            selections = []
            current_selection_id = 0
            update_display(predictor, frame, display_frame)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
