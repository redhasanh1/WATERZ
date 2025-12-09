"""
YOLO Training Annotation Tool - FRAME BY FRAME
===============================================
Go through each video frame by frame and draw boxes on each frame.

Controls:
- Drag mouse: Draw bounding box
- Right-click: Remove last box
- 'n': SAVE this frame + go to NEXT FRAME
- 's': SKIP this frame (no annotation)
- 'd': DONE with this video, go to next video
- 'r': Reset boxes on current frame
- 'x': Reset ALL progress (start over)
- 'q': Quit (progress saved)
"""

import cv2
import os
import json
import sys
import shutil
from pathlib import Path
from glob import glob

# Configuration
VIDEOS_DIR = r"D:\watermarkz\videostotrain\videostotrain2"
OUTPUT_DIR = r"D:\watermarkz\yolo_training"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")
MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 800

# Global state for mouse drawing
drawing = False
start_x, start_y = -1, -1
current_box = None
boxes = []
display_scale = 1.0
original_size = (0, 0)


def load_progress():
    """Load progress from JSON file"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        "completed_videos": [],
        "skipped_videos": [],
        "total_extracted_frames": 0,
        "current_video": None,
        "current_frame_idx": 0
    }


def save_progress(progress):
    """Save progress to JSON file"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def reset_all_progress():
    """Reset ALL progress - delete everything and start fresh"""
    print("\n" + "!" * 50)
    print("RESETTING ALL PROGRESS!")
    print("!" * 50)

    if os.path.exists(IMAGES_DIR):
        shutil.rmtree(IMAGES_DIR)
        print(f"[DELETED] {IMAGES_DIR}")

    if os.path.exists(LABELS_DIR):
        shutil.rmtree(LABELS_DIR)
        print(f"[DELETED] {LABELS_DIR}")

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print(f"[DELETED] {PROGRESS_FILE}")

    cache_file = os.path.join(OUTPUT_DIR, "labels.cache")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    print("[RESET COMPLETE]\n")
    return {
        "completed_videos": [],
        "skipped_videos": [],
        "total_extracted_frames": 0,
        "current_video": None,
        "current_frame_idx": 0
    }


def get_video_list():
    """Get list of video files in the folder"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    videos = []
    for ext in video_extensions:
        videos.extend(glob(os.path.join(VIDEOS_DIR, ext)))
    return sorted(videos)


def get_pending_videos(all_videos, progress):
    """Get videos that haven't been processed yet"""
    done = set(progress["completed_videos"] + progress["skipped_videos"])
    return [v for v in all_videos if os.path.basename(v) not in done]


def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for drawing boxes"""
    global drawing, start_x, start_y, current_box, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        current_box = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_box = (start_x, start_y, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if current_box:
            x1 = int(min(start_x, x) / display_scale)
            y1 = int(min(start_y, y) / display_scale)
            x2 = int(max(start_x, x) / display_scale)
            y2 = int(max(start_y, y) / display_scale)

            if (x2 - x1) > 5 and (y2 - y1) > 5:
                boxes.append((x1, y1, x2, y2))
                print(f"[BOX] Added: ({x1},{y1}) -> ({x2},{y2})")
        current_box = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        if boxes:
            removed = boxes.pop()
            print(f"[BOX] Removed: {removed}")


def draw_overlay(display_frame, frame_idx, total_frames, video_idx, total_videos):
    """Draw boxes and info on display frame"""
    overlay = display_frame.copy()

    # Draw completed boxes
    for (x1, y1, x2, y2) in boxes:
        x1_d = int(x1 * display_scale)
        y1_d = int(y1 * display_scale)
        x2_d = int(x2 * display_scale)
        y2_d = int(y2 * display_scale)
        cv2.rectangle(overlay, (x1_d, y1_d), (x2_d, y2_d), (0, 255, 0), 2)

    # Draw current box being drawn
    if current_box:
        x1, y1, x2, y2 = current_box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Info at top
    info = f"Video {video_idx}/{total_videos} | Frame {frame_idx+1}/{total_frames} | Boxes: {len(boxes)}"
    cv2.putText(overlay, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(overlay, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Controls at bottom
    h = overlay.shape[0]
    hint = "SPACE=next | B=back | S=skip | D=done | R=reset | X=restart | Q=quit"
    cv2.putText(overlay, hint, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(overlay, hint, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    return overlay


def convert_to_yolo_format(box, img_width, img_height):
    """Convert box to YOLO format"""
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def save_frame_and_label(frame, boxes, progress):
    """Save a single frame and its label"""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    file_idx = progress["total_extracted_frames"]
    img_path = os.path.join(IMAGES_DIR, f"frame_{file_idx:05d}.jpg")
    label_path = os.path.join(LABELS_DIR, f"frame_{file_idx:05d}.txt")

    # Save image
    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Save label
    height, width = frame.shape[:2]
    label_lines = [convert_to_yolo_format(box, width, height) for box in boxes]
    with open(label_path, 'w') as f:
        f.write("\n".join(label_lines))

    progress["total_extracted_frames"] += 1
    return file_idx


def main():
    global boxes, current_box, display_scale, original_size

    print("=" * 60)
    print("YOLO ANNOTATION - FRAME BY FRAME")
    print("=" * 60)

    if not os.path.exists(VIDEOS_DIR):
        print(f"[ERROR] Videos folder not found: {VIDEOS_DIR}")
        sys.exit(1)

    progress = load_progress()
    all_videos = get_video_list()

    if not all_videos:
        print(f"[ERROR] No videos found in {VIDEOS_DIR}")
        sys.exit(1)

    print(f"Found {len(all_videos)} videos")
    print(f"Completed: {len(progress['completed_videos'])}")
    print(f"Frames saved: {progress['total_extracted_frames']}")

    pending_videos = get_pending_videos(all_videos, progress)

    if not pending_videos:
        print("\n[DONE] All videos processed!")
        return

    print(f"Remaining: {len(pending_videos)} videos\n")

    window_name = "YOLO Annotation (Frame by Frame)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Process each video
    for video_path in pending_videos:
        video_name = os.path.basename(video_path)
        video_idx = len(progress['completed_videos']) + 1
        total_videos = len(all_videos)

        print(f"\n{'='*50}")
        print(f"[VIDEO {video_idx}/{total_videos}] {video_name}")
        print(f"{'='*50}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if resuming this video
        start_frame = 0
        if progress.get("current_video") == video_name:
            start_frame = progress.get("current_frame_idx", 0)
            print(f"[RESUME] Continuing from frame {start_frame}")

        progress["current_video"] = video_name

        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        video_done = False

        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate display scale
            original_size = (frame.shape[1], frame.shape[0])
            scale_w = MAX_DISPLAY_WIDTH / frame.shape[1]
            scale_h = MAX_DISPLAY_HEIGHT / frame.shape[0]
            display_scale = min(scale_w, scale_h, 1.0)

            if display_scale < 1.0:
                display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
            else:
                display_scale = 1.0
                display_frame = frame.copy()

            # Reset boxes for new frame
            boxes = []
            current_box = None

            # Frame annotation loop
            while True:
                overlay = draw_overlay(display_frame.copy(), frame_idx, total_frames, video_idx, total_videos)
                cv2.imshow(window_name, overlay)

                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    # Quit
                    progress["current_frame_idx"] = frame_idx
                    save_progress(progress)
                    print("\n[QUIT] Progress saved!")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                elif key == ord(' '):  # Spacebar
                    # Save and next frame
                    if boxes:
                        save_frame_and_label(frame, boxes, progress)
                        print(f"[SAVED] Frame {frame_idx+1}/{total_frames} with {len(boxes)} box(es)")
                    else:
                        print(f"[SKIP] Frame {frame_idx+1} - no boxes")

                    progress["current_frame_idx"] = frame_idx + 1
                    save_progress(progress)
                    frame_idx += 1
                    break  # Next frame

                elif key == ord('s'):
                    # Skip frame
                    print(f"[SKIP] Frame {frame_idx+1}")
                    progress["current_frame_idx"] = frame_idx + 1
                    save_progress(progress)
                    frame_idx += 1
                    break  # Next frame

                elif key == ord('b'):
                    # Go back one frame
                    if frame_idx > 0:
                        frame_idx -= 1
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        progress["current_frame_idx"] = frame_idx
                        save_progress(progress)
                        print(f"[BACK] Going to frame {frame_idx+1}")
                        break  # Re-read frame
                    else:
                        print("[BACK] Already at first frame!")

                elif key == ord('d'):
                    # Done with video
                    print(f"[DONE] Finished {video_name}")
                    video_done = True
                    break

                elif key == ord('r'):
                    # Reset boxes
                    boxes = []
                    print("[RESET] Cleared boxes")

                elif key == ord('x'):
                    # Reset all
                    cap.release()
                    cv2.destroyAllWindows()
                    progress = reset_all_progress()
                    main()
                    return

            if video_done:
                break

        cap.release()

        # Mark video as completed
        progress["completed_videos"].append(video_name)
        progress["current_video"] = None
        progress["current_frame_idx"] = 0
        save_progress(progress)
        print(f"[COMPLETE] {video_name} - Total frames saved: {progress['total_extracted_frames']}")

    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print("ALL VIDEOS DONE!")
    print(f"Total frames: {progress['total_extracted_frames']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
