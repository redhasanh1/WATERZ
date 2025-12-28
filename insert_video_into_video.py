"""
Insert Video Into Video - Overlay Video on Video

Usage: python insert_video_into_video.py <background_video> <foreground_video>

This tool:
1. Shows first frame of background video
2. Lets you draw a rectangle where the foreground should appear
3. Composites the foreground video onto background video at that location
4. Outputs a new video
"""

import cv2
import numpy as np
import sys
import os
from datetime import datetime

# Global variables for rectangle drawing
drawing = False
start_point = None
end_point = None
rect_confirmed = False
current_rect = None


def draw_rectangle(event, x, y, flags, param):
    """Mouse callback for drawing rectangle."""
    global drawing, start_point, end_point, current_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        # Normalize rectangle (ensure x1 < x2, y1 < y2)
        x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
        x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
        current_rect = (x1, y1, x2, y2)


def select_rectangle_from_frame(frame, window_title="Draw Rectangle"):
    """Show frame and let user draw rectangle. Returns (x1, y1, x2, y2) or None."""
    global drawing, start_point, end_point, rect_confirmed, current_rect

    # Reset globals
    drawing = False
    start_point = None
    end_point = None
    rect_confirmed = False
    current_rect = None

    # Resize for display if too large
    max_display = 1200
    h, w = frame.shape[:2]
    scale = 1.0
    if max(h, w) > max_display:
        scale = max_display / max(h, w)
        display_img = cv2.resize(frame, None, fx=scale, fy=scale)
    else:
        display_img = frame.copy()

    window_name = f"{window_title} - ENTER=Confirm, R=Reset, ESC=Cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, draw_rectangle)

    print("\n[INSTRUCTIONS]")
    print("  - Click and drag to draw rectangle")
    print("  - Press ENTER to confirm")
    print("  - Press R to reset")
    print("  - Press ESC to cancel")
    print()

    while True:
        # Draw current state
        temp_img = display_img.copy()

        # Draw rectangle preview while dragging
        if drawing and start_point and end_point:
            cv2.rectangle(temp_img, start_point, end_point, (0, 255, 0), 2)

        # Draw confirmed rectangle
        if current_rect:
            x1, y1, x2, y2 = current_rect
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Draw dimensions
            rect_w, rect_h = x2 - x1, y2 - y1
            text = f"{int(rect_w/scale)}x{int(rect_h/scale)}"
            cv2.putText(temp_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, temp_img)

        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER
            if current_rect:
                rect_confirmed = True
                break
            else:
                print("[!] Draw a rectangle first!")

        elif key == ord('r') or key == ord('R'):  # Reset
            current_rect = None
            start_point = None
            end_point = None
            print("[RESET] Draw again...")

        elif key == 27:  # ESC
            print("[CANCELLED]")
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    if current_rect:
        # Scale back to original frame coordinates
        x1, y1, x2, y2 = current_rect
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)
        print(f"[OK] Rectangle selected: ({x1}, {y1}) to ({x2}, {y2}) = {x2-x1}x{y2-y1}")
        return (x1, y1, x2, y2)

    return None


def insert_video_into_video(bg_video_path, fg_video_path, rect, video_crop=None):
    """
    Composite foreground video onto background video at specified rectangle.

    Args:
        bg_video_path: Path to background video
        fg_video_path: Path to foreground video
        rect: (x1, y1, x2, y2) rectangle coordinates for placement
        video_crop: (x1, y1, x2, y2) rectangle to crop from foreground (optional)
    """
    # Open background video
    bg_cap = cv2.VideoCapture(bg_video_path)
    if not bg_cap.isOpened():
        print(f"[ERROR] Could not open background video: {bg_video_path}")
        return None

    bg_fps = bg_cap.get(cv2.CAP_PROP_FPS)
    bg_total = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg_w = int(bg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    bg_h = int(bg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Open foreground video
    fg_cap = cv2.VideoCapture(fg_video_path)
    if not fg_cap.isOpened():
        print(f"[ERROR] Could not open foreground video: {fg_video_path}")
        bg_cap.release()
        return None

    fg_fps = fg_cap.get(cv2.CAP_PROP_FPS)
    fg_total = int(fg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fg_w = int(fg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fg_h = int(fg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x1, y1, x2, y2 = rect

    # H.264 requires even dimensions
    if bg_w % 2 != 0:
        bg_w -= 1
    if bg_h % 2 != 0:
        bg_h -= 1

    # Clamp rectangle to bounds
    x1 = max(0, min(x1, bg_w))
    y1 = max(0, min(y1, bg_h))
    x2 = max(0, min(x2, bg_w))
    y2 = max(0, min(y2, bg_h))
    rect_w, rect_h = x2 - x1, y2 - y1

    print(f"[INFO] Background video: {bg_w}x{bg_h} @ {bg_fps:.2f}fps, {bg_total} frames")
    print(f"[INFO] Foreground video: {fg_w}x{fg_h} @ {fg_fps:.2f}fps, {fg_total} frames")
    print(f"[INFO] Overlay area: {rect_w}x{rect_h} at ({x1}, {y1})")

    # Use background fps for output
    fps = bg_fps

    # Output uses the shorter video length
    total_frames = min(bg_total, fg_total)
    print(f"[INFO] Output: {total_frames} frames (shorter of the two videos)")

    # Output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(bg_video_path), "video_overlay_output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"video_overlay_{timestamp}.mp4")

    # Find ffmpeg
    import subprocess
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_paths = [
        os.path.join(script_dir, "ffmpeg", "ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        "ffmpeg"
    ]
    ffmpeg_exe = None
    for p in ffmpeg_paths:
        try:
            subprocess.run([p, "-version"], capture_output=True, check=True)
            ffmpeg_exe = p
            break
        except:
            continue

    if not ffmpeg_exe:
        print("[ERROR] ffmpeg not found! Install ffmpeg and add to PATH.")
        bg_cap.release()
        fg_cap.release()
        return None

    print(f"\n[PROCESSING] Compositing {total_frames} frames via ffmpeg pipe...")
    print(f"[FFMPEG] Output: {bg_w}x{bg_h} @ {fps:.2f}fps")

    # Use ffmpeg pipe
    cmd = [
        ffmpeg_exe, '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{bg_w}x{bg_h}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-profile:v', 'high',
        '-level', '5.1',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=None)

    frame_count = 0
    while frame_count < total_frames:
        bg_ret, bg_frame = bg_cap.read()
        fg_ret, fg_frame = fg_cap.read()

        if not bg_ret or not fg_ret:
            break

        # Ensure even dimensions
        bg_frame = bg_frame[:bg_h, :bg_w]

        # Crop foreground if specified
        if video_crop:
            vx1, vy1, vx2, vy2 = video_crop
            fg_frame = fg_frame[vy1:vy2, vx1:vx2]

        # Resize foreground to fit rectangle
        resized_fg = cv2.resize(fg_frame, (rect_w, rect_h))

        # Composite
        output_frame = bg_frame.copy()
        output_frame[y1:y2, x1:x2] = resized_fg

        proc.stdin.write(output_frame.tobytes())
        frame_count += 1

        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {frame_count}/{total_frames} ({progress:.1f}%)")

    bg_cap.release()
    fg_cap.release()
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        print(f"[ERROR] ffmpeg failed with code {proc.returncode}")
        return None

    print(f"\n[SUCCESS] Output saved to:")
    print(f"  {output_path}")
    print(f"\n[INFO] {frame_count} frames processed at {bg_w}x{bg_h}")

    return output_path


def main():
    if len(sys.argv) < 3:
        print("Usage: python insert_video_into_video.py <background_video> <foreground_video>")
        print("\nExample:")
        print("  python insert_video_into_video.py background.mp4 overlay.mp4")
        sys.exit(1)

    bg_video_path = sys.argv[1]
    fg_video_path = sys.argv[2]

    print("=" * 60)
    print("INSERT VIDEO INTO VIDEO")
    print("=" * 60)
    print(f"Background: {bg_video_path}")
    print(f"Foreground: {fg_video_path}")
    print()

    # Check files exist
    if not os.path.exists(bg_video_path):
        print(f"[ERROR] Background video not found: {bg_video_path}")
        sys.exit(1)

    if not os.path.exists(fg_video_path):
        print(f"[ERROR] Foreground video not found: {fg_video_path}")
        sys.exit(1)

    # Get first frame of foreground for crop selection
    fg_cap = cv2.VideoCapture(fg_video_path)
    ret, fg_first_frame = fg_cap.read()
    fg_cap.release()

    if not ret:
        print("[ERROR] Could not read foreground video frame")
        sys.exit(1)

    # Step 1: Select crop area from FOREGROUND (optional)
    print("\n" + "=" * 60)
    print("STEP 1: Crop the FOREGROUND VIDEO (optional)")
    print("=" * 60)
    print("Draw rectangle to crop foreground (remove borders)")
    print("Press ESC to skip and use full frame")

    video_crop = select_rectangle_from_frame(fg_first_frame, "Crop Foreground Video")

    if video_crop is None:
        print("[INFO] No crop selected - using full foreground frame")

    # Get first frame of background for placement selection
    bg_cap = cv2.VideoCapture(bg_video_path)
    ret, bg_first_frame = bg_cap.read()
    bg_cap.release()

    if not ret:
        print("[ERROR] Could not read background video frame")
        sys.exit(1)

    # Step 2: Select placement rectangle on BACKGROUND
    print("\n" + "=" * 60)
    print("STEP 2: Select WHERE to place the foreground on BACKGROUND")
    print("=" * 60)

    rect = select_rectangle_from_frame(bg_first_frame, "Place on Background")
    if rect is None:
        print("[CANCELLED] No rectangle selected.")
        sys.exit(0)

    # Step 3: Process
    output_path = insert_video_into_video(bg_video_path, fg_video_path, rect, video_crop)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
