"""
Insert Video Into Image - Create Video-in-Frame Effect

Usage: python insert_video_into_image.py <background_image> <foreground_video>

This tool:
1. Shows the background image
2. Lets you draw a rectangle where the video should appear
3. Composites the video onto the image at that location
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


def select_rectangle(image_path):
    """Show image and let user draw rectangle. Returns (x1, y1, x2, y2) or None."""
    global drawing, start_point, end_point, rect_confirmed, current_rect

    # Reset globals
    drawing = False
    start_point = None
    end_point = None
    rect_confirmed = False
    current_rect = None

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return None

    # Resize for display if too large
    max_display = 1200
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_display:
        scale = max_display / max(h, w)
        display_img = cv2.resize(img, None, fx=scale, fy=scale)
    else:
        display_img = img.copy()

    window_name = "Draw Rectangle - ENTER=Confirm, R=Reset, ESC=Cancel"
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
        # Scale back to original image coordinates
        x1, y1, x2, y2 = current_rect
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)
        print(f"[OK] Rectangle selected: ({x1}, {y1}) to ({x2}, {y2}) = {x2-x1}x{y2-y1}")
        return (x1, y1, x2, y2)

    return None


def insert_video_into_image(bg_image_path, fg_video_path, rect, video_crop=None):
    """
    Composite video onto image at specified rectangle.

    Args:
        bg_image_path: Path to background image
        fg_video_path: Path to foreground video
        rect: (x1, y1, x2, y2) rectangle coordinates for placement
        video_crop: (x1, y1, x2, y2) rectangle to crop from video (optional)
    """
    x1, y1, x2, y2 = rect

    # Load background image
    bg_img = cv2.imread(bg_image_path)
    if bg_img is None:
        print(f"[ERROR] Could not load background image: {bg_image_path}")
        return

    bg_h, bg_w = bg_img.shape[:2]

    # Clamp rectangle to image bounds
    x1 = max(0, min(x1, bg_w))
    y1 = max(0, min(y1, bg_h))
    x2 = max(0, min(x2, bg_w))
    y2 = max(0, min(y2, bg_h))
    rect_w, rect_h = x2 - x1, y2 - y1

    # H.264 requires even dimensions - adjust if needed
    if bg_w % 2 != 0:
        bg_w -= 1
        bg_img = bg_img[:, :bg_w]
    if bg_h % 2 != 0:
        bg_h -= 1
        bg_img = bg_img[:bg_h, :]
    # Re-clamp after adjusting
    x2 = min(x2, bg_w)
    y2 = min(y2, bg_h)
    rect_w, rect_h = x2 - x1, y2 - y1

    print(f"[INFO] Background image: {bg_w}x{bg_h}")
    print(f"[INFO] Video insert area: {rect_w}x{rect_h} at ({x1}, {y1})")

    # Open video
    cap = cv2.VideoCapture(fg_video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {fg_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Input video: {video_w}x{video_h} @ {fps:.2f}fps, {total_frames} frames")

    # Output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(bg_image_path), "video_in_frame_output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"video_in_frame_{timestamp}.mp4")

    # Find ffmpeg
    import subprocess
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_paths = [
        os.path.join(script_dir, "ffmpeg", "ffmpeg.exe"),  # D:\watermarkz\ffmpeg\
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
        print("        Or place ffmpeg.exe in C:\\ffmpeg\\bin\\")
        cap.release()
        return None

    print(f"\n[PROCESSING] Compositing {total_frames} frames via ffmpeg pipe...")
    print(f"[FFMPEG] Output: {bg_w}x{bg_h} @ {fps:.2f}fps")

    # Use ffmpeg pipe for ANY resolution (no codec limits!)
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

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=None)  # Show ffmpeg errors

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop video frame if specified
        if video_crop:
            vx1, vy1, vx2, vy2 = video_crop
            frame = frame[vy1:vy2, vx1:vx2]

        # Resize video frame to fit rectangle
        resized_frame = cv2.resize(frame, (rect_w, rect_h))

        # Composite onto background
        output_frame = bg_img.copy()
        output_frame[y1:y2, x1:x2] = resized_frame

        proc.stdin.write(output_frame.tobytes())
        frame_count += 1

        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {frame_count}/{total_frames} ({progress:.1f}%)")

    cap.release()
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
        print("Usage: python insert_video_into_image.py <background_image> <foreground_video>")
        print("\nExample:")
        print("  python insert_video_into_image.py wall_photo.jpg video.mp4")
        sys.exit(1)

    bg_image_path = sys.argv[1]
    fg_video_path = sys.argv[2]

    print("=" * 60)
    print("INSERT VIDEO INTO IMAGE")
    print("=" * 60)
    print(f"Background: {bg_image_path}")
    print(f"Video: {fg_video_path}")
    print()

    # Check files exist
    if not os.path.exists(bg_image_path):
        print(f"[ERROR] Background image not found: {bg_image_path}")
        sys.exit(1)

    if not os.path.exists(fg_video_path):
        print(f"[ERROR] Video not found: {fg_video_path}")
        sys.exit(1)

    # Step 1: Select crop area from VIDEO (to remove black borders)
    print("\n" + "=" * 60)
    print("STEP 1: Crop the VIDEO (remove black borders)")
    print("=" * 60)

    # Extract first frame from video for selection
    cap = cv2.VideoCapture(fg_video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Could not read video frame")
        sys.exit(1)

    # Save temp frame for selection
    temp_frame_path = os.path.join(os.path.dirname(fg_video_path), "_temp_first_frame.jpg")
    cv2.imwrite(temp_frame_path, first_frame)

    print("Draw rectangle on VIDEO to select area to USE (crop out black borders)")
    video_crop = select_rectangle(temp_frame_path)

    # Clean up temp file
    if os.path.exists(temp_frame_path):
        os.remove(temp_frame_path)

    if video_crop is None:
        print("[INFO] No video crop selected - using full frame")
        video_crop = None

    # Step 2: Select placement rectangle on IMAGE
    print("\n" + "=" * 60)
    print("STEP 2: Select WHERE to place the video in the IMAGE")
    print("=" * 60)

    rect = select_rectangle(bg_image_path)
    if rect is None:
        print("[CANCELLED] No rectangle selected.")
        sys.exit(0)

    # Step 3: Process
    output_path = insert_video_into_image(bg_image_path, fg_video_path, rect, video_crop)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
