import sys
sys.path.insert(0, 'python_packages')

import cv2
import os
import glob

# Global variables
bbox = None
drawing = False
start_point = None
display_frame = None

def mouse_callback(event, x, y, flags, param):
    global bbox, drawing, start_point, current_frame, display_frame

    # Scale mouse coords back to original size (80% -> 100%)
    scale = 1.25  # 1/0.8
    x_orig = int(x * scale)
    y_orig = int(y * scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing rectangle
        drawing = True
        start_point = (x_orig, y_orig)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw rectangle while dragging
            temp = current_frame.copy()
            cv2.rectangle(temp, start_point, (x_orig, y_orig), (0, 255, 0), 2)
            # Resize for display
            h, w = temp.shape[:2]
            temp_display = cv2.resize(temp, (int(w * 0.8), int(h * 0.8)))
            cv2.imshow('Label Tool', temp_display)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finish drawing
        drawing = False
        x1, y1 = start_point
        x2, y2 = x_orig, y_orig

        # Ensure x1 < x2 and y1 < y2
        bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

        # Draw final box
        temp = current_frame.copy()
        cv2.rectangle(temp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(temp, "Press SPACE to save, R to redraw", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Resize for display
        h, w = temp.shape[:2]
        temp_display = cv2.resize(temp, (int(w * 0.8), int(h * 0.8)))
        cv2.imshow('Label Tool', temp_display)

print("=" * 60)
print("YOLO Manual Labeling Tool")
print("=" * 60)
print("\nInstructions:")
print("1. Draw a box around the watermark with your mouse")
print("2. Press SPACE to save the label")
print("3. Press R to redraw the box")
print("4. Press Q to quit")
print("=" * 60)

# Ask for video path
print("\nWhich video to extract from?")
print("1. sora_with_watermark.mp4")
print("2. asser.mp4")
print("3. Both")
choice = input("Enter choice (1/2/3): ").strip()

videos = []
if choice == '1':
    videos = ['sora_with_watermark.mp4']
elif choice == '2':
    videos = ['asser.mp4']
else:
    videos = ['sora_with_watermark.mp4', 'asser.mp4']

# Number of frames per video
frames_per_video = 20

# Create directories
os.makedirs('yolo_training/images', exist_ok=True)
os.makedirs('yolo_training/labels', exist_ok=True)

# Setup OpenCV window
cv2.namedWindow('Label Tool')
cv2.setMouseCallback('Label Tool', mouse_callback)

frame_count = 0

for video_path in videos:
    if not os.path.exists(video_path):
        print(f"\n⚠️  {video_path} not found, skipping...")
        continue

    print(f"\n📹 Processing {video_path}...")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // frames_per_video

    for i in range(frames_per_video):
        frame_num = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        # Resize frame to 80% for display
        h, w = frame.shape[:2]
        display_frame = cv2.resize(frame, (int(w * 0.8), int(h * 0.8)))
        current_frame = frame.copy()
        bbox = None

        # Show frame
        cv2.imshow('Label Tool', display_frame)

        print(f"\nFrame {frame_count + 1}/{frames_per_video * len(videos)} - {video_path} frame {frame_num}")
        print("Draw box around watermark, then press SPACE")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Space - save
                if bbox is not None:
                    # Save image
                    img_filename = f'frame_{frame_count:04d}.jpg'
                    img_path = f'yolo_training/images/{img_filename}'
                    cv2.imwrite(img_path, frame)

                    # Convert bbox to YOLO format
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = bbox

                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h

                    # Save label
                    label_filename = f'frame_{frame_count:04d}.txt'
                    label_path = f'yolo_training/labels/{label_filename}'
                    with open(label_path, 'w') as f:
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                    print(f"✅ Saved {img_filename} and {label_filename}")
                    frame_count += 1
                    break
                else:
                    print("⚠️  Draw a box first!")

            elif key == ord('r'):  # R - redraw
                bbox = None
                h, w = current_frame.shape[:2]
                display_frame = cv2.resize(current_frame, (int(w * 0.8), int(h * 0.8)))
                cv2.imshow('Label Tool', display_frame)
                print("Box cleared, draw again")

            elif key == ord('q'):  # Q - quit
                print("\n❌ Labeling cancelled")
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cap.release()

cv2.destroyAllWindows()

print("\n" + "=" * 60)
print(f"✅ LABELING COMPLETE!")
print("=" * 60)
print(f"Labeled {frame_count} frames")
print("\nNext: Run TRAIN_YOLO.bat to train the model")
