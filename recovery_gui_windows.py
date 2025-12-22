"""
Windows helper for WSL2 recovery clicks.

Run this on Windows while WSL2 worker is processing videos.
When object is LOST, this script will show a popup for you to click.

Usage:
    python recovery_gui_windows.py

The script watches for D:\\watermarkz\\recovery_signal.json
When found, shows the frame and waits for user click.
"""
import os
import json
import time
import cv2

# File paths for communication
SIGNAL_FILE = r"D:\watermarkz\recovery_signal.json"
RESPONSE_FILE = r"D:\watermarkz\recovery_response.json"
FRAME_FILE = r"D:\watermarkz\recovery_frame.jpg"


def main():
    print("=" * 60)
    print("  RECOVERY GUI - Windows Helper for WSL2")
    print("=" * 60)
    print(f"Watching: {SIGNAL_FILE}")
    print("When object is LOST, a window will pop up.")
    print("Click on the object to re-identify, or press ESC to skip.")
    print("-" * 60)

    while True:
        # Check for signal file
        if os.path.exists(SIGNAL_FILE):
            try:
                with open(SIGNAL_FILE, 'r') as f:
                    signal = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[ERROR] Failed to read signal: {e}")
                time.sleep(0.5)
                continue

            if signal.get("status") == "waiting":
                frame_idx = signal.get("frame_idx", "?")
                print(f"\n[!] OBJECT LOST at frame {frame_idx}!")

                # Load the frame
                if not os.path.exists(FRAME_FILE):
                    print(f"[ERROR] Frame file not found: {FRAME_FILE}")
                    # Write skip response
                    with open(RESPONSE_FILE, 'w') as f:
                        json.dump({"click": None}, f)
                    os.remove(SIGNAL_FILE)
                    continue

                frame = cv2.imread(FRAME_FILE)
                if frame is None:
                    print("[ERROR] Could not load frame image")
                    with open(RESPONSE_FILE, 'w') as f:
                        json.dump({"click": None}, f)
                    os.remove(SIGNAL_FILE)
                    continue

                # Setup click handler
                click_point = None

                def mouse_callback(event, x, y, flags, param):
                    nonlocal click_point
                    if event == cv2.EVENT_LBUTTONDOWN:
                        click_point = (x, y)
                        print(f"    Clicked at: ({x}, {y})")

                # Create window
                window_name = "RECOVERY - Click on Object"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(window_name, mouse_callback)

                # Add overlay text
                display = frame.copy()
                cv2.putText(display, f"Frame {frame_idx}: OBJECT LOST", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(display, "Click on object to re-identify", (20, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display, "Press ESC to skip this frame", (20, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                cv2.imshow(window_name, display)

                # Wait for click or ESC
                while click_point is None:
                    key = cv2.waitKey(100)
                    if key == 27:  # ESC
                        print("    User pressed ESC - skipping")
                        break
                    # Check if window was closed
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            print("    Window closed - skipping")
                            break
                    except cv2.error:
                        break

                cv2.destroyAllWindows()
                cv2.waitKey(1)  # Flush

                # Write response
                if click_point:
                    response = {"click": list(click_point)}
                    print(f"    Sending click: {click_point}")
                else:
                    response = {"click": None}
                    print("    Sending skip")

                with open(RESPONSE_FILE, 'w') as f:
                    json.dump(response, f)

                # Cleanup signal file
                try:
                    os.remove(SIGNAL_FILE)
                except:
                    pass

                print("-" * 60)

        time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Recovery GUI] Stopped.")
