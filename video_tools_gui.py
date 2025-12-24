"""
Video Tools GUI - Unified Video Processing (No Inpainting)

Operations:
- Keep Object (remove background: transparent/color/blur)
- Remove Object (keep background: fill with color/blur)
- Fill Inside/Outside with Color
- Blur Inside/Outside

Export: MP4 (NVENC) / WebM (Alpha) / PNG Sequence

Usage:
    python video_tools_gui.py

Requirements:
    - 4090_sender_local.bat running (SAM2 worker)
    - Redis running locally
    - FFmpeg installed (for NVENC encoding)
"""

import sys
import os
import subprocess
import shutil
import threading
import uuid
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Add script directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import cv2
import numpy as np
from PIL import Image, ImageTk

# Import mask utilities
try:
    from static_mask_generator import dilate_mask
except ImportError:
    def dilate_mask(mask, iterations=5):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(mask, kernel, iterations=iterations)


# ============================================================
# Operation Functions
# ============================================================

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color (#RRGGBB) to BGR tuple"""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


def apply_operation(frame: np.ndarray, mask: np.ndarray, operation: str, options: dict) -> np.ndarray:
    """
    Apply operation to frame using mask.

    Args:
        frame: BGR image (H, W, 3)
        mask: Grayscale mask (H, W), 255=object, 0=background
        operation: One of keep_object, remove_object, fill_inside, fill_outside, blur_inside, blur_outside
        options: Dict with bg_type, color_bgr, blur_amount

    Returns:
        Processed frame (BGR or BGRA for transparent)
    """
    # Ensure mask matches frame size
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    mask_f = mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_f] * 3, axis=-1)

    color_bgr = options.get('color_bgr', (0, 255, 0))
    blur_amount = options.get('blur_amount', 25)
    bg_type = options.get('bg_type', 'color')

    # Ensure blur kernel is odd
    blur_kernel = blur_amount * 2 + 1

    if operation == 'keep_object':
        # Keep inside mask, replace outside
        if bg_type == 'transparent':
            # Return BGRA with alpha channel
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = mask  # Alpha from mask
            return rgba
        elif bg_type == 'color':
            bg = np.full_like(frame, color_bgr, dtype=np.uint8)
            return (frame * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)
        else:  # blur
            blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
            return (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)

    elif operation == 'remove_object':
        # Keep outside mask, fill inside with color/blur
        inv_mask_3ch = 1 - mask_3ch
        if bg_type == 'color':
            fill = np.full_like(frame, color_bgr, dtype=np.uint8)
        else:  # blur
            fill = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        return (frame * inv_mask_3ch + fill * mask_3ch).astype(np.uint8)

    elif operation == 'fill_inside':
        # Solid color inside mask
        fill = np.full_like(frame, color_bgr, dtype=np.uint8)
        return (fill * mask_3ch + frame * (1 - mask_3ch)).astype(np.uint8)

    elif operation == 'fill_outside':
        # Solid color outside mask
        fill = np.full_like(frame, color_bgr, dtype=np.uint8)
        return (frame * mask_3ch + fill * (1 - mask_3ch)).astype(np.uint8)

    elif operation == 'blur_inside':
        # Blur inside mask
        blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        return (blurred * mask_3ch + frame * (1 - mask_3ch)).astype(np.uint8)

    elif operation == 'blur_outside':
        # Blur outside mask
        blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        return (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)

    return frame


# ============================================================
# Export Functions
# ============================================================

def export_mp4_nvenc(frames_dir: str, output_path: str, fps: float, width: int, height: int,
                     callback=None) -> bool:
    """Export to MP4 using NVENC H.265"""
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, '%05d.png'),
        '-c:v', 'hevc_nvenc',
        '-cq', '18',
        '-preset', 'p4',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[EXPORT] FFmpeg error: {result.stderr}")
            # Fallback to libx265 if NVENC not available
            ffmpeg_cmd[6] = 'libx265'
            ffmpeg_cmd[7:9] = ['-crf', '18']
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"[EXPORT] Error: {e}")
        return False


def export_webm_alpha(frames_dir: str, output_path: str, fps: float) -> bool:
    """Export to WebM with alpha channel (VP9)"""
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, '%05d.png'),
        '-c:v', 'libvpx-vp9',
        '-pix_fmt', 'yuva420p',
        '-b:v', '0',
        '-crf', '30',
        output_path
    ]

    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"[EXPORT] Error: {e}")
        return False


# ============================================================
# Main GUI Class
# ============================================================

class VideoToolsGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video Tools - GPU Processing")
        self.root.geometry("1200x900")

        # Video state
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 30.0
        self.video_width = 0
        self.video_height = 0
        self.current_frame: Optional[np.ndarray] = None

        # Canvas state
        self.canvas_width = 960
        self.canvas_height = 540
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # SAM2 tracking state
        self.click_points: List[Tuple[int, int, int]] = []  # (x, y, label) - label: 1=foreground, 0=background
        self.tracking_frame_idx = 0

        # Masks state
        self.masks_dir: Optional[str] = None
        self.masks_loaded = False
        self.current_mask: Optional[np.ndarray] = None

        # Processing state
        self.task_id: Optional[str] = None
        self.celery_task = None
        self.is_tracking = False
        self.is_exporting = False

        # Progress
        self.progress_var = tk.DoubleVar(value=0)

        self._create_ui()

    def _create_ui(self):
        # ================== TOP BAR ==================
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(top_frame, text="Load Video", command=self._load_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="<<", command=self._prev_frame, width=3).pack(side=tk.LEFT, padx=2)

        self.frame_label = ttk.Label(top_frame, text="Frame: 0 / 0")
        self.frame_label.pack(side=tk.LEFT, padx=10)

        ttk.Button(top_frame, text=">>", command=self._next_frame, width=3).pack(side=tk.LEFT, padx=2)

        # Frame slider
        self.frame_slider = ttk.Scale(top_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                       length=300, command=self._on_slider)
        self.frame_slider.pack(side=tk.LEFT, padx=10)

        # Video info
        self.video_info = ttk.Label(top_frame, text="No video loaded")
        self.video_info.pack(side=tk.RIGHT, padx=10)

        # ================== CANVAS ==================
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Mouse bindings for click-to-track
        self.canvas.bind("<Button-1>", self._on_left_click)   # Foreground point
        self.canvas.bind("<Button-3>", self._on_right_click)  # Background point

        # ================== CLICK INFO ==================
        click_frame = ttk.Frame(self.root)
        click_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(click_frame, text="Left-click: Add foreground point | Right-click: Add background point | ",
                  font=('TkDefaultFont', 9)).pack(side=tk.LEFT)
        self.points_label = ttk.Label(click_frame, text="Points: 0", font=('TkDefaultFont', 9, 'bold'))
        self.points_label.pack(side=tk.LEFT)

        ttk.Button(click_frame, text="Clear Points", command=self._clear_points).pack(side=tk.LEFT, padx=10)
        ttk.Button(click_frame, text="Track Object", command=self._track_object).pack(side=tk.LEFT, padx=5)

        # ================== OPERATION SELECTOR ==================
        op_frame = ttk.LabelFrame(self.root, text="Operation", padding=10)
        op_frame.pack(fill=tk.X, padx=10, pady=5)

        self.operation_var = tk.StringVar(value='keep_object')
        operations = [
            ('Keep Object, Remove Background', 'keep_object'),
            ('Remove Object, Keep Background', 'remove_object'),
            ('Fill Inside with Color', 'fill_inside'),
            ('Fill Outside with Color', 'fill_outside'),
            ('Blur Inside Mask', 'blur_inside'),
            ('Blur Outside Mask', 'blur_outside'),
        ]

        for i, (text, value) in enumerate(operations):
            ttk.Radiobutton(op_frame, text=text, variable=self.operation_var,
                           value=value).grid(row=i//3, column=i%3, sticky=tk.W, padx=10, pady=2)

        # ================== OPTIONS PANEL ==================
        options_frame = ttk.LabelFrame(self.root, text="Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)

        # Row 1: Background type
        ttk.Label(options_frame, text="Background:").grid(row=0, column=0, sticky=tk.W, padx=5)

        self.bg_type_var = tk.StringVar(value='color')
        ttk.Radiobutton(options_frame, text="Transparent", variable=self.bg_type_var,
                       value='transparent').grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Radiobutton(options_frame, text="Color", variable=self.bg_type_var,
                       value='color').grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Radiobutton(options_frame, text="Blur", variable=self.bg_type_var,
                       value='blur').grid(row=0, column=3, sticky=tk.W, padx=5)

        # Row 2: Color picker
        ttk.Label(options_frame, text="Color:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)

        self.color_hex = '#00FF00'  # Default green
        self.color_preview = tk.Label(options_frame, text='  #00FF00  ', bg='#00FF00',
                                       relief=tk.SUNKEN, width=12)
        self.color_preview.grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Button(options_frame, text="Pick Color", command=self._pick_color).grid(row=1, column=2, sticky=tk.W, padx=5)

        # Row 3: Blur amount
        ttk.Label(options_frame, text="Blur:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)

        self.blur_var = tk.IntVar(value=25)
        self.blur_slider = ttk.Scale(options_frame, from_=5, to=100, orient=tk.HORIZONTAL,
                                      length=200, variable=self.blur_var, command=self._on_blur_change)
        self.blur_slider.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5)

        self.blur_label = ttk.Label(options_frame, text="25 px")
        self.blur_label.grid(row=2, column=3, sticky=tk.W, padx=5)

        # Row 4: Dilation
        ttk.Label(options_frame, text="Dilation:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)

        self.dilation_var = tk.IntVar(value=0)
        self.dilation_spin = ttk.Spinbox(options_frame, from_=0, to=20, textvariable=self.dilation_var, width=5)
        self.dilation_spin.grid(row=3, column=1, sticky=tk.W, padx=5)

        # ================== FORMAT SELECTOR ==================
        format_frame = ttk.LabelFrame(self.root, text="Export Format", padding=10)
        format_frame.pack(fill=tk.X, padx=10, pady=5)

        self.format_var = tk.StringVar(value='mp4')
        ttk.Radiobutton(format_frame, text="MP4 (H.265 NVENC)", variable=self.format_var,
                       value='mp4').pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(format_frame, text="WebM (Alpha)", variable=self.format_var,
                       value='webm').pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(format_frame, text="PNG Sequence", variable=self.format_var,
                       value='png').pack(side=tk.LEFT, padx=10)

        # ================== ACTION BUTTONS ==================
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(action_frame, text="Preview", command=self._preview).pack(side=tk.LEFT, padx=10)

        self.process_btn = ttk.Button(action_frame, text="PROCESS & EXPORT",
                                       command=self._process_and_export)
        self.process_btn.pack(side=tk.LEFT, padx=10)

        ttk.Button(action_frame, text="Open Results Folder", command=self._open_results).pack(side=tk.RIGHT, padx=10)

        # ================== PROGRESS BAR ==================
        progress_frame = ttk.Frame(self.root)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                             maximum=100, length=400)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side=tk.LEFT, padx=5)

        # ================== STATUS BAR ==================
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        self.status_label = ttk.Label(status_frame, text="Ready - Load a video to start")
        self.status_label.pack(side=tk.LEFT)

        self.masks_status = ttk.Label(status_frame, text="Masks: Not loaded", foreground='gray')
        self.masks_status.pack(side=tk.RIGHT)

    # ================== VIDEO LOADING ==================

    def _load_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.webm *.mkv"), ("All files", "*.*")]
        )
        if not path:
            return

        self.video_path = path
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open video: {path}")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0

        # Calculate scale to fit canvas
        scale_w = self.canvas_width / self.video_width
        scale_h = self.canvas_height / self.video_height
        self.scale = min(scale_w, scale_h)

        # Calculate offset for centering
        scaled_w = int(self.video_width * self.scale)
        scaled_h = int(self.video_height * self.scale)
        self.offset_x = (self.canvas_width - scaled_w) // 2
        self.offset_y = (self.canvas_height - scaled_h) // 2

        # Update slider
        self.frame_slider.configure(to=self.total_frames - 1)

        # Update info
        self.video_info.configure(text=f"{self.video_width}x{self.video_height} @ {self.fps:.1f}fps")

        # Reset state
        self.click_points = []
        self.masks_dir = None
        self.masks_loaded = False
        self.current_mask = None
        self._update_points_label()
        self._update_masks_status()

        # Load first frame
        self._load_frame(0)
        self._set_status(f"Loaded: {os.path.basename(path)}")

    def _load_frame(self, idx: int):
        if not self.cap:
            return

        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()

        if ret:
            self.current_frame_idx = idx
            self.current_frame = frame

            # Load mask for this frame if available
            self._load_mask_for_frame(idx)

            self._update_canvas()
            self.frame_label.configure(text=f"Frame: {idx} / {self.total_frames - 1}")
            self.frame_slider.set(idx)

    def _load_mask_for_frame(self, idx: int):
        """Load mask for specific frame if masks are available"""
        self.current_mask = None

        if not self.masks_dir or not os.path.isdir(self.masks_dir):
            return

        mask_path = os.path.join(self.masks_dir, f"{idx:05d}.png")
        if os.path.exists(mask_path):
            self.current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    def _update_canvas(self):
        if self.current_frame is None:
            return

        # Convert to RGB
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)

        # Overlay mask if loaded (semi-transparent green)
        if self.current_mask is not None:
            mask_resized = self.current_mask
            if mask_resized.shape[:2] != frame_rgb.shape[:2]:
                mask_resized = cv2.resize(mask_resized, (frame_rgb.shape[1], frame_rgb.shape[0]))

            green_overlay = np.zeros_like(frame_rgb)
            green_overlay[:, :, 1] = mask_resized  # Green channel
            frame_rgb = cv2.addWeighted(frame_rgb, 1.0, green_overlay, 0.4, 0)

        # Draw click points
        for x, y, label in self.click_points:
            color = (0, 255, 0) if label == 1 else (255, 0, 0)  # Green for fg, Red for bg
            cv2.circle(frame_rgb, (x, y), 8, color, -1)
            cv2.circle(frame_rgb, (x, y), 10, (255, 255, 255), 2)

        # Scale frame
        scaled = cv2.resize(frame_rgb, (int(self.video_width * self.scale), int(self.video_height * self.scale)))

        # Create canvas image with black background
        canvas_img = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)

        # Place scaled frame in center
        y1 = self.offset_y
        y2 = y1 + scaled.shape[0]
        x1 = self.offset_x
        x2 = x1 + scaled.shape[1]
        canvas_img[y1:y2, x1:x2] = scaled

        # Convert to PhotoImage
        img = Image.fromarray(canvas_img)
        self.photo = ImageTk.PhotoImage(img)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def _canvas_to_video(self, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        """Convert canvas coordinates to video coordinates"""
        vx = (cx - self.offset_x) / self.scale
        vy = (cy - self.offset_y) / self.scale

        if 0 <= vx < self.video_width and 0 <= vy < self.video_height:
            return (int(vx), int(vy))
        return None

    # ================== NAVIGATION ==================

    def _on_slider(self, value):
        idx = int(float(value))
        if idx != self.current_frame_idx:
            self._load_frame(idx)

    def _prev_frame(self):
        self._load_frame(self.current_frame_idx - 1)

    def _next_frame(self):
        self._load_frame(self.current_frame_idx + 1)

    # ================== CLICK-TO-TRACK ==================

    def _on_left_click(self, event):
        """Left click = foreground point"""
        if self.current_frame is None:
            return

        pt = self._canvas_to_video(event.x, event.y)
        if pt:
            self.click_points.append((pt[0], pt[1], 1))  # label=1 for foreground
            self.tracking_frame_idx = self.current_frame_idx
            self._update_points_label()
            self._update_canvas()

    def _on_right_click(self, event):
        """Right click = background point (negative)"""
        if self.current_frame is None:
            return

        pt = self._canvas_to_video(event.x, event.y)
        if pt:
            self.click_points.append((pt[0], pt[1], 0))  # label=0 for background
            self._update_points_label()
            self._update_canvas()

    def _clear_points(self):
        self.click_points = []
        self._update_points_label()
        self._update_canvas()

    def _update_points_label(self):
        fg = sum(1 for _, _, l in self.click_points if l == 1)
        bg = sum(1 for _, _, l in self.click_points if l == 0)
        self.points_label.configure(text=f"Points: {len(self.click_points)} (FG: {fg}, BG: {bg})")

    # ================== SAM2 TRACKING ==================

    def _track_object(self):
        """Start SAM2 tracking via Celery"""
        if not self.video_path:
            messagebox.showerror("Error", "No video loaded")
            return

        if not self.click_points:
            messagebox.showerror("Error", "No points selected. Click on the object to track.")
            return

        if self.is_tracking:
            messagebox.showinfo("Info", "Already tracking...")
            return

        self.task_id = f"vtools_{uuid.uuid4().hex[:8]}"
        self.masks_dir = f"D:/watermarkz/temp_masks/{self.task_id}"
        os.makedirs(self.masks_dir, exist_ok=True)

        self._set_status("Submitting to SAM2 worker...")
        self.is_tracking = True

        def run_tracking():
            try:
                from celery import Celery

                redis_url = self._get_redis_url()
                celery = Celery('watermark', broker=redis_url, backend=redis_url)

                # Prepare points for SAM2
                points = [[x, y] for x, y, _ in self.click_points]
                labels = [l for _, _, l in self.click_points]

                self.root.after(0, lambda: self._set_status("Sending to SAM2 (wsl_sam2_local)..."))

                # Send task to SAM2 worker
                result = celery.send_task(
                    'sam2.generate_masks_fullfps',
                    args=[self.video_path, self.masks_dir],
                    kwargs={
                        'prompt_mode': 'point',
                        'points': points,
                        'labels': labels,
                        'frame_idx': self.tracking_frame_idx,
                        'api_base': None
                    },
                    queue='wsl_sam2_local'
                )

                self.celery_task = result
                self.root.after(0, lambda: self._set_status(f"Tracking... Task ID: {result.id}"))

                # Poll for completion
                while not result.ready():
                    time.sleep(1)
                    # Check if masks are being created
                    mask_count = len([f for f in os.listdir(self.masks_dir) if f.endswith('.png')]) if os.path.exists(self.masks_dir) else 0
                    progress = min(95, (mask_count / max(self.total_frames, 1)) * 100)
                    self.root.after(0, lambda p=progress, m=mask_count: self._update_progress(p, f"Tracking: {m}/{self.total_frames} frames"))

                # Check result
                if result.successful():
                    self.masks_loaded = True
                    self.root.after(0, lambda: self._on_tracking_complete())
                else:
                    error = str(result.result) if result.result else "Unknown error"
                    self.root.after(0, lambda e=error: self._on_tracking_error(e))

            except Exception as e:
                self.root.after(0, lambda: self._on_tracking_error(str(e)))
            finally:
                self.is_tracking = False

        threading.Thread(target=run_tracking, daemon=True).start()

    def _on_tracking_complete(self):
        self.masks_loaded = True
        self._update_masks_status()
        self._update_progress(100, "Tracking complete!")
        self._load_mask_for_frame(self.current_frame_idx)
        self._update_canvas()
        self._set_status("Tracking complete! Masks loaded.")
        messagebox.showinfo("Success", f"SAM2 tracking complete!\n\nMasks saved to: {self.masks_dir}\n\nYou can now preview and export.")

    def _on_tracking_error(self, error: str):
        self._set_status(f"Tracking error: {error}")
        self._update_progress(0, "Error")
        messagebox.showerror("Tracking Error", f"SAM2 tracking failed:\n{error}")

    def _update_masks_status(self):
        if self.masks_loaded and self.masks_dir:
            mask_count = len([f for f in os.listdir(self.masks_dir) if f.endswith('.png')]) if os.path.exists(self.masks_dir) else 0
            self.masks_status.configure(text=f"Masks: {mask_count} loaded", foreground='green')
        else:
            self.masks_status.configure(text="Masks: Not loaded", foreground='gray')

    # ================== OPTIONS ==================

    def _pick_color(self):
        color = colorchooser.askcolor(color=self.color_hex, title="Choose Color")
        if color[1]:
            self.color_hex = color[1]
            self.color_preview.configure(text=f'  {self.color_hex}  ', bg=self.color_hex)

    def _on_blur_change(self, value):
        val = int(float(value))
        self.blur_label.configure(text=f"{val} px")

    # ================== PREVIEW ==================

    def _preview(self):
        """Preview operation on current frame"""
        if self.current_frame is None:
            messagebox.showerror("Error", "No video loaded")
            return

        if self.current_mask is None:
            messagebox.showerror("Error", "No mask loaded. Track an object first.")
            return

        operation = self.operation_var.get()
        options = {
            'bg_type': self.bg_type_var.get(),
            'color_bgr': hex_to_bgr(self.color_hex),
            'blur_amount': self.blur_var.get()
        }

        # Apply dilation if set
        mask = self.current_mask.copy()
        dilation = self.dilation_var.get()
        if dilation > 0:
            mask = dilate_mask(mask, dilation)

        # Apply operation
        result = apply_operation(self.current_frame, mask, operation, options)

        # Show preview window
        preview = tk.Toplevel(self.root)
        preview.title("Preview")

        # Handle RGBA for transparent
        if result.shape[2] == 4:
            # Create checkerboard background for alpha preview
            h, w = result.shape[:2]
            checker = np.zeros((h, w, 3), dtype=np.uint8)
            block = 20
            for i in range(0, h, block):
                for j in range(0, w, block):
                    if ((i // block) + (j // block)) % 2 == 0:
                        checker[i:i+block, j:j+block] = [200, 200, 200]
                    else:
                        checker[i:i+block, j:j+block] = [150, 150, 150]

            # Blend with alpha
            alpha = result[:, :, 3:4].astype(np.float32) / 255.0
            rgb = result[:, :, :3].astype(np.float32)
            blended = (rgb * alpha + checker.astype(np.float32) * (1 - alpha)).astype(np.uint8)
            result_display = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        else:
            result_display = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Scale for display
        scale = min(900 / self.video_width, 700 / self.video_height)
        preview_w = int(self.video_width * scale)
        preview_h = int(self.video_height * scale)

        result_scaled = cv2.resize(result_display, (preview_w, preview_h))

        img = Image.fromarray(result_scaled)
        photo = ImageTk.PhotoImage(img)

        label = ttk.Label(preview, image=photo)
        label.image = photo
        label.pack()

        ttk.Label(preview, text=f"Operation: {operation} | Format: {self.format_var.get()}").pack()

    # ================== EXPORT ==================

    def _process_and_export(self):
        """Process all frames and export"""
        if not self.video_path:
            messagebox.showerror("Error", "No video loaded")
            return

        if not self.masks_loaded or not self.masks_dir:
            messagebox.showerror("Error", "No masks loaded. Track an object first.")
            return

        if self.is_exporting:
            messagebox.showinfo("Info", "Already exporting...")
            return

        # Get output path
        output_format = self.format_var.get()
        if output_format == 'png':
            output_dir = filedialog.askdirectory(title="Select Output Directory for PNG Sequence")
            if not output_dir:
                return
            output_path = output_dir
        else:
            ext = 'webm' if output_format == 'webm' else 'mp4'
            default_name = f"{os.path.splitext(os.path.basename(self.video_path))[0]}_processed.{ext}"
            output_path = filedialog.asksaveasfilename(
                title="Save Output Video",
                defaultextension=f".{ext}",
                initialfile=default_name,
                filetypes=[(f"{ext.upper()} files", f"*.{ext}"), ("All files", "*.*")]
            )
            if not output_path:
                return

        self.is_exporting = True
        self._set_status("Processing...")

        def run_export():
            try:
                operation = self.operation_var.get()
                options = {
                    'bg_type': self.bg_type_var.get(),
                    'color_bgr': hex_to_bgr(self.color_hex),
                    'blur_amount': self.blur_var.get()
                }
                dilation = self.dilation_var.get()
                output_format = self.format_var.get()

                # Create temp directory for processed frames
                temp_frames_dir = f"D:/watermarkz/temp_output/{self.task_id}_frames"
                os.makedirs(temp_frames_dir, exist_ok=True)

                # Process each frame
                cap = cv2.VideoCapture(self.video_path)
                frame_idx = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Load mask
                    mask_path = os.path.join(self.masks_dir, f"{frame_idx:05d}.png")
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if dilation > 0:
                            mask = dilate_mask(mask, dilation)
                    else:
                        mask = np.zeros((self.video_height, self.video_width), dtype=np.uint8)

                    # Apply operation
                    result = apply_operation(frame, mask, operation, options)

                    # Save frame
                    if output_format == 'webm' and options['bg_type'] == 'transparent':
                        # Save as RGBA PNG for WebM alpha
                        cv2.imwrite(os.path.join(temp_frames_dir, f"{frame_idx:05d}.png"), result)
                    else:
                        # Save as BGR PNG
                        if result.shape[2] == 4:
                            result = result[:, :, :3]  # Drop alpha
                        cv2.imwrite(os.path.join(temp_frames_dir, f"{frame_idx:05d}.png"), result)

                    frame_idx += 1
                    progress = (frame_idx / self.total_frames) * 80  # 80% for processing
                    self.root.after(0, lambda p=progress, f=frame_idx: self._update_progress(p, f"Processing: {f}/{self.total_frames}"))

                cap.release()

                # Export
                self.root.after(0, lambda: self._update_progress(85, "Encoding..."))

                if output_format == 'mp4':
                    success = export_mp4_nvenc(temp_frames_dir, output_path, self.fps,
                                               self.video_width, self.video_height)
                elif output_format == 'webm':
                    success = export_webm_alpha(temp_frames_dir, output_path, self.fps)
                else:  # png
                    # Just copy frames to output directory
                    for f in os.listdir(temp_frames_dir):
                        shutil.copy(os.path.join(temp_frames_dir, f), os.path.join(output_path, f))
                    success = True

                # Cleanup temp frames
                shutil.rmtree(temp_frames_dir, ignore_errors=True)

                if success:
                    self.root.after(0, lambda: self._on_export_complete(output_path))
                else:
                    self.root.after(0, lambda: self._on_export_error("Encoding failed"))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self._on_export_error(str(e)))
            finally:
                self.is_exporting = False

        threading.Thread(target=run_export, daemon=True).start()

    def _on_export_complete(self, output_path: str):
        self._update_progress(100, "Export complete!")
        self._set_status(f"Export complete: {output_path}")
        messagebox.showinfo("Success", f"Export complete!\n\nSaved to: {output_path}")

    def _on_export_error(self, error: str):
        self._update_progress(0, "Error")
        self._set_status(f"Export error: {error}")
        messagebox.showerror("Export Error", f"Export failed:\n{error}")

    # ================== UTILITIES ==================

    def _update_progress(self, value: float, text: str = None):
        self.progress_var.set(value)
        self.progress_label.configure(text=f"{int(value)}%")
        if text:
            self._set_status(text)

    def _set_status(self, msg: str):
        self.status_label.configure(text=msg)
        self.root.update_idletasks()

    def _open_results(self):
        results_dir = "D:/watermarkz/results"
        os.makedirs(results_dir, exist_ok=True)
        os.startfile(results_dir)

    def _get_redis_url(self) -> str:
        """Load Redis URL from file or use default"""
        if os.path.exists('redis_url.txt'):
            with open('redis_url.txt', 'r') as f:
                return f.read().strip()
        return 'redis://:watermarkz_secure_2024@localhost:6379/0'


# ============================================================
# Main
# ============================================================

def main():
    root = tk.Tk()
    app = VideoToolsGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
