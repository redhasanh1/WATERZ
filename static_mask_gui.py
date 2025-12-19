"""
Static Mask GUI - Local Testing Tool for Watermark Removal

Draw shapes on video frames to create static masks, then process with ProPainter.
Bypasses web UI and SAM2 - sends directly to local Celery queue.

Usage:
    python static_mask_gui.py

Requirements:
    - 4090_receiver.bat running (ProPainter worker)
    - Redis running locally
"""

import sys
import os

# Add script directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import uuid
import json
import threading
from typing import List, Dict, Optional, Tuple

# Import mask generator
from static_mask_generator import generate_mask_gpu, generate_combined_mask, dilate_mask


class Shape:
    """Base class for drawable shapes"""
    def __init__(self, shape_type: str):
        self.type = shape_type
        self.color = (0, 255, 0)  # Green for preview

    def to_dict(self) -> Dict:
        raise NotImplementedError

    def draw_preview(self, canvas, scale: float, offset: Tuple[int, int]):
        raise NotImplementedError


class Rectangle(Shape):
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        super().__init__('rectangle')
        self.x1, self.y1 = min(x1, x2), min(y1, y2)
        self.x2, self.y2 = max(x1, x2), max(y1, y2)

    def to_dict(self) -> Dict:
        return {
            'type': 'rectangle',
            'x': self.x1,
            'y': self.y1,
            'width': self.x2 - self.x1,
            'height': self.y2 - self.y1
        }

    def draw_on_image(self, img: np.ndarray, color=(0, 255, 0), thickness=2):
        cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), color, thickness)


class Ellipse(Shape):
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        super().__init__('ellipse')
        self.cx = (x1 + x2) // 2
        self.cy = (y1 + y2) // 2
        self.rx = abs(x2 - x1) // 2
        self.ry = abs(y2 - y1) // 2

    def to_dict(self) -> Dict:
        return {
            'type': 'ellipse',
            'cx': self.cx,
            'cy': self.cy,
            'rx': self.rx,
            'ry': self.ry
        }

    def draw_on_image(self, img: np.ndarray, color=(0, 255, 0), thickness=2):
        cv2.ellipse(img, (self.cx, self.cy), (self.rx, self.ry), 0, 0, 360, color, thickness)


class Polygon(Shape):
    def __init__(self, points: List[Tuple[int, int]]):
        super().__init__('polygon')
        self.points = points

    def to_dict(self) -> Dict:
        return {
            'type': 'polygon',
            'points': [{'x': p[0], 'y': p[1]} for p in self.points]
        }

    def draw_on_image(self, img: np.ndarray, color=(0, 255, 0), thickness=2):
        if len(self.points) > 1:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


class BrushStroke(Shape):
    def __init__(self, points: List[Tuple[int, int]], brush_size: int = 20):
        super().__init__('brush')
        self.points = points
        self.brush_size = brush_size

    def to_dict(self) -> Dict:
        return {
            'type': 'brush',
            'points': [{'x': p[0], 'y': p[1]} for p in self.points],
            'brushSize': self.brush_size
        }

    def draw_on_image(self, img: np.ndarray, color=(0, 255, 0), thickness=None):
        if len(self.points) > 1:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=False, color=color, thickness=self.brush_size)


class StaticMaskGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Mask GUI - Static & Tracked Modes")
        self.root.geometry("1200x800")

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

        # Drawing state
        self.shapes: List[Shape] = []
        self.current_tool = 'rectangle'
        self.brush_size = 20
        self.is_drawing = False
        self.draw_start = None
        self.polygon_points: List[Tuple[int, int]] = []
        self.brush_points: List[Tuple[int, int]] = []

        # Processing state
        self.task_id: Optional[str] = None
        self.is_processing = False

        self._create_ui()

    def _create_ui(self):
        # Top bar - video controls
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(top_frame, text="Load Video", command=self._load_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="<< Prev", command=self._prev_frame).pack(side=tk.LEFT, padx=2)

        self.frame_label = ttk.Label(top_frame, text="Frame: 0 / 0")
        self.frame_label.pack(side=tk.LEFT, padx=10)

        ttk.Button(top_frame, text="Next >>", command=self._next_frame).pack(side=tk.LEFT, padx=2)

        # Frame slider
        self.frame_slider = ttk.Scale(top_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=300, command=self._on_slider)
        self.frame_slider.pack(side=tk.LEFT, padx=10)

        # Video info
        self.video_info = ttk.Label(top_frame, text="No video loaded")
        self.video_info.pack(side=tk.RIGHT, padx=10)

        # Mode selector bar
        mode_frame = ttk.Frame(self.root)
        mode_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(mode_frame, text="Mode:", font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT, padx=5)

        self.mode_var = tk.StringVar(value='static')
        ttk.Radiobutton(mode_frame, text="Static (fixed mask, no tracking)",
                       variable=self.mode_var, value='static').pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="Tracked (SAM2 detection + tracking)",
                       variable=self.mode_var, value='tracked').pack(side=tk.LEFT, padx=10)

        # Main canvas
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<Button-3>", self._on_right_click)  # Right click to cancel polygon

        # Tools bar
        tools_frame = ttk.Frame(self.root)
        tools_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(tools_frame, text="Tools:").pack(side=tk.LEFT, padx=5)

        self.tool_var = tk.StringVar(value='rectangle')
        tools = [('Rectangle', 'rectangle'), ('Ellipse', 'ellipse'), ('Polygon', 'polygon'), ('Brush', 'brush')]
        for text, value in tools:
            ttk.Radiobutton(tools_frame, text=text, variable=self.tool_var, value=value,
                           command=self._on_tool_change).pack(side=tk.LEFT, padx=5)

        ttk.Separator(tools_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(tools_frame, text="Brush Size:").pack(side=tk.LEFT, padx=5)
        self.brush_slider = ttk.Scale(tools_frame, from_=5, to=100, orient=tk.HORIZONTAL, length=150,
                                      command=self._on_brush_size)
        self.brush_slider.set(20)
        self.brush_slider.pack(side=tk.LEFT, padx=5)
        self.brush_label = ttk.Label(tools_frame, text="20px")
        self.brush_label.pack(side=tk.LEFT)

        ttk.Separator(tools_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(tools_frame, text="Dilation:").pack(side=tk.LEFT, padx=5)
        self.dilation_var = tk.IntVar(value=5)
        self.dilation_spin = ttk.Spinbox(tools_frame, from_=0, to=20, textvariable=self.dilation_var, width=5)
        self.dilation_spin.pack(side=tk.LEFT, padx=5)

        # Actions bar
        actions_frame = ttk.Frame(self.root)
        actions_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(actions_frame, text="Clear All Shapes", command=self._clear_shapes).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Undo Last", command=self._undo_shape).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Preview Mask", command=self._preview_mask).pack(side=tk.LEFT, padx=5)

        ttk.Separator(actions_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.process_btn = ttk.Button(actions_frame, text="Process Video", command=self._process_video)
        self.process_btn.pack(side=tk.LEFT, padx=5)

        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        self.status_label = ttk.Label(status_frame, text="Ready - Load a video to start")
        self.status_label.pack(side=tk.LEFT)

        self.shapes_label = ttk.Label(status_frame, text="Shapes: 0")
        self.shapes_label.pack(side=tk.RIGHT)

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

        # Clear shapes
        self.shapes = []
        self._update_shapes_label()

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
            self._update_canvas()
            self.frame_label.configure(text=f"Frame: {idx} / {self.total_frames - 1}")
            self.frame_slider.set(idx)

    def _update_canvas(self):
        if self.current_frame is None:
            return

        # Convert to RGB and create copy for drawing
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)

        # Draw shapes on frame
        for shape in self.shapes:
            shape.draw_on_image(frame_rgb, color=(0, 255, 0), thickness=2)

        # Draw current polygon points if in polygon mode
        if self.current_tool == 'polygon' and self.polygon_points:
            pts = np.array(self.polygon_points, dtype=np.int32)
            cv2.polylines(frame_rgb, [pts], isClosed=False, color=(255, 255, 0), thickness=2)
            for pt in self.polygon_points:
                cv2.circle(frame_rgb, pt, 5, (255, 0, 0), -1)

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
        # Check if within video area
        vx = (cx - self.offset_x) / self.scale
        vy = (cy - self.offset_y) / self.scale

        if 0 <= vx < self.video_width and 0 <= vy < self.video_height:
            return (int(vx), int(vy))
        return None

    def _on_mouse_down(self, event):
        if self.current_frame is None:
            return

        pt = self._canvas_to_video(event.x, event.y)
        if not pt:
            return

        self.current_tool = self.tool_var.get()

        if self.current_tool in ('rectangle', 'ellipse'):
            self.is_drawing = True
            self.draw_start = pt
        elif self.current_tool == 'polygon':
            # Auto-close polygon if clicking near start point (within 15px)
            if len(self.polygon_points) >= 3:
                start = self.polygon_points[0]
                dist = ((pt[0] - start[0])**2 + (pt[1] - start[1])**2)**0.5
                if dist < 15:
                    # Close polygon
                    shape = Polygon(self.polygon_points.copy())
                    self.shapes.append(shape)
                    self.polygon_points = []
                    self._update_shapes_label()
                    self._update_canvas()
                    return
            self.polygon_points.append(pt)
            self._update_canvas()
        elif self.current_tool == 'brush':
            self.is_drawing = True
            self.brush_points = [pt]

    def _on_mouse_move(self, event):
        if not self.is_drawing:
            return

        pt = self._canvas_to_video(event.x, event.y)
        if not pt:
            return

        if self.current_tool == 'brush':
            self.brush_points.append(pt)
            # Draw preview
            if self.current_frame is not None:
                frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                for shape in self.shapes:
                    shape.draw_on_image(frame_rgb)
                # Draw current brush stroke
                if len(self.brush_points) > 1:
                    pts = np.array(self.brush_points, dtype=np.int32)
                    cv2.polylines(frame_rgb, [pts], isClosed=False, color=(255, 255, 0), thickness=self.brush_size)
                self._update_canvas_with_frame(frame_rgb)

        elif self.current_tool in ('rectangle', 'ellipse') and self.draw_start:
            # Draw preview
            if self.current_frame is not None:
                frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                for shape in self.shapes:
                    shape.draw_on_image(frame_rgb)
                # Draw current shape preview
                x1, y1 = self.draw_start
                x2, y2 = pt
                if self.current_tool == 'rectangle':
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 255, 0), 2)
                else:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    rx, ry = abs(x2 - x1) // 2, abs(y2 - y1) // 2
                    cv2.ellipse(frame_rgb, (cx, cy), (rx, ry), 0, 0, 360, (255, 255, 0), 2)
                self._update_canvas_with_frame(frame_rgb)

    def _update_canvas_with_frame(self, frame_rgb: np.ndarray):
        """Update canvas with given RGB frame"""
        scaled = cv2.resize(frame_rgb, (int(self.video_width * self.scale), int(self.video_height * self.scale)))
        canvas_img = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        y1 = self.offset_y
        y2 = y1 + scaled.shape[0]
        x1 = self.offset_x
        x2 = x1 + scaled.shape[1]
        canvas_img[y1:y2, x1:x2] = scaled
        img = Image.fromarray(canvas_img)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def _on_mouse_up(self, event):
        if not self.is_drawing:
            return

        pt = self._canvas_to_video(event.x, event.y)

        if self.current_tool == 'rectangle' and self.draw_start and pt:
            shape = Rectangle(self.draw_start[0], self.draw_start[1], pt[0], pt[1])
            self.shapes.append(shape)
            self._update_shapes_label()

        elif self.current_tool == 'ellipse' and self.draw_start and pt:
            shape = Ellipse(self.draw_start[0], self.draw_start[1], pt[0], pt[1])
            self.shapes.append(shape)
            self._update_shapes_label()

        elif self.current_tool == 'brush' and len(self.brush_points) > 1:
            shape = BrushStroke(self.brush_points, self.brush_size)
            self.shapes.append(shape)
            self._update_shapes_label()

        self.is_drawing = False
        self.draw_start = None
        self.brush_points = []
        self._update_canvas()

    def _on_double_click(self, event):
        """Double click to close polygon"""
        if self.current_tool == 'polygon' and len(self.polygon_points) >= 3:
            shape = Polygon(self.polygon_points.copy())
            self.shapes.append(shape)
            self.polygon_points = []
            self._update_shapes_label()
            self._update_canvas()

    def _on_right_click(self, event):
        """Right click to cancel polygon"""
        if self.current_tool == 'polygon':
            self.polygon_points = []
            self._update_canvas()

    def _on_tool_change(self):
        # Clear any in-progress polygon
        self.polygon_points = []
        self._update_canvas()

    def _on_brush_size(self, value):
        self.brush_size = int(float(value))
        self.brush_label.configure(text=f"{self.brush_size}px")

    def _on_slider(self, value):
        idx = int(float(value))
        if idx != self.current_frame_idx:
            self._load_frame(idx)

    def _prev_frame(self):
        self._load_frame(self.current_frame_idx - 1)

    def _next_frame(self):
        self._load_frame(self.current_frame_idx + 1)

    def _clear_shapes(self):
        self.shapes = []
        self.polygon_points = []
        self._update_shapes_label()
        self._update_canvas()

    def _undo_shape(self):
        if self.shapes:
            self.shapes.pop()
            self._update_shapes_label()
            self._update_canvas()

    def _update_shapes_label(self):
        self.shapes_label.configure(text=f"Shapes: {len(self.shapes)}")

    def _set_status(self, msg: str):
        self.status_label.configure(text=msg)
        self.root.update_idletasks()

    def _shapes_to_bbox(self) -> Tuple[int, int, int, int]:
        """Convert all shapes to a single bounding box [x1, y1, x2, y2]"""
        if not self.shapes:
            return (0, 0, 0, 0)

        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0

        for shape in self.shapes:
            if isinstance(shape, Rectangle):
                min_x = min(min_x, shape.x1)
                min_y = min(min_y, shape.y1)
                max_x = max(max_x, shape.x2)
                max_y = max(max_y, shape.y2)
            elif isinstance(shape, Ellipse):
                min_x = min(min_x, shape.cx - shape.rx)
                min_y = min(min_y, shape.cy - shape.ry)
                max_x = max(max_x, shape.cx + shape.rx)
                max_y = max(max_y, shape.cy + shape.ry)
            elif isinstance(shape, (Polygon, BrushStroke)):
                for pt in shape.points:
                    min_x = min(min_x, pt[0])
                    min_y = min(min_y, pt[1])
                    max_x = max(max_x, pt[0])
                    max_y = max(max_y, pt[1])
                # Add brush size padding for brush strokes
                if isinstance(shape, BrushStroke):
                    min_x -= shape.brush_size // 2
                    min_y -= shape.brush_size // 2
                    max_x += shape.brush_size // 2
                    max_y += shape.brush_size // 2

        # Clamp to video bounds
        min_x = max(0, int(min_x))
        min_y = max(0, int(min_y))
        max_x = min(self.video_width, int(max_x))
        max_y = min(self.video_height, int(max_y))

        return (min_x, min_y, max_x, max_y)

    def _preview_mask(self):
        if not self.shapes:
            messagebox.showinfo("Info", "No shapes to preview. Draw some shapes first.")
            return

        # Generate mask
        shape_dicts = [s.to_dict() for s in self.shapes]
        mask = generate_combined_mask(shape_dicts, self.video_width, self.video_height)

        # Apply dilation
        dilation = self.dilation_var.get()
        if dilation > 0:
            mask = dilate_mask(mask, dilation)

        # Show preview window
        preview = tk.Toplevel(self.root)
        preview.title("Mask Preview")

        # Scale mask for display
        scale = min(800 / self.video_width, 600 / self.video_height)
        preview_w = int(self.video_width * scale)
        preview_h = int(self.video_height * scale)

        mask_scaled = cv2.resize(mask, (preview_w, preview_h))
        mask_rgb = cv2.cvtColor(mask_scaled, cv2.COLOR_GRAY2RGB)

        img = Image.fromarray(mask_rgb)
        photo = ImageTk.PhotoImage(img)

        label = ttk.Label(preview, image=photo)
        label.image = photo  # Keep reference
        label.pack()

        ttk.Label(preview, text=f"Size: {self.video_width}x{self.video_height}, Dilation: {dilation}").pack()

    def _process_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "No video loaded")
            return

        if not self.shapes:
            messagebox.showerror("Error", "No shapes drawn. Draw at least one shape.")
            return

        if self.is_processing:
            messagebox.showinfo("Info", "Already processing...")
            return

        mode = self.mode_var.get()
        prefix = "static" if mode == "static" else "tracked"
        self.task_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
        masks_dir = f"D:/watermarkz/temp_masks/{self.task_id}"
        os.makedirs(masks_dir, exist_ok=True)

        if mode == 'static':
            self._process_static_mode(masks_dir)
        else:
            self._process_tracked_mode(masks_dir)

    def _process_static_mode(self, masks_dir: str):
        """Static mode: Generate mask locally, upload video+masks to B2, send to ProPainter"""
        self._set_status("Generating static mask...")
        shape_dicts = [s.to_dict() for s in self.shapes]
        mask = generate_combined_mask(shape_dicts, self.video_width, self.video_height)

        # Apply dilation
        dilation = self.dilation_var.get()
        if dilation > 0:
            mask = dilate_mask(mask, dilation)

        # Replicate mask for ALL frames (Docker expects 1 mask per frame)
        self._set_status(f"Replicating mask for {self.total_frames} frames...")
        for i in range(self.total_frames):
            mask_path = os.path.join(masks_dir, f"{i:05d}.png")
            cv2.imwrite(mask_path, mask)
        print(f"[STATIC] Wrote {self.total_frames} mask files (same mask replicated)")
        self._set_status(f"Mask saved, uploading to B2...")

        def send_task():
            try:
                import zipfile
                import time
                import shutil
                from celery import Celery

                # Upload video to B2 first
                self.root.after(0, lambda: self._set_status(f"Uploading video to B2..."))
                video_url = self._upload_video_to_b2()
                if not video_url:
                    raise RuntimeError("Failed to upload video to B2")

                # Upload masks to B2
                self.root.after(0, lambda: self._set_status(f"Uploading masks to B2..."))
                masks_url = self._upload_masks_to_b2(masks_dir)
                if not masks_url:
                    raise RuntimeError("Failed to upload masks to B2")

                self.root.after(0, lambda: self._set_status(f"Uploaded to B2, sending task..."))

                redis_url = self._get_redis_url()
                celery = Celery('watermark', broker=redis_url)

                # Create sam2_result with B2 URL (Docker can download it)
                sam2_result = {
                    'masks_url': masks_url,
                    'mode': 'static'
                }

                # Send task with video URL (not local path!)
                celery.send_task(
                    'watermark._continue_after_masks',
                    args=[sam2_result, video_url, self.task_id, [], self.video_width, self.video_height, 0, None],
                    queue='propainter'
                )

                self.root.after(0, lambda: self._set_status(f"Static task submitted: {self.task_id}"))
                self.root.after(0, lambda: messagebox.showinfo("Success",
                    f"STATIC mode task submitted!\n\nTask ID: {self.task_id}\nVideo URL: {video_url}\nMasks URL: {masks_url}\n\nCheck Docker worker for progress."))

            except Exception as e:
                self.root.after(0, lambda: self._set_status(f"Error: {str(e)}"))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to send task:\n{str(e)}"))
            finally:
                self.is_processing = False

        self.is_processing = True
        threading.Thread(target=send_task, daemon=True).start()

    def _upload_masks_to_b2(self, masks_dir: str) -> Optional[str]:
        """Upload masks to B2 CDN, return URL"""
        import zipfile
        import time
        import shutil
        import glob

        try:
            from b2sdk.v2 import B2Api, InMemoryAccountInfo
        except ImportError:
            print("[B2] b2sdk not installed, run: pip install b2sdk")
            return None

        # B2 credentials (same as 4090_sender.bat)
        B2_KEY_ID = os.environ.get('B2_KEY_ID', '00539db5c1104b50000000003')
        B2_APP_KEY = os.environ.get('B2_APP_KEY', 'K005384b8lPoBT11wScxkZ2Gx0fszus')
        B2_BUCKET = os.environ.get('B2_BUCKET', 'watermarkz')
        B2_CDN_URL = os.environ.get('B2_CDN_URL', 'https://markz.humblewoslayer.workers.dev')

        # Zip masks
        zip_path = os.path.join(os.path.dirname(masks_dir), f"{self.task_id}_masks.zip")
        mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        print(f"[B2] Zipping {len(mask_files)} masks...")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for mask_file in mask_files:
                zf.write(mask_file, os.path.basename(mask_file))

        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"[B2] Zip created: {zip_size_mb:.2f} MB")

        # Upload to B2
        print(f"[B2] Uploading to {B2_BUCKET}...")
        upload_start = time.time()

        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
        bucket = b2_api.get_bucket_by_name(B2_BUCKET)

        timestamp = int(time.time())
        remote_path = f"masks/{timestamp}_{self.task_id}_masks.zip"
        bucket.upload_local_file(local_file=zip_path, file_name=remote_path)

        masks_url = f"{B2_CDN_URL}/{remote_path}"
        upload_time = time.time() - upload_start
        print(f"[B2] Upload complete in {upload_time:.1f}s: {masks_url}")

        # Cleanup local files
        shutil.rmtree(masks_dir, ignore_errors=True)
        os.remove(zip_path)
        print(f"[B2] Local masks cleaned up")

        return masks_url

    def _upload_video_to_b2(self) -> Optional[str]:
        """Upload video to B2 CDN, return URL"""
        import time

        try:
            from b2sdk.v2 import B2Api, InMemoryAccountInfo
        except ImportError:
            print("[B2] b2sdk not installed, run: pip install b2sdk")
            return None

        # B2 credentials (same as masks upload)
        B2_KEY_ID = os.environ.get('B2_KEY_ID', '00539db5c1104b50000000003')
        B2_APP_KEY = os.environ.get('B2_APP_KEY', 'K005384b8lPoBT11wScxkZ2Gx0fszus')
        B2_BUCKET = os.environ.get('B2_BUCKET', 'watermarkz')
        B2_CDN_URL = os.environ.get('B2_CDN_URL', 'https://markz.humblewoslayer.workers.dev')

        video_size_mb = os.path.getsize(self.video_path) / (1024 * 1024)
        print(f"[B2] Uploading video ({video_size_mb:.2f} MB)...")

        upload_start = time.time()

        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
        bucket = b2_api.get_bucket_by_name(B2_BUCKET)

        timestamp = int(time.time())
        video_filename = os.path.basename(self.video_path)
        remote_path = f"videos/{timestamp}_{self.task_id}_{video_filename}"
        bucket.upload_local_file(local_file=self.video_path, file_name=remote_path)

        video_url = f"{B2_CDN_URL}/{remote_path}"
        upload_time = time.time() - upload_start
        print(f"[B2] Video upload complete in {upload_time:.1f}s: {video_url}")

        return video_url

    def _process_tracked_mode(self, masks_dir: str):
        """Tracked mode: Upload video to B2, send to SAM2 → ProPainter chain"""
        bbox = self._shapes_to_bbox()
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            messagebox.showerror("Error", "Invalid bounding box from shapes")
            return

        self._set_status(f"Uploading video to B2 for tracking... bbox={bbox}")

        def send_task():
            try:
                from celery import Celery, chain, signature

                # Upload video to B2 first (Docker can't access local Windows paths!)
                self.root.after(0, lambda: self._set_status(f"Uploading video to B2..."))
                video_url = self._upload_video_to_b2()
                if not video_url:
                    raise RuntimeError("Failed to upload video to B2")

                self.root.after(0, lambda: self._set_status(f"Sending to SAM2 for tracking..."))

                redis_url = self._get_redis_url()
                celery = Celery('watermark', broker=redis_url)

                # Chain: SAM2 generates masks → ProPainter inpaints
                # Use Docker temp path (not Windows path!)
                docker_masks_dir = f"/tmp/{self.task_id}_sam2_masks"
                s1 = signature(
                    'sam2.generate_masks_fullfps',
                    args=[video_url, docker_masks_dir],  # video URL, Docker temp path
                    kwargs={
                        'prompt_mode': 'bbox',
                        'bbox': list(bbox),
                        'frame_idx': self.current_frame_idx,
                        'api_base': None
                    },
                    queue='wsl_sam2'
                )
                s2 = signature(
                    'watermark._continue_after_masks',
                    args=[video_url, self.task_id, [], self.video_width, self.video_height, self.current_frame_idx, None],
                    queue='propainter'
                )

                chain(s1, s2).apply_async(task_id=self.task_id)

                self.root.after(0, lambda: self._set_status(f"Tracked task submitted: {self.task_id}"))
                self.root.after(0, lambda: messagebox.showinfo("Success",
                    f"TRACKED mode task submitted!\n\nTask ID: {self.task_id}\nVideo URL: {video_url}\nBBox: {bbox}\nFrame: {self.current_frame_idx}\n\nSAM2 will track object and upload masks to B2."))

            except Exception as e:
                self.root.after(0, lambda: self._set_status(f"Error: {str(e)}"))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to send task:\n{str(e)}"))
            finally:
                self.is_processing = False

        self.is_processing = True
        threading.Thread(target=send_task, daemon=True).start()

    def _get_redis_url(self) -> str:
        """Load Redis URL from file or use default"""
        if os.path.exists('redis_url.txt'):
            with open('redis_url.txt', 'r') as f:
                return f.read().strip()
        return 'redis://:watermarkz_secure_2024@localhost:6379/0'


def main():
    root = tk.Tk()
    app = StaticMaskGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
