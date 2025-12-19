/**
 * MaskDrawer - Canvas Drawing Tools for Mask Selection
 * Supports: Rectangle, Ellipse, Polygon, Freehand Brush
 * Works with SAM2Selector for tracked mode or standalone for static mode
 */
class MaskDrawer {
    constructor(videoElement, canvasElement, options = {}) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');

        this.options = {
            mode: 'tracked',        // 'tracked' or 'static'
            tool: 'click',          // 'click', 'rectangle', 'ellipse', 'polygon', 'brush'
            brushSize: 20,
            fillColor: 'rgba(102, 126, 234, 0.4)',
            strokeColor: '#667eea',
            strokeWidth: 2,
            onShapeComplete: null,  // callback(shapeData)
            onModeChange: null,     // callback(mode)
            onToolChange: null,     // callback(tool)
            sam2Selector: null,     // Reference to SAM2Selector for click passthrough
            ...options
        };

        // Drawing state
        this.isDrawing = false;
        this.currentShape = null;
        this.shapes = [];           // Completed shapes
        this.polygonPoints = [];    // For polygon tool
        this.brushPath = [];        // For brush tool
        this.eraserPath = [];       // For eraser tool
        this.startPoint = null;
        this.currentPoint = null;

        // Raster mask bitmap (for pixel-level editing with eraser)
        this.maskBitmap = null;     // Canvas element for bitmap operations
        this.maskCtx = null;        // 2D context for bitmap

        // Video dimensions (native resolution)
        this.videoWidth = 0;
        this.videoHeight = 0;

        // Display scale factors
        this.displayScaleX = 1;
        this.displayScaleY = 1;
        this.offsetX = 0;
        this.offsetY = 0;

        // Bound event handlers
        this._handleMouseDown = this.handleMouseDown.bind(this);
        this._handleMouseMove = this.handleMouseMove.bind(this);
        this._handleMouseUp = this.handleMouseUp.bind(this);
        this._handleDblClick = this.handleDblClick.bind(this);
        this._handleKeyDown = this.handleKeyDown.bind(this);

        this.enabled = false;
    }

    /**
     * Initialize the drawer and attach event listeners
     */
    init() {
        if (!this.video || !this.canvas) {
            console.error('MaskDrawer: Video or canvas element not found');
            return;
        }

        this.updateDimensions();
        this.attachListeners();
        console.log('MaskDrawer initialized with tool:', this.options.tool);
    }

    /**
     * Update video dimensions and scale factors
     */
    updateDimensions() {
        this.videoWidth = this.video.videoWidth || 1920;
        this.videoHeight = this.video.videoHeight || 1080;

        // Get display dimensions
        const rect = this.video.getBoundingClientRect();
        const displayWidth = rect.width;
        const displayHeight = rect.height;

        // Calculate letterbox/pillarbox offsets
        const videoAspect = this.videoWidth / this.videoHeight;
        const displayAspect = displayWidth / displayHeight;

        if (videoAspect > displayAspect) {
            // Pillarbox (black bars top/bottom)
            const scaledHeight = displayWidth / videoAspect;
            this.offsetY = (displayHeight - scaledHeight) / 2;
            this.offsetX = 0;
            this.displayScaleX = displayWidth / this.videoWidth;
            this.displayScaleY = scaledHeight / this.videoHeight;
        } else {
            // Letterbox (black bars left/right)
            const scaledWidth = displayHeight * videoAspect;
            this.offsetX = (displayWidth - scaledWidth) / 2;
            this.offsetY = 0;
            this.displayScaleX = scaledWidth / this.videoWidth;
            this.displayScaleY = displayHeight / this.videoHeight;
        }

        // Update canvas size to match video native resolution
        this.canvas.width = this.videoWidth;
        this.canvas.height = this.videoHeight;
    }

    /**
     * Initialize the raster mask bitmap canvas (lazy init)
     */
    initMaskBitmap() {
        if (!this.maskBitmap && this.videoWidth > 0 && this.videoHeight > 0) {
            this.maskBitmap = document.createElement('canvas');
            this.maskBitmap.width = this.videoWidth;
            this.maskBitmap.height = this.videoHeight;
            this.maskCtx = this.maskBitmap.getContext('2d');
            // Start with black (no mask)
            this.maskCtx.fillStyle = '#000000';
            this.maskCtx.fillRect(0, 0, this.videoWidth, this.videoHeight);
            console.log('MaskDrawer: Initialized mask bitmap', this.videoWidth, 'x', this.videoHeight);
        }
    }

    /**
     * Bake a shape onto the mask bitmap (white = masked area)
     */
    bakeShapeToBitmap(shape) {
        this.initMaskBitmap();
        if (!this.maskCtx) return;

        this.maskCtx.fillStyle = '#FFFFFF';
        this.maskCtx.strokeStyle = '#FFFFFF';

        switch (shape.type) {
            case 'rectangle':
                this.maskCtx.fillRect(shape.x, shape.y, shape.width, shape.height);
                break;

            case 'ellipse':
                this.maskCtx.beginPath();
                this.maskCtx.ellipse(shape.cx, shape.cy, shape.rx, shape.ry, 0, 0, 2 * Math.PI);
                this.maskCtx.fill();
                break;

            case 'polygon':
                if (shape.points.length >= 3) {
                    this.maskCtx.beginPath();
                    this.maskCtx.moveTo(shape.points[0].x, shape.points[0].y);
                    for (let i = 1; i < shape.points.length; i++) {
                        this.maskCtx.lineTo(shape.points[i].x, shape.points[i].y);
                    }
                    this.maskCtx.closePath();
                    this.maskCtx.fill();
                }
                break;

            case 'brush':
                if (shape.points.length >= 2) {
                    this.maskCtx.lineWidth = shape.brushSize;
                    this.maskCtx.lineCap = 'round';
                    this.maskCtx.lineJoin = 'round';
                    this.maskCtx.beginPath();
                    this.maskCtx.moveTo(shape.points[0].x, shape.points[0].y);
                    for (let i = 1; i < shape.points.length; i++) {
                        this.maskCtx.lineTo(shape.points[i].x, shape.points[i].y);
                    }
                    this.maskCtx.stroke();
                }
                break;
        }
    }

    /**
     * Rebuild mask bitmap from all shapes (used after undo)
     */
    rebuildMaskBitmap() {
        if (!this.maskBitmap) return;

        // Clear to black
        this.maskCtx.fillStyle = '#000000';
        this.maskCtx.fillRect(0, 0, this.videoWidth, this.videoHeight);

        // Rebake all shapes
        for (const shape of this.shapes) {
            this.bakeShapeToBitmap(shape);
        }
    }

    /**
     * Attach event listeners
     */
    attachListeners() {
        this.canvas.addEventListener('mousedown', this._handleMouseDown);
        this.canvas.addEventListener('mousemove', this._handleMouseMove);
        this.canvas.addEventListener('mouseup', this._handleMouseUp);
        this.canvas.addEventListener('dblclick', this._handleDblClick);
        document.addEventListener('keydown', this._handleKeyDown);

        // Touch support
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            this.handleMouseDown({ clientX: touch.clientX, clientY: touch.clientY, button: 0 });
        });
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            this.handleMouseMove({ clientX: touch.clientX, clientY: touch.clientY });
        });
        this.canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            this.handleMouseUp({});
        });
    }

    /**
     * Detach event listeners
     */
    detachListeners() {
        this.canvas.removeEventListener('mousedown', this._handleMouseDown);
        this.canvas.removeEventListener('mousemove', this._handleMouseMove);
        this.canvas.removeEventListener('mouseup', this._handleMouseUp);
        this.canvas.removeEventListener('dblclick', this._handleDblClick);
        document.removeEventListener('keydown', this._handleKeyDown);
    }

    /**
     * Enable/disable drawing
     */
    enable() {
        this.enabled = true;
        this.canvas.style.pointerEvents = 'auto';
        this.canvas.style.cursor = this.getCursorForTool();
    }

    disable() {
        this.enabled = false;
        this.canvas.style.pointerEvents = 'none';
    }

    /**
     * Get cursor style for current tool
     */
    getCursorForTool() {
        switch (this.options.tool) {
            case 'click': return 'crosshair';
            case 'rectangle': return 'crosshair';
            case 'ellipse': return 'crosshair';
            case 'polygon': return 'crosshair';
            case 'brush': return 'crosshair';
            case 'eraser': return 'crosshair';
            default: return 'default';
        }
    }

    /**
     * Convert screen coordinates to video coordinates
     */
    screenToVideo(clientX, clientY) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (clientX - rect.left - this.offsetX) / this.displayScaleX;
        const y = (clientY - rect.top - this.offsetY) / this.displayScaleY;
        return {
            x: Math.max(0, Math.min(this.videoWidth, x)),
            y: Math.max(0, Math.min(this.videoHeight, y))
        };
    }

    /**
     * Handle mouse down
     */
    handleMouseDown(e) {
        if (!this.enabled) return;

        const point = this.screenToVideo(e.clientX, e.clientY);

        // Click tool - pass through to SAM2Selector
        if (this.options.tool === 'click') {
            if (this.options.sam2Selector) {
                // Let SAM2Selector handle clicks
                return;
            }
            return;
        }

        // Polygon tool - add point on click
        if (this.options.tool === 'polygon') {
            this.polygonPoints.push(point);
            this.redraw();
            return;
        }

        // Eraser tool - paint black on mask bitmap
        if (this.options.tool === 'eraser') {
            this.isDrawing = true;
            this.eraserPath = [point];
            this.initMaskBitmap();
            // Erase at click point immediately (single click support)
            if (this.maskCtx) {
                this.maskCtx.globalCompositeOperation = 'destination-out';
                this.maskCtx.beginPath();
                this.maskCtx.arc(point.x, point.y, this.options.brushSize / 2, 0, Math.PI * 2);
                this.maskCtx.fill();
                this.maskCtx.globalCompositeOperation = 'source-over';
            }
            this.redraw();
            return;
        }

        // Start drawing for other tools
        this.isDrawing = true;
        this.startPoint = point;

        if (this.options.tool === 'brush') {
            this.brushPath = [point];
        }
    }

    /**
     * Handle mouse move
     */
    handleMouseMove(e) {
        if (!this.enabled) return;

        const point = this.screenToVideo(e.clientX, e.clientY);
        this.currentPoint = point;

        if (this.options.tool === 'polygon' && this.polygonPoints.length > 0) {
            // Show preview line to cursor
            this.redraw();
            return;
        }

        if (!this.isDrawing) return;

        if (this.options.tool === 'brush') {
            this.brushPath.push(point);
        }

        // Eraser tool - continuous erasing during drag
        if (this.options.tool === 'eraser') {
            this.eraserPath.push(point);
            if (this.maskCtx && this.eraserPath.length >= 2) {
                const prev = this.eraserPath[this.eraserPath.length - 2];
                this.maskCtx.globalCompositeOperation = 'destination-out';
                this.maskCtx.lineWidth = this.options.brushSize;
                this.maskCtx.lineCap = 'round';
                this.maskCtx.lineJoin = 'round';
                this.maskCtx.beginPath();
                this.maskCtx.moveTo(prev.x, prev.y);
                this.maskCtx.lineTo(point.x, point.y);
                this.maskCtx.stroke();
                this.maskCtx.globalCompositeOperation = 'source-over';
            }
        }

        this.redraw();
    }

    /**
     * Handle mouse up
     */
    handleMouseUp(e) {
        if (!this.enabled || !this.isDrawing) return;

        const tool = this.options.tool;

        // Eraser tool - just finish, erasing already done during mouse move
        if (tool === 'eraser') {
            this.isDrawing = false;
            this.eraserPath = [];
            this.redraw();
            return;
        }

        if (tool === 'rectangle' && this.startPoint && this.currentPoint) {
            const shape = {
                type: 'rectangle',
                x: Math.min(this.startPoint.x, this.currentPoint.x),
                y: Math.min(this.startPoint.y, this.currentPoint.y),
                width: Math.abs(this.currentPoint.x - this.startPoint.x),
                height: Math.abs(this.currentPoint.y - this.startPoint.y)
            };
            if (shape.width > 5 && shape.height > 5) {
                this.completeShape(shape);
            }
        } else if (tool === 'ellipse' && this.startPoint && this.currentPoint) {
            const shape = {
                type: 'ellipse',
                cx: (this.startPoint.x + this.currentPoint.x) / 2,
                cy: (this.startPoint.y + this.currentPoint.y) / 2,
                rx: Math.abs(this.currentPoint.x - this.startPoint.x) / 2,
                ry: Math.abs(this.currentPoint.y - this.startPoint.y) / 2
            };
            if (shape.rx > 5 && shape.ry > 5) {
                this.completeShape(shape);
            }
        } else if (tool === 'brush' && this.brushPath.length > 2) {
            const shape = {
                type: 'brush',
                points: [...this.brushPath],
                brushSize: this.options.brushSize
            };
            this.completeShape(shape);
        }

        this.isDrawing = false;
        this.startPoint = null;
        this.brushPath = [];
        this.redraw();
    }

    /**
     * Handle double click (close polygon)
     */
    handleDblClick(e) {
        if (!this.enabled) return;

        if (this.options.tool === 'polygon' && this.polygonPoints.length >= 3) {
            const shape = {
                type: 'polygon',
                points: [...this.polygonPoints]
            };
            this.completeShape(shape);
            this.polygonPoints = [];
            this.redraw();
        }
    }

    /**
     * Handle keyboard events
     */
    handleKeyDown(e) {
        if (!this.enabled) return;

        // Escape - cancel current drawing
        if (e.key === 'Escape') {
            this.cancelDrawing();
        }

        // Enter - close polygon
        if (e.key === 'Enter' && this.options.tool === 'polygon' && this.polygonPoints.length >= 3) {
            const shape = {
                type: 'polygon',
                points: [...this.polygonPoints]
            };
            this.completeShape(shape);
            this.polygonPoints = [];
            this.redraw();
        }

        // Z - undo last shape
        if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            this.undoLastShape();
        }
    }

    /**
     * Cancel current drawing operation
     */
    cancelDrawing() {
        this.isDrawing = false;
        this.startPoint = null;
        this.currentPoint = null;
        this.brushPath = [];
        this.polygonPoints = [];
        this.redraw();
    }

    /**
     * Complete a shape and trigger callback
     */
    completeShape(shape) {
        // Add bbox to all shapes
        shape.bbox = this.shapeToBBox(shape);

        this.shapes.push(shape);

        // Bake shape to mask bitmap for pixel-level editing support
        this.bakeShapeToBitmap(shape);

        if (this.options.onShapeComplete) {
            this.options.onShapeComplete(shape, this.options.mode);
        }

        console.log('Shape completed:', shape.type, 'Mode:', this.options.mode);
    }

    /**
     * Get bounding box for a shape [x1, y1, x2, y2]
     */
    shapeToBBox(shape) {
        switch (shape.type) {
            case 'rectangle':
                return [shape.x, shape.y, shape.x + shape.width, shape.y + shape.height];
            case 'ellipse':
                return [
                    shape.cx - shape.rx,
                    shape.cy - shape.ry,
                    shape.cx + shape.rx,
                    shape.cy + shape.ry
                ];
            case 'polygon':
            case 'brush':
                const xs = shape.points.map(p => p.x);
                const ys = shape.points.map(p => p.y);
                return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
            default:
                return [0, 0, this.videoWidth, this.videoHeight];
        }
    }

    /**
     * Convert shape to mask (base64 PNG)
     */
    shapeToMask(shape) {
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = this.videoWidth;
        maskCanvas.height = this.videoHeight;
        const maskCtx = maskCanvas.getContext('2d');

        // Black background
        maskCtx.fillStyle = '#000000';
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);

        // White shape (mask area)
        maskCtx.fillStyle = '#FFFFFF';
        maskCtx.strokeStyle = '#FFFFFF';

        switch (shape.type) {
            case 'rectangle':
                maskCtx.fillRect(shape.x, shape.y, shape.width, shape.height);
                break;

            case 'ellipse':
                maskCtx.beginPath();
                maskCtx.ellipse(shape.cx, shape.cy, shape.rx, shape.ry, 0, 0, 2 * Math.PI);
                maskCtx.fill();
                break;

            case 'polygon':
                if (shape.points.length >= 3) {
                    maskCtx.beginPath();
                    maskCtx.moveTo(shape.points[0].x, shape.points[0].y);
                    for (let i = 1; i < shape.points.length; i++) {
                        maskCtx.lineTo(shape.points[i].x, shape.points[i].y);
                    }
                    maskCtx.closePath();
                    maskCtx.fill();
                }
                break;

            case 'brush':
                if (shape.points.length >= 2) {
                    maskCtx.lineWidth = shape.brushSize;
                    maskCtx.lineCap = 'round';
                    maskCtx.lineJoin = 'round';
                    maskCtx.beginPath();
                    maskCtx.moveTo(shape.points[0].x, shape.points[0].y);
                    for (let i = 1; i < shape.points.length; i++) {
                        maskCtx.lineTo(shape.points[i].x, shape.points[i].y);
                    }
                    maskCtx.stroke();
                }
                break;
        }

        return maskCanvas.toDataURL('image/png');
    }

    /**
     * Get combined mask for all shapes (base64 PNG)
     * Uses raster bitmap if available (eraser was used), otherwise generates from shapes
     */
    getAllShapesMask() {
        // If mask bitmap exists (eraser was used or shapes were baked), use it directly
        if (this.maskBitmap) {
            console.log('MaskDrawer: Using bitmap mask (eraser/baked)');
            return this.maskBitmap.toDataURL('image/png');
        }

        // Otherwise generate from shapes (fallback)
        console.log('MaskDrawer: Generating mask from shapes');
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = this.videoWidth;
        maskCanvas.height = this.videoHeight;
        const maskCtx = maskCanvas.getContext('2d');

        // Black background
        maskCtx.fillStyle = '#000000';
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);

        // Draw all shapes in white
        maskCtx.fillStyle = '#FFFFFF';
        maskCtx.strokeStyle = '#FFFFFF';

        for (const shape of this.shapes) {
            switch (shape.type) {
                case 'rectangle':
                    maskCtx.fillRect(shape.x, shape.y, shape.width, shape.height);
                    break;

                case 'ellipse':
                    maskCtx.beginPath();
                    maskCtx.ellipse(shape.cx, shape.cy, shape.rx, shape.ry, 0, 0, 2 * Math.PI);
                    maskCtx.fill();
                    break;

                case 'polygon':
                    if (shape.points.length >= 3) {
                        maskCtx.beginPath();
                        maskCtx.moveTo(shape.points[0].x, shape.points[0].y);
                        for (let i = 1; i < shape.points.length; i++) {
                            maskCtx.lineTo(shape.points[i].x, shape.points[i].y);
                        }
                        maskCtx.closePath();
                        maskCtx.fill();
                    }
                    break;

                case 'brush':
                    if (shape.points.length >= 2) {
                        maskCtx.lineWidth = shape.brushSize;
                        maskCtx.lineCap = 'round';
                        maskCtx.lineJoin = 'round';
                        maskCtx.beginPath();
                        maskCtx.moveTo(shape.points[0].x, shape.points[0].y);
                        for (let i = 1; i < shape.points.length; i++) {
                            maskCtx.lineTo(shape.points[i].x, shape.points[i].y);
                        }
                        maskCtx.stroke();
                    }
                    break;
            }
        }

        return maskCanvas.toDataURL('image/png');
    }

    /**
     * Redraw canvas with all shapes
     */
    redraw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw mask bitmap as green overlay (if exists)
        if (this.maskBitmap) {
            this.drawGreenOverlay();
        }

        // Draw shape outlines only when NOT erasing (green overlay is the truth during erasing)
        if (this.options.tool !== 'eraser') {
            // Draw completed shapes
            for (const shape of this.shapes) {
                this.drawShape(shape, false);
            }
        }

        // Draw current shape being drawn
        if (this.isDrawing && this.startPoint && this.currentPoint) {
            if (this.options.tool === 'rectangle') {
                this.drawShape({
                    type: 'rectangle',
                    x: Math.min(this.startPoint.x, this.currentPoint.x),
                    y: Math.min(this.startPoint.y, this.currentPoint.y),
                    width: Math.abs(this.currentPoint.x - this.startPoint.x),
                    height: Math.abs(this.currentPoint.y - this.startPoint.y)
                }, true);
            } else if (this.options.tool === 'ellipse') {
                this.drawShape({
                    type: 'ellipse',
                    cx: (this.startPoint.x + this.currentPoint.x) / 2,
                    cy: (this.startPoint.y + this.currentPoint.y) / 2,
                    rx: Math.abs(this.currentPoint.x - this.startPoint.x) / 2,
                    ry: Math.abs(this.currentPoint.y - this.startPoint.y) / 2
                }, true);
            } else if (this.options.tool === 'brush' && this.brushPath.length > 0) {
                this.drawShape({
                    type: 'brush',
                    points: this.brushPath,
                    brushSize: this.options.brushSize
                }, true);
            }
        }

        // Draw polygon preview
        if (this.options.tool === 'polygon' && this.polygonPoints.length > 0) {
            this.drawPolygonPreview();
        }
    }

    /**
     * Draw green overlay from mask bitmap
     */
    drawGreenOverlay() {
        if (!this.maskBitmap || !this.maskCtx) return;

        // Get mask bitmap image data
        const maskData = this.maskCtx.getImageData(0, 0, this.videoWidth, this.videoHeight);
        const data = maskData.data;

        // Create green overlay image data
        const overlayData = this.ctx.createImageData(this.videoWidth, this.videoHeight);
        const overlay = overlayData.data;

        // Convert white mask to semi-transparent green
        for (let i = 0; i < data.length; i += 4) {
            // Check if pixel is white (masked area)
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            const a = data[i + 3];

            if (r > 128 || g > 128 || b > 128 || a > 128) {
                // Green with 40% opacity
                overlay[i] = 0;       // R
                overlay[i + 1] = 255; // G
                overlay[i + 2] = 0;   // B
                overlay[i + 3] = 102; // A (40% of 255)
            } else {
                // Transparent
                overlay[i] = 0;
                overlay[i + 1] = 0;
                overlay[i + 2] = 0;
                overlay[i + 3] = 0;
            }
        }

        // Draw the green overlay
        this.ctx.putImageData(overlayData, 0, 0);
    }

    /**
     * Draw a shape on canvas
     */
    drawShape(shape, isPreview = false) {
        const alpha = isPreview ? 0.3 : 0.4;
        const fillColor = `rgba(102, 126, 234, ${alpha})`;
        const strokeColor = isPreview ? '#a78bfa' : '#667eea';

        this.ctx.fillStyle = fillColor;
        this.ctx.strokeStyle = strokeColor;
        this.ctx.lineWidth = this.options.strokeWidth;

        switch (shape.type) {
            case 'rectangle':
                this.ctx.fillRect(shape.x, shape.y, shape.width, shape.height);
                this.ctx.strokeRect(shape.x, shape.y, shape.width, shape.height);
                break;

            case 'ellipse':
                this.ctx.beginPath();
                this.ctx.ellipse(shape.cx, shape.cy, shape.rx, shape.ry, 0, 0, 2 * Math.PI);
                this.ctx.fill();
                this.ctx.stroke();
                break;

            case 'polygon':
                if (shape.points.length >= 3) {
                    this.ctx.beginPath();
                    this.ctx.moveTo(shape.points[0].x, shape.points[0].y);
                    for (let i = 1; i < shape.points.length; i++) {
                        this.ctx.lineTo(shape.points[i].x, shape.points[i].y);
                    }
                    this.ctx.closePath();
                    this.ctx.fill();
                    this.ctx.stroke();
                }
                break;

            case 'brush':
                if (shape.points.length >= 2) {
                    this.ctx.lineWidth = shape.brushSize;
                    this.ctx.lineCap = 'round';
                    this.ctx.lineJoin = 'round';
                    this.ctx.strokeStyle = fillColor;
                    this.ctx.beginPath();
                    this.ctx.moveTo(shape.points[0].x, shape.points[0].y);
                    for (let i = 1; i < shape.points.length; i++) {
                        this.ctx.lineTo(shape.points[i].x, shape.points[i].y);
                    }
                    this.ctx.stroke();

                    // Draw outline
                    this.ctx.lineWidth = 2;
                    this.ctx.strokeStyle = strokeColor;
                    this.ctx.stroke();
                }
                break;
        }
    }

    /**
     * Draw polygon preview while user is adding points
     */
    drawPolygonPreview() {
        if (this.polygonPoints.length === 0) return;

        this.ctx.fillStyle = 'rgba(102, 126, 234, 0.2)';
        this.ctx.strokeStyle = '#a78bfa';
        this.ctx.lineWidth = 2;

        // Draw points
        for (const point of this.polygonPoints) {
            this.ctx.beginPath();
            this.ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
            this.ctx.fillStyle = '#667eea';
            this.ctx.fill();
            this.ctx.strokeStyle = '#ffffff';
            this.ctx.stroke();
        }

        // Draw lines between points
        if (this.polygonPoints.length >= 2) {
            this.ctx.beginPath();
            this.ctx.moveTo(this.polygonPoints[0].x, this.polygonPoints[0].y);
            for (let i = 1; i < this.polygonPoints.length; i++) {
                this.ctx.lineTo(this.polygonPoints[i].x, this.polygonPoints[i].y);
            }
            this.ctx.strokeStyle = '#667eea';
            this.ctx.stroke();
        }

        // Draw preview line to cursor
        if (this.currentPoint && this.polygonPoints.length > 0) {
            this.ctx.beginPath();
            this.ctx.setLineDash([5, 5]);
            this.ctx.moveTo(this.polygonPoints[this.polygonPoints.length - 1].x,
                           this.polygonPoints[this.polygonPoints.length - 1].y);
            this.ctx.lineTo(this.currentPoint.x, this.currentPoint.y);
            this.ctx.strokeStyle = '#a78bfa';
            this.ctx.stroke();
            this.ctx.setLineDash([]);

            // Draw closing line preview
            if (this.polygonPoints.length >= 2) {
                this.ctx.beginPath();
                this.ctx.setLineDash([5, 5]);
                this.ctx.moveTo(this.currentPoint.x, this.currentPoint.y);
                this.ctx.lineTo(this.polygonPoints[0].x, this.polygonPoints[0].y);
                this.ctx.strokeStyle = 'rgba(167, 139, 250, 0.5)';
                this.ctx.stroke();
                this.ctx.setLineDash([]);
            }
        }
    }

    /**
     * Undo last shape
     */
    undoLastShape() {
        if (this.shapes.length > 0) {
            this.shapes.pop();
            // Rebuild the mask bitmap from remaining shapes
            if (this.maskBitmap) {
                this.rebuildMaskBitmap();
            }
            this.redraw();
            console.log('Undo: removed last shape');
        }
    }

    /**
     * Set drawing mode
     */
    setMode(mode) {
        this.options.mode = mode;
        if (this.options.onModeChange) {
            this.options.onModeChange(mode);
        }
        console.log('Mode changed to:', mode);
    }

    /**
     * Set drawing tool
     */
    setTool(tool) {
        // Cancel any current drawing
        this.cancelDrawing();

        this.options.tool = tool;
        this.canvas.style.cursor = this.getCursorForTool();

        if (this.options.onToolChange) {
            this.options.onToolChange(tool);
        }
        console.log('Tool changed to:', tool);
    }

    /**
     * Set brush size
     */
    setBrushSize(size) {
        this.options.brushSize = Math.max(5, Math.min(100, size));
    }

    /**
     * Get current mode
     */
    getMode() {
        return this.options.mode;
    }

    /**
     * Get current tool
     */
    getTool() {
        return this.options.tool;
    }

    /**
     * Get all completed shapes
     */
    getShapes() {
        return this.shapes;
    }

    /**
     * Get shape count
     */
    getShapeCount() {
        return this.shapes.length;
    }

    /**
     * Reset all shapes and state
     */
    reset() {
        this.shapes = [];
        this.polygonPoints = [];
        this.brushPath = [];
        this.eraserPath = [];
        this.isDrawing = false;
        this.startPoint = null;
        this.currentPoint = null;
        // Clear the mask bitmap
        this.maskBitmap = null;
        this.maskCtx = null;
        this.redraw();
        console.log('MaskDrawer reset');
    }

    /**
     * Destroy the drawer and cleanup
     */
    destroy() {
        this.detachListeners();
        this.reset();
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MaskDrawer;
}
