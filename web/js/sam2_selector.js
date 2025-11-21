/**
 * SAM2 Interactive Object Selector
 * Fast click-to-select objects with instant mask preview
 * Ported from SELECT_OBJECT_FAST.py for web integration
 */

class SAM2Selector {
    constructor(videoElement, canvasElement, options = {}) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');

        this.options = {
            onMaskUpdate: null,
            onPointsChange: null,
            maskColor: [0, 255, 0], // Green overlay
            maskAlpha: 0.5,
            ...options
        };

        // Multi-object selection state
        this.selections = []; // Array of {id, points, mask}
        this.currentSelectionId = 0;
        this.isLoading = false;

        // All objects use green color
        this.maskColor = [0, 255, 0]; // Green for all selections

        // Video metadata
        this.videoWidth = 0;
        this.videoHeight = 0;
        this.currentFrameIndex = 0;

        // Display state
        this.displayScale = 1.0;
        this.offsetX = 0;
        this.offsetY = 0;

        this.init();
    }

    /**
     * Initialize canvas and event listeners
     */
    init() {
        // Set canvas size to match container
        this.updateCanvasSize();

        // Attach mouse event listeners
        this.canvas.addEventListener('click', (e) => this.handleClick(e));
        this.canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.handleRightClick(e);
        });

        // Update on window resize
        window.addEventListener('resize', () => this.updateCanvasSize());

        console.log('[SAM2Selector] Initialized');
    }

    /**
     * Calculate actual video render bounds accounting for object-fit: contain
     */
    calculateVideoRenderBounds() {
        if (!this.video.videoWidth) return;

        const canvasRect = this.canvas.getBoundingClientRect();
        const containerWidth = canvasRect.width;
        const containerHeight = canvasRect.height;

        // Calculate video's aspect ratio
        const videoAspect = this.video.videoWidth / this.video.videoHeight;
        const containerAspect = containerWidth / containerHeight;

        let renderWidth, renderHeight, offsetX, offsetY;

        if (videoAspect > containerAspect) {
            // Video is wider - letterbox top/bottom
            renderWidth = containerWidth;
            renderHeight = containerWidth / videoAspect;
            offsetX = 0;
            offsetY = (containerHeight - renderHeight) / 2;
        } else {
            // Video is taller - pillarbox left/right
            renderHeight = containerHeight;
            renderWidth = containerHeight * videoAspect;
            offsetX = (containerWidth - renderWidth) / 2;
            offsetY = 0;
        }

        // Store render bounds for coordinate conversion
        this.renderWidth = renderWidth;
        this.renderHeight = renderHeight;
        this.offsetX = offsetX;
        this.offsetY = offsetY;

        console.log(`[SAM2Selector] Video render: ${renderWidth.toFixed(1)}x${renderHeight.toFixed(1)}, Offset: (${offsetX.toFixed(1)}, ${offsetY.toFixed(1)})`);
    }

    /**
     * Update canvas size to match video element
     */
    updateCanvasSize() {
        if (!this.video.videoWidth) return;

        this.videoWidth = this.video.videoWidth;
        this.videoHeight = this.video.videoHeight;

        // Set canvas internal resolution to native video resolution
        this.canvas.width = this.videoWidth;
        this.canvas.height = this.videoHeight;

        // Canvas display size is handled by CSS (100% width/height of parent)
        // No need to set style.width/height - it auto-resizes responsively

        // Get current display dimensions for scale calculation
        const canvasRect = this.canvas.getBoundingClientRect();
        const displayWidth = canvasRect.width;
        const displayHeight = canvasRect.height;

        // Calculate scale for coordinate conversion
        this.displayScaleX = displayWidth / this.videoWidth;
        this.displayScaleY = displayHeight / this.videoHeight;

        console.log(`[SAM2Selector] Canvas: ${this.canvas.width}x${this.canvas.height}, Display: ${displayWidth}x${displayHeight}`);

        // Calculate video render bounds for letterbox compensation
        this.calculateVideoRenderBounds();

        // Redraw with current mask
        this.draw();
    }

    /**
     * Handle left click - add positive point
     */
    handleClick(e) {
        if (this.isLoading) return;

        const point = this.getCanvasPoint(e);
        if (!point) return; // Ignore clicks in letterbox area
        this.addPoint(point.x, point.y, 1);
    }

    /**
     * Handle right click - add negative point
     */
    handleRightClick(e) {
        if (this.isLoading) return;

        const point = this.getCanvasPoint(e);
        if (!point) return; // Ignore clicks in letterbox area
        this.addPoint(point.x, point.y, 0);
    }

    /**
     * Convert mouse event to canvas coordinates
     */
    getCanvasPoint(e) {
        const rect = this.canvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;

        // Subtract letterbox offset to get position relative to rendered video
        const relativeX = clickX - this.offsetX;
        const relativeY = clickY - this.offsetY;

        // Check if click is outside the rendered video area (in letterbox)
        if (relativeX < 0 || relativeY < 0 || relativeX > this.renderWidth || relativeY > this.renderHeight) {
            console.log('[SAM2Selector] Click outside video area (in letterbox), ignoring');
            return null;
        }

        // Convert to canvas coordinates using actual rendered video dimensions
        const videoX = Math.floor((relativeX / this.renderWidth) * this.canvas.width);
        const videoY = Math.floor((relativeY / this.renderHeight) * this.canvas.height);

        // Clamp to video bounds
        return {
            x: Math.max(0, Math.min(videoX, this.videoWidth - 1)),
            y: Math.max(0, Math.min(videoY, this.videoHeight - 1)),
            displayX: clickX,
            displayY: clickY
        };
    }

    /**
     * Add point and create new selection
     */
    async addPoint(x, y, label) {
        // Create new selection for this click
        const selectionId = this.currentSelectionId++;
        const points = [{x, y, label}];

        console.log(`[SAM2Selector] Creating selection #${selectionId} at (${x}, ${y})`);

        // Request mask from server for this single point
        const mask = await this.requestMask(points);

        if (mask) {
            // Add to selections array
            this.selections.push({
                id: selectionId,
                points: points,
                mask: mask
            });

            console.log(`[SAM2Selector] Added selection #${selectionId}, total: ${this.selections.length}`);

            // Callback for selection changes
            if (this.options.onPointsChange) {
                this.options.onPointsChange(this.getAllPoints());
            }
        }

        // Redraw all masks
        this.draw();
    }

    /**
     * Request mask from server for given points
     */
    async requestMask(points) {
        if (points.length === 0) {
            return null;
        }

        this.isLoading = true;

        try {
            // Get current frame as base64
            const frameData = this.captureCurrentFrame();

            // Send to server
            const response = await fetch('/api/sam2/select-object', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    frame_data: frameData,
                    frame_index: this.currentFrameIndex,
                    points: points,
                    video_width: this.videoWidth,
                    video_height: this.videoHeight
                })
            });

            const data = await response.json();

            if (data.status === 'success' && data.mask) {
                // Decode mask from base64
                const mask = await this.decodeMask(data.mask);
                console.log(`[SAM2Selector] Mask received (${mask.width}x${mask.height})`);
                return mask;
            } else {
                console.error('[SAM2Selector] Failed to get mask:', data.message);
                return null;
            }
        } catch (error) {
            console.error('[SAM2Selector] Mask request failed:', error);
            return null;
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Capture current video frame as base64
     */
    captureCurrentFrame() {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.videoWidth;
        tempCanvas.height = this.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');

        tempCtx.drawImage(this.video, 0, 0, this.videoWidth, this.videoHeight);

        // Use PNG for lossless quality (no JPEG compression artifacts)
        return tempCanvas.toDataURL('image/png').split(',')[1];
    }

    /**
     * Decode mask from base64 PNG
     */
    async decodeMask(base64Mask) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = img.width;
                tempCanvas.height = img.height;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(img, 0, 0);

                const imageData = tempCtx.getImageData(0, 0, img.width, img.height);
                resolve(imageData);
            };
            img.onerror = reject;
            img.src = 'data:image/png;base64,' + base64Mask;
        });
    }

    /**
     * Draw current state (video frame + mask overlay + points)
     */
    draw() {
        if (!this.videoWidth) return;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw video frame (scaled)
        this.ctx.drawImage(
            this.video,
            0, 0, this.videoWidth, this.videoHeight,
            0, 0, this.canvas.width, this.canvas.height
        );

        // Draw all mask overlays
        this.drawAllMasks();

        // Draw all points
        this.drawAllPoints();

        // Draw loading indicator
        if (this.isLoading) {
            this.drawLoadingIndicator();
        }
    }

    /**
     * Draw all mask overlays with transparency
     */
    drawAllMasks() {
        if (this.selections.length === 0) return;

        // Create temporary canvas for the mask overlay
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.canvas.width;
        tempCanvas.height = this.canvas.height;
        const tempCtx = tempCanvas.getContext('2d');

        // Create ImageData to build mask
        const maskData = tempCtx.createImageData(this.canvas.width, this.canvas.height);

        // Loop through all selections and composite their masks
        this.selections.forEach(selection => {
            const mask = selection.mask;
            if (!mask) return;

            // Resize mask data to match canvas
            for (let y = 0; y < this.canvas.height; y++) {
                for (let x = 0; x < this.canvas.width; x++) {
                    // Map to mask coordinates
                    const maskX = Math.floor(x / this.canvas.width * mask.width);
                    const maskY = Math.floor(y / this.canvas.height * mask.height);
                    const maskIdx = (maskY * mask.width + maskX) * 4;

                    // If mask pixel is white (255), apply green overlay
                    if (mask.data[maskIdx] > 127) {
                        const idx = (y * this.canvas.width + x) * 4;
                        maskData.data[idx] = this.maskColor[0];     // R (Green)
                        maskData.data[idx + 1] = this.maskColor[1]; // G (Green)
                        maskData.data[idx + 2] = this.maskColor[2]; // B (Green)
                        maskData.data[idx + 3] = 255; // Full alpha on temp canvas
                    }
                }
            }
        });

        // Put mask data on temp canvas
        tempCtx.putImageData(maskData, 0, 0);

        // Draw overlay with transparency using globalAlpha (proper compositing)
        this.ctx.save();
        this.ctx.globalAlpha = this.options.maskAlpha;
        this.ctx.drawImage(tempCanvas, 0, 0);
        this.ctx.restore();
    }

    /**
     * Draw all selection points
     */
    drawAllPoints() {
        this.selections.forEach(selection => {
            selection.points.forEach(point => {
                // Convert from canvas coordinates to rendered video coordinates
                const relativeX = (point.x / this.canvas.width) * this.renderWidth;
                const relativeY = (point.y / this.canvas.height) * this.renderHeight;

                // Add letterbox offset to get display coordinates
                const x = relativeX + this.offsetX;
                const y = relativeY + this.offsetY;
                const color = point.label ? '#00FF00' : '#FF0000'; // Green=positive, Red=negative

                // Draw circle
                this.ctx.beginPath();
                this.ctx.arc(x, y, 6, 0, Math.PI * 2);
                this.ctx.fillStyle = color;
                this.ctx.fill();
                this.ctx.strokeStyle = '#FFFFFF';
                this.ctx.lineWidth = 2;
                this.ctx.stroke();

                // Draw crosshair
                this.ctx.strokeStyle = color;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.moveTo(x - 12, y);
                this.ctx.lineTo(x + 12, y);
                this.ctx.moveTo(x, y - 12);
                this.ctx.lineTo(x, y + 12);
                this.ctx.stroke();
            });
        });
    }

    /**
     * Draw loading indicator
     */
    drawLoadingIndicator() {
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.font = '16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('Generating mask...', this.canvas.width / 2, this.canvas.height / 2);
    }

    /**
     * Reset all selections
     */
    reset() {
        this.selections = [];
        this.currentSelectionId = 0;
        this.draw();

        console.log('[SAM2Selector] Reset all selections');

        if (this.options.onPointsChange) {
            this.options.onPointsChange([]);
        }
    }

    /**
     * Get all points from all selections
     */
    getAllPoints() {
        const allPoints = [];
        this.selections.forEach(selection => {
            allPoints.push(...selection.points);
        });
        return allPoints;
    }

    /**
     * Update when video time changes
     */
    onVideoTimeUpdate() {
        const newFrameIndex = Math.floor(this.video.currentTime * (this.video.duration > 0 ? 30 : 30));

        if (newFrameIndex !== this.currentFrameIndex) {
            this.currentFrameIndex = newFrameIndex;
            this.draw();
        }
    }

    /**
     * Get current selection data for processing
     */
    getSelectionData() {
        return {
            points: this.getAllPoints(),
            selections: this.selections,
            frame_index: this.currentFrameIndex,
            video_width: this.videoWidth,
            video_height: this.videoHeight,
            has_mask: this.selections.length > 0
        };
    }
}

// Export for use in other scripts
window.SAM2Selector = SAM2Selector;
