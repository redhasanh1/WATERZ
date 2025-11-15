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

        // Selection state
        this.points = []; // Array of {x, y, label} where label: 1=positive, 0=negative
        this.currentMask = null;
        this.isLoading = false;

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
     * Update canvas size to match video element
     */
    updateCanvasSize() {
        if (!this.video.videoWidth) return;

        this.videoWidth = this.video.videoWidth;
        this.videoHeight = this.video.videoHeight;

        // Calculate scale to fit canvas while maintaining aspect ratio
        const containerWidth = this.canvas.parentElement.clientWidth;
        const containerHeight = this.canvas.parentElement.clientHeight || 600;

        const scaleX = containerWidth / this.videoWidth;
        const scaleY = containerHeight / this.videoHeight;
        this.displayScale = Math.min(scaleX, scaleY, 1.0);

        // Set canvas display size
        this.canvas.width = Math.floor(this.videoWidth * this.displayScale);
        this.canvas.height = Math.floor(this.videoHeight * this.displayScale);

        // Calculate offsets for centering
        this.offsetX = (containerWidth - this.canvas.width) / 2;
        this.offsetY = (containerHeight - this.canvas.height) / 2;

        console.log(`[SAM2Selector] Canvas: ${this.canvas.width}x${this.canvas.height}, Scale: ${this.displayScale.toFixed(2)}x`);

        // Redraw with current mask
        this.draw();
    }

    /**
     * Handle left click - add positive point
     */
    handleClick(e) {
        if (this.isLoading) return;

        const point = this.getCanvasPoint(e);
        this.addPoint(point.x, point.y, 1);
    }

    /**
     * Handle right click - add negative point
     */
    handleRightClick(e) {
        if (this.isLoading) return;

        const point = this.getCanvasPoint(e);
        this.addPoint(point.x, point.y, 0);
    }

    /**
     * Convert mouse event to canvas coordinates
     */
    getCanvasPoint(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Convert to original video coordinates
        const videoX = Math.floor(x / this.displayScale);
        const videoY = Math.floor(y / this.displayScale);

        // Clamp to video bounds
        return {
            x: Math.max(0, Math.min(videoX, this.videoWidth - 1)),
            y: Math.max(0, Math.min(videoY, this.videoHeight - 1)),
            displayX: x,
            displayY: y
        };
    }

    /**
     * Add point and request mask update
     */
    async addPoint(x, y, label) {
        this.points.push({x, y, label});

        console.log(`[SAM2Selector] Added ${label ? 'positive' : 'negative'} point at (${x}, ${y})`);

        // Callback for point changes
        if (this.options.onPointsChange) {
            this.options.onPointsChange(this.points);
        }

        // Request mask from server
        await this.updateMask();

        // Redraw
        this.draw();
    }

    /**
     * Request mask update from server
     */
    async updateMask() {
        if (this.points.length === 0) {
            this.currentMask = null;
            return;
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
                    points: this.points,
                    video_width: this.videoWidth,
                    video_height: this.videoHeight
                })
            });

            const data = await response.json();

            if (data.status === 'success' && data.mask) {
                // Decode mask from base64
                this.currentMask = await this.decodeMask(data.mask);

                console.log(`[SAM2Selector] Mask updated (${this.currentMask.width}x${this.currentMask.height})`);

                // Callback for mask updates
                if (this.options.onMaskUpdate) {
                    this.options.onMaskUpdate(this.currentMask, this.points);
                }
            } else {
                console.error('[SAM2Selector] Failed to get mask:', data.message);
            }
        } catch (error) {
            console.error('[SAM2Selector] Mask update failed:', error);
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

        return tempCanvas.toDataURL('image/jpeg', 0.95).split(',')[1];
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

        // Draw mask overlay
        if (this.currentMask) {
            this.drawMaskOverlay();
        }

        // Draw points
        this.drawPoints();

        // Draw loading indicator
        if (this.isLoading) {
            this.drawLoadingIndicator();
        }
    }

    /**
     * Draw mask overlay with transparency
     */
    drawMaskOverlay() {
        // Create temporary canvas for mask at display size
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.canvas.width;
        tempCanvas.height = this.canvas.height;
        const tempCtx = tempCanvas.getContext('2d');

        // Scale mask to display size
        tempCtx.putImageData(this.currentMask, 0, 0);
        const scaledMask = tempCtx.getImageData(0, 0, this.currentMask.width, this.currentMask.height);

        // Create overlay with green color
        const overlay = this.ctx.createImageData(this.canvas.width, this.canvas.height);

        // Resize mask data to match canvas
        for (let y = 0; y < this.canvas.height; y++) {
            for (let x = 0; x < this.canvas.width; x++) {
                // Map to mask coordinates
                const maskX = Math.floor(x / this.canvas.width * scaledMask.width);
                const maskY = Math.floor(y / this.canvas.height * scaledMask.height);
                const maskIdx = (maskY * scaledMask.width + maskX) * 4;

                // If mask pixel is white (255), apply green overlay
                if (scaledMask.data[maskIdx] > 127) {
                    const idx = (y * this.canvas.width + x) * 4;
                    overlay.data[idx] = this.options.maskColor[0];     // R
                    overlay.data[idx + 1] = this.options.maskColor[1]; // G
                    overlay.data[idx + 2] = this.options.maskColor[2]; // B
                    overlay.data[idx + 3] = 255 * this.options.maskAlpha; // A
                }
            }
        }

        // Draw overlay
        this.ctx.putImageData(overlay, 0, 0);
    }

    /**
     * Draw selection points
     */
    drawPoints() {
        this.points.forEach(point => {
            const x = point.x * this.displayScale;
            const y = point.y * this.displayScale;
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
     * Reset selection
     */
    reset() {
        this.points = [];
        this.currentMask = null;
        this.draw();

        console.log('[SAM2Selector] Reset');

        if (this.options.onPointsChange) {
            this.options.onPointsChange(this.points);
        }
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
            points: this.points,
            frame_index: this.currentFrameIndex,
            video_width: this.videoWidth,
            video_height: this.videoHeight,
            has_mask: this.currentMask !== null
        };
    }
}

// Export for use in other scripts
window.SAM2Selector = SAM2Selector;
