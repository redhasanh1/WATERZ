/**
 * Professional Video Timeline Component
 * - Frame thumbnails
 * - Timecode display
 * - Click-to-jump navigation
 * - Draggable scrubber
 * - Mask status indicators
 */

class VideoTimeline {
    constructor(videoElement, options = {}) {
        this.video = videoElement;
        this.options = {
            thumbnailHeight: 68,
            thumbnailWidth: 120,
            onFrameChange: null,
            ...options
        };

        this.frames = [];
        this.totalFrames = 0;
        this.fps = 30;
        this.duration = 0;
        this.currentFrame = 0;
        this.maskedFrames = new Set(); // Track frames with masks

        this.container = null;
        this.thumbnailsContainer = null;
        this.playhead = null;
        this.timecodeDisplay = null;

        this.isDragging = false;
        this.isInitialized = false;
    }

    /**
     * Initialize timeline after video metadata is loaded
     */
    async init(taskId) {
        if (this.isInitialized) return;

        try {
            // Fetch frame data from server
            const response = await fetch(`/api/extract-frames/${taskId}`);
            const data = await response.json();

            if (data.status === 'success') {
                this.frames = data.frames;
                this.totalFrames = data.total_frames;
                this.fps = data.fps;
                this.duration = data.duration;

                this.render();
                this.attachEventListeners();
                this.isInitialized = true;

                console.log(`[Timeline] Initialized with ${this.frames.length} thumbnails, ${this.totalFrames} total frames @ ${this.fps.toFixed(2)}fps`);
            }
        } catch (error) {
            console.error('[Timeline] Failed to load frames:', error);
        }
    }

    /**
     * Render timeline UI
     */
    render() {
        // Create main container
        this.container = document.createElement('div');
        this.container.className = 'video-timeline';
        this.container.style.cssText = `
            position: relative;
            width: 100%;
            background: rgba(0, 0, 0, 0.8);
            padding: 0.75rem 0.5rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            overflow-x: auto;
            overflow-y: hidden;
        `;

        // Create thumbnails container
        this.thumbnailsContainer = document.createElement('div');
        this.thumbnailsContainer.className = 'timeline-thumbnails';
        this.thumbnailsContainer.style.cssText = `
            display: flex;
            gap: 4px;
            position: relative;
            height: ${this.options.thumbnailHeight}px;
        `;

        // Render thumbnails
        this.frames.forEach((frame, index) => {
            const thumbWrapper = document.createElement('div');
            thumbWrapper.className = 'timeline-thumb-wrapper';
            thumbWrapper.dataset.frameNumber = frame.frame_number;
            thumbWrapper.dataset.timestamp = frame.timestamp;
            thumbWrapper.style.cssText = `
                position: relative;
                flex-shrink: 0;
                cursor: pointer;
                border: 2px solid transparent;
                border-radius: 4px;
                transition: all 0.2s ease;
            `;

            // Thumbnail image
            const thumb = document.createElement('img');
            thumb.src = frame.thumbnail_url;
            thumb.alt = `Frame ${frame.frame_number}`;
            thumb.style.cssText = `
                display: block;
                width: ${this.options.thumbnailWidth}px;
                height: ${this.options.thumbnailHeight}px;
                object-fit: cover;
                border-radius: 2px;
            `;

            // Timecode overlay
            const timecode = document.createElement('div');
            timecode.textContent = this.formatTimecode(frame.frame_number);
            timecode.style.cssText = `
                position: absolute;
                bottom: 2px;
                right: 2px;
                background: rgba(0, 0, 0, 0.75);
                color: white;
                font-size: 10px;
                font-family: monospace;
                padding: 2px 4px;
                border-radius: 2px;
                pointer-events: none;
            `;

            // Mask indicator (hidden by default)
            const maskIndicator = document.createElement('div');
            maskIndicator.className = 'mask-indicator';
            maskIndicator.style.cssText = `
                position: absolute;
                top: -4px;
                right: -4px;
                width: 12px;
                height: 12px;
                background: #10b981;
                border: 2px solid white;
                border-radius: 50%;
                display: none;
                pointer-events: none;
            `;

            thumbWrapper.appendChild(thumb);
            thumbWrapper.appendChild(timecode);
            thumbWrapper.appendChild(maskIndicator);

            // Click to jump
            thumbWrapper.addEventListener('click', () => {
                this.jumpToFrame(frame.frame_number);
            });

            // Hover effect
            thumbWrapper.addEventListener('mouseenter', () => {
                thumbWrapper.style.borderColor = '#667eea';
                thumbWrapper.style.transform = 'scale(1.05)';
            });

            thumbWrapper.addEventListener('mouseleave', () => {
                if (this.currentFrame !== frame.frame_number) {
                    thumbWrapper.style.borderColor = 'transparent';
                    thumbWrapper.style.transform = 'scale(1)';
                }
            });

            this.thumbnailsContainer.appendChild(thumbWrapper);
        });

        // Create playhead (current position indicator)
        this.playhead = document.createElement('div');
        this.playhead.className = 'timeline-playhead';
        this.playhead.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 3px;
            height: 100%;
            background: #667eea;
            box-shadow: 0 0 8px rgba(102, 126, 234, 0.6);
            pointer-events: none;
            z-index: 10;
        `;

        this.thumbnailsContainer.appendChild(this.playhead);
        this.container.appendChild(this.thumbnailsContainer);

        // Create timecode display
        const timecodeWrapper = document.createElement('div');
        timecodeWrapper.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 0.5rem;
            padding: 0 0.5rem;
            color: white;
            font-size: 12px;
            font-family: monospace;
        `;

        this.timecodeDisplay = document.createElement('div');
        this.timecodeDisplay.textContent = this.formatTimecode(0);

        const durationDisplay = document.createElement('div');
        durationDisplay.textContent = this.formatTimecode(this.totalFrames - 1);

        timecodeWrapper.appendChild(this.timecodeDisplay);
        timecodeWrapper.appendChild(durationDisplay);
        this.container.appendChild(timecodeWrapper);

        return this.container;
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Update playhead position on video timeupdate
        this.video.addEventListener('timeupdate', () => {
            this.updatePlayhead();
        });

        // Enable dragging on thumbnails container
        this.thumbnailsContainer.addEventListener('mousedown', (e) => {
            if (e.target === this.thumbnailsContainer || e.target.closest('.timeline-thumb-wrapper')) {
                this.isDragging = true;
                this.handleDrag(e);
            }
        });

        document.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                this.handleDrag(e);
            }
        });

        document.addEventListener('mouseup', () => {
            this.isDragging = false;
        });
    }

    /**
     * Handle drag to scrub through video
     */
    handleDrag(e) {
        const rect = this.thumbnailsContainer.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percentage = Math.max(0, Math.min(1, x / rect.width));
        const frameNumber = Math.floor(percentage * this.totalFrames);

        this.jumpToFrame(frameNumber);
    }

    /**
     * Update playhead position based on current video time
     */
    updatePlayhead() {
        if (!this.isInitialized) return;

        this.currentFrame = Math.floor(this.video.currentTime * this.fps);

        const percentage = this.currentFrame / this.totalFrames;
        const containerWidth = this.thumbnailsContainer.scrollWidth;

        this.playhead.style.left = `${percentage * containerWidth}px`;
        this.timecodeDisplay.textContent = this.formatTimecode(this.currentFrame);

        // Highlight current thumbnail
        this.highlightCurrentThumbnail();

        // Auto-scroll to keep playhead visible
        this.scrollToPlayhead();
    }

    /**
     * Highlight the current thumbnail
     */
    highlightCurrentThumbnail() {
        const thumbs = this.thumbnailsContainer.querySelectorAll('.timeline-thumb-wrapper');
        thumbs.forEach(thumb => {
            thumb.style.borderColor = 'transparent';
            thumb.style.transform = 'scale(1)';
        });

        // Find closest thumbnail to current frame
        let closestThumb = null;
        let minDiff = Infinity;

        thumbs.forEach(thumb => {
            const frameNum = parseInt(thumb.dataset.frameNumber);
            const diff = Math.abs(frameNum - this.currentFrame);
            if (diff < minDiff) {
                minDiff = diff;
                closestThumb = thumb;
            }
        });

        if (closestThumb) {
            closestThumb.style.borderColor = '#667eea';
            closestThumb.style.transform = 'scale(1.05)';
        }
    }

    /**
     * Scroll timeline to keep playhead visible
     */
    scrollToPlayhead() {
        const playheadLeft = parseFloat(this.playhead.style.left);
        const containerScrollLeft = this.container.scrollLeft;
        const containerWidth = this.container.clientWidth;

        // Scroll if playhead is near edges
        if (playheadLeft < containerScrollLeft + 100) {
            this.container.scrollLeft = Math.max(0, playheadLeft - 100);
        } else if (playheadLeft > containerScrollLeft + containerWidth - 100) {
            this.container.scrollLeft = playheadLeft - containerWidth + 100;
        }
    }

    /**
     * Jump to specific frame
     */
    jumpToFrame(frameNumber) {
        if (!this.isInitialized) return;

        frameNumber = Math.max(0, Math.min(this.totalFrames - 1, frameNumber));
        this.currentFrame = frameNumber;

        const time = frameNumber / this.fps;
        this.video.currentTime = time;

        this.updatePlayhead();

        if (this.options.onFrameChange) {
            this.options.onFrameChange(frameNumber);
        }
    }

    /**
     * Mark frame as having a mask
     */
    markFrameMasked(frameNumber) {
        this.maskedFrames.add(frameNumber);

        // Find thumbnail and show indicator
        const thumbs = this.thumbnailsContainer.querySelectorAll('.timeline-thumb-wrapper');
        thumbs.forEach(thumb => {
            const frameNum = parseInt(thumb.dataset.frameNumber);
            if (Math.abs(frameNum - frameNumber) < this.fps) { // Within 1 second
                const indicator = thumb.querySelector('.mask-indicator');
                if (indicator) {
                    indicator.style.display = 'block';
                }
            }
        });
    }

    /**
     * Clear mask indicators
     */
    clearMaskIndicators() {
        this.maskedFrames.clear();
        const indicators = this.thumbnailsContainer.querySelectorAll('.mask-indicator');
        indicators.forEach(indicator => {
            indicator.style.display = 'none';
        });
    }

    /**
     * Format frame number to timecode (MM:SS:FF)
     */
    formatTimecode(frameNumber) {
        const totalSeconds = frameNumber / this.fps;
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = Math.floor(totalSeconds % 60);
        const frames = frameNumber % Math.floor(this.fps);

        return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}:${frames.toString().padStart(2, '0')}`;
    }

    /**
     * Get timeline container element
     */
    getContainer() {
        return this.container;
    }

    /**
     * Destroy timeline and clean up
     */
    destroy() {
        if (this.container && this.container.parentNode) {
            this.container.parentNode.removeChild(this.container);
        }
        this.isInitialized = false;
    }
}

// Export for use in other scripts
window.VideoTimeline = VideoTimeline;
