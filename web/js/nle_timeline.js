/**
 * Professional Non-Linear Editor Timeline Component
 * Full-featured timeline like Premiere Pro / DaVinci Resolve
 */

class NLETimeline {
    constructor(videoElement, options = {}) {
        this.video = videoElement;
        this.options = {
            width: 1200,
            height: 400,
            tracksHeight: 60,
            rulerHeight: 40,
            controlsHeight: 50,
            fps: 30,
            pixelsPerSecond: 100, // Initial zoom level
            minPixelsPerSecond: 20,
            maxPixelsPerSecond: 500,
            snapThreshold: 10,
            ...options
        };

        // Timeline state
        this.duration = 0;
        this.currentTime = 0;
        this.pixelsPerSecond = this.options.pixelsPerSecond;
        this.scrollX = 0;
        this.isPlaying = false;
        this.isDraggingPlayhead = false;
        this.isDraggingClip = null;
        this.isResizingClip = null;

        // Tracks and clips
        this.tracks = [
            { id: 'video1', type: 'video', name: 'Video 1', clips: [], visible: true, locked: false },
            { id: 'video2', type: 'video', name: 'Video 2', clips: [], visible: true, locked: false },
            { id: 'audio1', type: 'audio', name: 'Audio 1', clips: [], visible: true, locked: false, muted: false },
            { id: 'effects1', type: 'effects', name: 'Effects', clips: [], visible: true, locked: false }
        ];

        // Markers
        this.markers = [];

        // Canvas elements
        this.canvas = null;
        this.ctx = null;
        this.previewCanvas = null;

        // Container
        this.container = null;
    }

    /**
     * Initialize timeline
     */
    init() {
        this.duration = this.video.duration || 10;
        this.createUI();
        this.attachEventListeners();
        this.render();

        console.log('[NLE Timeline] Initialized');
    }

    /**
     * Create UI structure
     */
    createUI() {
        // Main container
        this.container = document.createElement('div');
        this.container.className = 'nle-timeline-container';
        this.container.style.cssText = `
            width: 100%;
            background: #1a1a1a;
            border-radius: 8px;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        `;

        // Header with controls
        const header = this.createHeader();
        this.container.appendChild(header);

        // Canvas for timeline
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.options.width;
        this.canvas.height = this.options.height;
        this.canvas.style.cssText = `
            display: block;
            cursor: crosshair;
            background: #242424;
        `;
        this.ctx = this.canvas.getContext('2d');
        this.container.appendChild(this.canvas);

        // Zoom controls
        const zoomControls = this.createZoomControls();
        this.container.appendChild(zoomControls);

        return this.container;
    }

    /**
     * Create header with playback controls
     */
    createHeader() {
        const header = document.createElement('div');
        header.style.cssText = `
            background: #1a1a1a;
            padding: 1rem;
            display: flex;
            gap: 1rem;
            align-items: center;
            border-bottom: 1px solid #333;
        `;

        // Play/Pause button
        this.playBtn = document.createElement('button');
        this.playBtn.innerHTML = '▶';
        this.playBtn.style.cssText = `
            background: #667eea;
            color: white;
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.2s;
        `;
        this.playBtn.onclick = () => this.togglePlay();
        header.appendChild(this.playBtn);

        // Step backward
        const stepBackBtn = document.createElement('button');
        stepBackBtn.innerHTML = '⏮';
        stepBackBtn.style.cssText = this.getControlButtonStyle();
        stepBackBtn.onclick = () => this.stepFrame(-1);
        header.appendChild(stepBackBtn);

        // Step forward
        const stepFwdBtn = document.createElement('button');
        stepFwdBtn.innerHTML = '⏭';
        stepFwdBtn.style.cssText = this.getControlButtonStyle();
        stepFwdBtn.onclick = () => this.stepFrame(1);
        header.appendChild(stepFwdBtn);

        // Timecode display
        this.timecodeDisplay = document.createElement('div');
        this.timecodeDisplay.textContent = '00:00:00:00';
        this.timecodeDisplay.style.cssText = `
            font-family: 'Courier New', monospace;
            font-size: 1.25rem;
            color: #fff;
            background: #2a2a2a;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            min-width: 150px;
            text-align: center;
        `;
        header.appendChild(this.timecodeDisplay);

        // Add marker button
        const addMarkerBtn = document.createElement('button');
        addMarkerBtn.innerHTML = '🚩 Add Marker';
        addMarkerBtn.style.cssText = this.getControlButtonStyle();
        addMarkerBtn.onclick = () => this.addMarker();
        header.appendChild(addMarkerBtn);

        return header;
    }

    /**
     * Create zoom controls
     */
    createZoomControls() {
        const controls = document.createElement('div');
        controls.style.cssText = `
            background: #1a1a1a;
            padding: 0.75rem 1rem;
            display: flex;
            gap: 1rem;
            align-items: center;
            border-top: 1px solid #333;
        `;

        const label = document.createElement('span');
        label.textContent = 'Zoom:';
        label.style.color = '#999';
        controls.appendChild(label);

        // Zoom out
        const zoomOutBtn = document.createElement('button');
        zoomOutBtn.innerHTML = '−';
        zoomOutBtn.style.cssText = this.getControlButtonStyle();
        zoomOutBtn.onclick = () => this.zoom(-0.2);
        controls.appendChild(zoomOutBtn);

        // Zoom slider
        this.zoomSlider = document.createElement('input');
        this.zoomSlider.type = 'range';
        this.zoomSlider.min = this.options.minPixelsPerSecond;
        this.zoomSlider.max = this.options.maxPixelsPerSecond;
        this.zoomSlider.value = this.pixelsPerSecond;
        this.zoomSlider.style.cssText = `
            flex: 1;
            max-width: 200px;
        `;
        this.zoomSlider.oninput = (e) => this.setZoom(parseFloat(e.target.value));
        controls.appendChild(this.zoomSlider);

        // Zoom in
        const zoomInBtn = document.createElement('button');
        zoomInBtn.innerHTML = '+';
        zoomInBtn.style.cssText = this.getControlButtonStyle();
        zoomInBtn.onclick = () => this.zoom(0.2);
        controls.appendChild(zoomInBtn);

        // Zoom level display
        this.zoomDisplay = document.createElement('span');
        this.zoomDisplay.textContent = '100%';
        this.zoomDisplay.style.cssText = `
            color: #999;
            font-size: 0.875rem;
            min-width: 50px;
            font-family: 'Courier New', monospace;
        `;
        controls.appendChild(this.zoomDisplay);

        return controls;
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Canvas mouse events
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('wheel', (e) => this.onWheel(e));

        // Video time update
        this.video.addEventListener('timeupdate', () => {
            this.currentTime = this.video.currentTime;
            this.updateTimecode();
            this.render();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space') {
                e.preventDefault();
                this.togglePlay();
            } else if (e.code === 'ArrowLeft') {
                this.stepFrame(-1);
            } else if (e.code === 'ArrowRight') {
                this.stepFrame(1);
            }
        });
    }

    /**
     * Mouse down handler
     */
    onMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Check if clicking on playhead
        const playheadX = this.timeToX(this.currentTime);
        if (Math.abs(x - playheadX) < 10 && y < this.options.rulerHeight + 30) {
            this.isDraggingPlayhead = true;
            return;
        }

        // Check if clicking on ruler (jump to time)
        if (y < this.options.rulerHeight) {
            const time = this.xToTime(x);
            this.seekTo(time);
            this.isDraggingPlayhead = true;
            return;
        }

        // Check if clicking on clip
        // TODO: Implement clip selection and dragging
    }

    /**
     * Mouse move handler
     */
    onMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;

        if (this.isDraggingPlayhead) {
            const time = this.xToTime(x);
            this.seekTo(time);
            this.render();
        }

        // TODO: Show thumbnail preview on hover
    }

    /**
     * Mouse up handler
     */
    onMouseUp(e) {
        this.isDraggingPlayhead = false;
        this.isDraggingClip = null;
        this.isResizingClip = null;
    }

    /**
     * Wheel handler for horizontal scroll
     */
    onWheel(e) {
        e.preventDefault();

        if (e.ctrlKey || e.metaKey) {
            // Zoom
            const delta = e.deltaY > 0 ? -0.1 : 0.1;
            this.zoom(delta);
        } else {
            // Horizontal scroll
            this.scrollX -= e.deltaY;
            this.scrollX = Math.max(0, this.scrollX);
            this.render();
        }
    }

    /**
     * Render timeline
     */
    render() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear canvas
        ctx.fillStyle = '#242424';
        ctx.fillRect(0, 0, width, height);

        // Render time ruler
        this.renderRuler();

        // Render tracks
        this.renderTracks();

        // Render playhead
        this.renderPlayhead();

        // Render markers
        this.renderMarkers();
    }

    /**
     * Render time ruler
     */
    renderRuler() {
        const ctx = this.ctx;
        const rulerHeight = this.options.rulerHeight;
        const width = this.canvas.width;

        // Background
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, width, rulerHeight);

        // Time markings
        ctx.strokeStyle = '#444';
        ctx.fillStyle = '#999';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';

        const interval = this.pixelsPerSecond > 100 ? 1 : 5; // Seconds between marks
        const totalSeconds = Math.ceil(this.duration);

        for (let sec = 0; sec <= totalSeconds; sec += interval) {
            const x = this.timeToX(sec);

            if (x < 0 || x > width) continue;

            // Major tick
            ctx.beginPath();
            ctx.moveTo(x, rulerHeight - 15);
            ctx.lineTo(x, rulerHeight);
            ctx.stroke();

            // Time label
            const minutes = Math.floor(sec / 60);
            const seconds = sec % 60;
            ctx.fillText(`${minutes}:${seconds.toString().padStart(2, '0')}`, x, rulerHeight - 20);

            // Minor ticks (frames)
            if (this.pixelsPerSecond > 50) {
                const framesPerSecond = this.options.fps;
                for (let frame = 1; frame < framesPerSecond; frame++) {
                    const frameTime = sec + (frame / framesPerSecond);
                    const frameX = this.timeToX(frameTime);

                    if (frameX < 0 || frameX > width) continue;

                    ctx.beginPath();
                    ctx.moveTo(frameX, rulerHeight - 8);
                    ctx.lineTo(frameX, rulerHeight);
                    ctx.stroke();
                }
            }
        }

        // Border
        ctx.strokeStyle = '#333';
        ctx.beginPath();
        ctx.moveTo(0, rulerHeight);
        ctx.lineTo(width, rulerHeight);
        ctx.stroke();
    }

    /**
     * Render tracks
     */
    renderTracks() {
        const ctx = this.ctx;
        const trackY = this.options.rulerHeight;
        const trackHeight = this.options.tracksHeight;
        const width = this.canvas.width;

        this.tracks.forEach((track, index) => {
            const y = trackY + (index * trackHeight);

            // Track background
            ctx.fillStyle = index % 2 === 0 ? '#2a2a2a' : '#242424';
            ctx.fillRect(0, y, width, trackHeight);

            // Track label
            ctx.fillStyle = '#999';
            ctx.font = '12px Arial';
            ctx.textAlign = 'left';
            ctx.fillText(track.name, 10, y + 20);

            // Track border
            ctx.strokeStyle = '#333';
            ctx.beginPath();
            ctx.moveTo(0, y + trackHeight);
            ctx.lineTo(width, y + trackHeight);
            ctx.stroke();

            // Render clips on this track
            track.clips.forEach(clip => {
                this.renderClip(clip, y, trackHeight);
            });
        });
    }

    /**
     * Render clip block
     */
    renderClip(clip, trackY, trackHeight) {
        const ctx = this.ctx;
        const x = this.timeToX(clip.startTime);
        const width = (clip.duration * this.pixelsPerSecond);
        const y = trackY + 5;
        const height = trackHeight - 10;

        // Clip colors by type
        const colors = {
            video: '#4CAF50',
            audio: '#2196F3',
            effects: '#FF9800'
        };

        const color = colors[clip.type] || '#999';

        // Clip background
        ctx.fillStyle = color + '40';
        ctx.fillRect(x, y, width, height);

        // Clip border
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        // Clip label
        ctx.fillStyle = '#fff';
        ctx.font = '11px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(clip.name, x + 5, y + 15);

        // Trim handles
        ctx.fillStyle = color;
        ctx.fillRect(x, y, 5, height); // Left handle
        ctx.fillRect(x + width - 5, y, 5, height); // Right handle
    }

    /**
     * Render playhead
     */
    renderPlayhead() {
        const ctx = this.ctx;
        const x = this.timeToX(this.currentTime);
        const height = this.canvas.height;

        // Playhead line
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        // Playhead handle
        ctx.fillStyle = '#667eea';
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x - 8, 12);
        ctx.lineTo(x + 8, 12);
        ctx.closePath();
        ctx.fill();
    }

    /**
     * Render markers
     */
    renderMarkers() {
        const ctx = this.ctx;

        this.markers.forEach(marker => {
            const x = this.timeToX(marker.time);

            // Marker flag
            ctx.fillStyle = '#ff4444';
            ctx.beginPath();
            ctx.moveTo(x, 15);
            ctx.lineTo(x, 5);
            ctx.lineTo(x + 10, 10);
            ctx.closePath();
            ctx.fill();
        });
    }

    /**
     * Convert time to X position
     */
    timeToX(time) {
        return (time * this.pixelsPerSecond) - this.scrollX;
    }

    /**
     * Convert X position to time
     */
    xToTime(x) {
        return Math.max(0, Math.min(this.duration, (x + this.scrollX) / this.pixelsPerSecond));
    }

    /**
     * Zoom timeline
     */
    zoom(delta) {
        const oldPixelsPerSecond = this.pixelsPerSecond;
        this.pixelsPerSecond *= (1 + delta);
        this.pixelsPerSecond = Math.max(this.options.minPixelsPerSecond,
                                        Math.min(this.options.maxPixelsPerSecond, this.pixelsPerSecond));

        // Maintain center point
        const centerTime = this.xToTime(this.canvas.width / 2);
        this.scrollX = (centerTime * this.pixelsPerSecond) - (this.canvas.width / 2);
        this.scrollX = Math.max(0, this.scrollX);

        this.zoomSlider.value = this.pixelsPerSecond;
        this.updateZoomDisplay();
        this.render();
    }

    /**
     * Set zoom level
     */
    setZoom(pixelsPerSecond) {
        this.pixelsPerSecond = pixelsPerSecond;
        this.updateZoomDisplay();
        this.render();
    }

    /**
     * Update zoom display
     */
    updateZoomDisplay() {
        const percent = Math.round((this.pixelsPerSecond / this.options.pixelsPerSecond) * 100);
        this.zoomDisplay.textContent = `${percent}%`;
    }

    /**
     * Toggle play/pause
     */
    togglePlay() {
        if (this.video.paused) {
            this.video.play();
            this.playBtn.innerHTML = '⏸';
        } else {
            this.video.pause();
            this.playBtn.innerHTML = '▶';
        }
    }

    /**
     * Step frame forward/backward
     */
    stepFrame(direction) {
        const frameTime = 1 / this.options.fps;
        const newTime = this.currentTime + (frameTime * direction);
        this.seekTo(newTime);
    }

    /**
     * Seek to time
     */
    seekTo(time) {
        time = Math.max(0, Math.min(this.duration, time));
        this.currentTime = time;
        this.video.currentTime = time;
        this.updateTimecode();
        this.render();
    }

    /**
     * Update timecode display
     */
    updateTimecode() {
        const totalFrames = Math.floor(this.currentTime * this.options.fps);
        const frames = totalFrames % this.options.fps;
        const totalSeconds = Math.floor(this.currentTime);
        const seconds = totalSeconds % 60;
        const minutes = Math.floor(totalSeconds / 60) % 60;
        const hours = Math.floor(totalSeconds / 3600);

        this.timecodeDisplay.textContent =
            `${hours.toString().padStart(2, '0')}:` +
            `${minutes.toString().padStart(2, '0')}:` +
            `${seconds.toString().padStart(2, '0')}:` +
            `${frames.toString().padStart(2, '0')}`;
    }

    /**
     * Add marker at current time
     */
    addMarker() {
        const label = prompt('Marker label:', 'Marker ' + (this.markers.length + 1));
        if (label) {
            this.markers.push({
                time: this.currentTime,
                label: label
            });
            this.render();
        }
    }

    /**
     * Add clip to track
     */
    addClip(trackId, clip) {
        const track = this.tracks.find(t => t.id === trackId);
        if (track) {
            track.clips.push(clip);
            this.render();
        }
    }

    /**
     * Get control button style
     */
    getControlButtonStyle() {
        return `
            background: #333;
            color: #fff;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
        `;
    }

    /**
     * Get timeline container
     */
    getContainer() {
        return this.container;
    }
}

// Export
window.NLETimeline = NLETimeline;
