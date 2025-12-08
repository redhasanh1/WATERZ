/**
 * VideoTimeline - Professional timeline with frame thumbnails
 * For watermarkz video editing application
 */
class VideoTimeline {
    constructor(videoElement, options = {}) {
        this.video = videoElement;
        this.options = {
            thumbnailWidth: 80,
            thumbnailHeight: 60,
            maxThumbnails: 50,
            thumbnailInterval: null,
            onSeek: null,
            onKeyframeAdd: null,
            onKeyframeRemove: null,
            ...options
        };

        this.thumbnails = [];
        this.keyframes = [];
        this.isGenerating = false;
        this.isDragging = false;
        this.zoomLevel = 1;
        this.keyframeIdCounter = 0;

        this.container = null;
        this.thumbnailTrack = null;
        this.thumbnailContainer = null;
        this.playhead = null;
        this.playheadTime = null;
        this.keyframeMarkersContainer = null;
        this.timeScale = null;

        this.extractCanvas = document.createElement('canvas');
        this.extractCtx = this.extractCanvas.getContext('2d');

        this.init();
    }

    init() {
        this.container = document.getElementById('videoTimelineContainer');
        this.thumbnailTrack = document.getElementById('thumbnailTrack');
        this.thumbnailContainer = document.getElementById('thumbnailContainer');
        this.playhead = document.getElementById('timelinePlayhead');
        this.playheadTime = document.getElementById('playheadTime');
        this.keyframeMarkersContainer = document.getElementById('keyframeMarkersContainer');
        this.timeScale = document.getElementById('timeScale');

        if (!this.container) {
            console.error('[VideoTimeline] Container not found');
            return;
        }

        this.bindEvents();
        console.log('[VideoTimeline] Initialized');
    }

    bindEvents() {
        this.video.addEventListener('timeupdate', () => this.updatePlayhead());
        this.video.addEventListener('loadedmetadata', () => this.onVideoLoaded());

        if (this.thumbnailTrack) {
            this.thumbnailTrack.addEventListener('click', (e) => this.handleTrackClick(e));
        }

        if (this.playhead) {
            this.playhead.addEventListener('mousedown', (e) => this.startDragging(e));
            this.playhead.addEventListener('touchstart', (e) => this.startDragging(e));
        }

        document.addEventListener('mousemove', (e) => this.handleDragging(e));
        document.addEventListener('mouseup', () => this.stopDragging());
        document.addEventListener('touchmove', (e) => this.handleDragging(e));
        document.addEventListener('touchend', () => this.stopDragging());

        document.getElementById('addKeyframeBtn')?.addEventListener('click',
            () => this.addKeyframe(this.video.currentTime));
        document.getElementById('clearKeyframesBtn')?.addEventListener('click',
            () => this.clearKeyframes());

        document.getElementById('zoomInBtn')?.addEventListener('click', () => this.zoom(1.25));
        document.getElementById('zoomOutBtn')?.addEventListener('click', () => this.zoom(0.8));

        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (e.key.toLowerCase() === 'k' && this.container?.style.display !== 'none') {
                e.preventDefault();
                this.addKeyframe(this.video.currentTime);
            }
        });
    }

    async onVideoLoaded() {
        if (!this.video.duration || this.video.duration === Infinity) return;

        this.container.style.display = 'block';

        const duration = this.video.duration;
        const interval = Math.max(0.5, duration / this.options.maxThumbnails);
        this.options.thumbnailInterval = interval;

        this.extractCanvas.width = this.options.thumbnailWidth * 2;
        this.extractCanvas.height = this.options.thumbnailHeight * 2;

        await this.generateThumbnails();
        this.buildTimeScale();
        this.updatePlayhead();
    }

    async generateThumbnails() {
        if (this.isGenerating) return;
        this.isGenerating = true;

        this.thumbnailContainer.innerHTML = '<div class="thumbnail-loading">Generating thumbnails...</div>';

        const duration = this.video.duration;
        const interval = this.options.thumbnailInterval;
        const originalTime = this.video.currentTime;
        const wasMuted = this.video.muted;
        const wasPlaying = !this.video.paused;

        if (wasPlaying) this.video.pause();
        this.video.muted = true;

        this.thumbnails = [];

        try {
            const thumbnailCount = Math.ceil(duration / interval);

            for (let i = 0; i < thumbnailCount && i < this.options.maxThumbnails; i++) {
                const time = i * interval;
                const thumbnail = await this.extractFrameAtTime(time);
                this.thumbnails.push({
                    time: time,
                    dataUrl: thumbnail
                });
            }

            this.renderThumbnails();
            console.log(`[VideoTimeline] Generated ${this.thumbnails.length} thumbnails`);

        } catch (error) {
            console.error('[VideoTimeline] Thumbnail generation failed:', error);
            this.thumbnailContainer.innerHTML = '<div class="thumbnail-loading">Click track to seek</div>';
        } finally {
            this.video.currentTime = originalTime;
            this.video.muted = wasMuted;
            if (wasPlaying) this.video.play();
            this.isGenerating = false;
        }
    }

    extractFrameAtTime(time) {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                this.video.removeEventListener('seeked', seekHandler);
                resolve(this.createPlaceholder(time));
            }, 2000);

            const seekHandler = () => {
                clearTimeout(timeout);
                try {
                    this.extractCtx.drawImage(
                        this.video,
                        0, 0,
                        this.extractCanvas.width,
                        this.extractCanvas.height
                    );
                    const dataUrl = this.extractCanvas.toDataURL('image/jpeg', 0.6);
                    this.video.removeEventListener('seeked', seekHandler);
                    resolve(dataUrl);
                } catch (e) {
                    this.video.removeEventListener('seeked', seekHandler);
                    resolve(this.createPlaceholder(time));
                }
            };

            this.video.addEventListener('seeked', seekHandler);
            this.video.currentTime = time;
        });
    }

    createPlaceholder(time) {
        const canvas = document.createElement('canvas');
        canvas.width = this.options.thumbnailWidth;
        canvas.height = this.options.thumbnailHeight;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#667eea';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(this.formatTime(time), canvas.width / 2, canvas.height / 2);
        return canvas.toDataURL('image/jpeg', 0.8);
    }

    renderThumbnails() {
        this.thumbnailContainer.innerHTML = '';

        const totalWidth = this.thumbnails.length * this.options.thumbnailWidth;
        this.thumbnailContainer.style.width = `${Math.max(100, totalWidth)}%`;

        this.thumbnails.forEach((thumb, index) => {
            const div = document.createElement('div');
            div.className = 'thumbnail-item';
            div.style.width = `${this.options.thumbnailWidth}px`;
            div.style.backgroundImage = `url(${thumb.dataUrl})`;
            div.dataset.time = thumb.time;
            div.title = this.formatTime(thumb.time);
            this.thumbnailContainer.appendChild(div);
        });
    }

    buildTimeScale() {
        if (!this.timeScale) return;

        this.timeScale.innerHTML = '';
        const duration = this.video.duration;
        const markerCount = Math.min(10, Math.ceil(duration / 5));
        const interval = duration / markerCount;

        for (let i = 0; i <= markerCount; i++) {
            const time = i * interval;
            const marker = document.createElement('span');
            marker.className = 'time-marker';
            marker.textContent = this.formatTime(time);
            this.timeScale.appendChild(marker);
        }
    }

    updatePlayhead() {
        if (this.isDragging || !this.playhead || !this.video.duration) return;

        const percent = (this.video.currentTime / this.video.duration) * 100;
        this.playhead.style.left = `${percent}%`;

        if (this.playheadTime) {
            this.playheadTime.textContent = this.formatTimeDetailed(this.video.currentTime);
        }
    }

    handleTrackClick(e) {
        if (e.target.closest('.keyframe-marker')) return;

        const rect = this.thumbnailTrack.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        const time = percent * this.video.duration;

        this.video.currentTime = Math.max(0, Math.min(time, this.video.duration));

        if (this.options.onSeek) {
            this.options.onSeek(time);
        }
    }

    startDragging(e) {
        e.preventDefault();
        this.isDragging = true;
        this.playhead.style.cursor = 'grabbing';
        this.video.pause();
    }

    handleDragging(e) {
        if (!this.isDragging || !this.thumbnailTrack) return;

        const clientX = e.clientX || (e.touches && e.touches[0]?.clientX);
        if (clientX === undefined) return;

        const rect = this.thumbnailTrack.getBoundingClientRect();
        const percent = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        const time = percent * this.video.duration;

        this.playhead.style.left = `${percent * 100}%`;
        if (this.playheadTime) {
            this.playheadTime.textContent = this.formatTimeDetailed(time);
        }
        this.video.currentTime = time;
    }

    stopDragging() {
        if (!this.isDragging) return;
        this.isDragging = false;
        if (this.playhead) {
            this.playhead.style.cursor = 'ew-resize';
        }
    }

    addKeyframe(time) {
        const exists = this.keyframes.some(kf => Math.abs(kf.time - time) < 0.5);
        if (exists) {
            console.log('[VideoTimeline] Keyframe already exists at this time');
            return null;
        }

        const id = ++this.keyframeIdCounter;
        const percent = (time / this.video.duration) * 100;

        const keyframe = {
            id: id,
            time: time,
            position: percent
        };

        this.keyframes.push(keyframe);
        this.renderKeyframeMarker(keyframe);

        console.log(`[VideoTimeline] Keyframe added at ${this.formatTimeDetailed(time)}`);

        if (this.options.onKeyframeAdd) {
            this.options.onKeyframeAdd(keyframe);
        }

        return keyframe;
    }

    removeKeyframe(id) {
        const index = this.keyframes.findIndex(kf => kf.id === id);
        if (index === -1) return;

        const keyframe = this.keyframes[index];
        this.keyframes.splice(index, 1);

        const marker = document.querySelector(`.keyframe-marker[data-id="${id}"]`);
        if (marker) marker.remove();

        console.log(`[VideoTimeline] Keyframe ${id} removed`);

        if (this.options.onKeyframeRemove) {
            this.options.onKeyframeRemove(keyframe);
        }
    }

    renderKeyframeMarker(keyframe) {
        const marker = document.createElement('div');
        marker.className = 'keyframe-marker';
        marker.dataset.id = keyframe.id;
        marker.style.left = `${keyframe.position}%`;
        marker.title = `Keyframe at ${this.formatTimeDetailed(keyframe.time)} (click to remove)`;

        marker.innerHTML = `
            <svg width="12" height="16" viewBox="0 0 12 16" fill="#48bb78">
                <path d="M6 0L12 8L6 16L0 8L6 0Z"/>
            </svg>
        `;

        marker.addEventListener('click', (e) => {
            e.stopPropagation();
            this.removeKeyframe(keyframe.id);
        });

        marker.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            this.video.currentTime = keyframe.time;
        });

        this.keyframeMarkersContainer.appendChild(marker);
    }

    clearKeyframes() {
        this.keyframes = [];
        if (this.keyframeMarkersContainer) {
            this.keyframeMarkersContainer.innerHTML = '';
        }
        console.log('[VideoTimeline] All keyframes cleared');
    }

    zoom(factor) {
        this.zoomLevel = Math.max(0.5, Math.min(3, this.zoomLevel * factor));

        if (this.thumbnailContainer) {
            this.thumbnailContainer.style.transform = `scaleX(${this.zoomLevel})`;
            this.thumbnailContainer.style.transformOrigin = 'left';
        }

        const zoomLevelEl = document.getElementById('zoomLevel');
        if (zoomLevelEl) {
            zoomLevelEl.textContent = `${Math.round(this.zoomLevel * 100)}%`;
        }
    }

    getKeyframes() {
        return [...this.keyframes];
    }

    addMaskIndicator(time) {
        const keyframe = this.addKeyframe(time);
        if (keyframe) {
            const marker = document.querySelector(`.keyframe-marker[data-id="${keyframe.id}"]`);
            if (marker) {
                marker.classList.add('mask-keyframe');
                marker.querySelector('svg').setAttribute('fill', '#00ff00');
            }
        }
        return keyframe;
    }

    clearMaskIndicators() {
        const maskMarkers = document.querySelectorAll('.keyframe-marker.mask-keyframe');
        maskMarkers.forEach(marker => {
            const id = parseInt(marker.dataset.id);
            this.removeKeyframe(id);
        });
    }

    show() {
        if (this.container) this.container.style.display = 'block';
    }

    hide() {
        if (this.container) this.container.style.display = 'none';
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatTimeDetailed(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 100);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
    }

    destroy() {
        this.clearKeyframes();
        this.thumbnails = [];
        if (this.container) {
            this.container.style.display = 'none';
        }
    }
}

window.VideoTimeline = VideoTimeline;
