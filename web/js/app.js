// AI Watermark Remover - Main Application JavaScript

class WatermarkRemover {
    constructor() {
        this.currentFile = null;
        this.processedFile = null;
        this.isVideo = false;

        this.initElements();
        this.initEventListeners();
    }

    initElements() {
        // Upload elements
        this.uploadZone = document.getElementById('uploadZone');
        this.fileInput = document.getElementById('fileInput');
        this.uploadContent = document.getElementById('uploadContent');
        this.uploadPreview = document.getElementById('uploadPreview');
        this.previewImage = document.getElementById('previewImage');
        this.previewVideo = document.getElementById('previewVideo');
        this.previewFilename = document.getElementById('previewFilename');
        this.removeFileBtn = document.getElementById('removeFile');
        this.processBtn = document.getElementById('processBtn');

        // Processing elements
        this.processingCard = document.getElementById('processingCard');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');

        // Results elements
        this.resultsSection = document.getElementById('resultsSection');
        this.beforeImage = document.getElementById('beforeImage');
        this.beforeVideo = document.getElementById('beforeVideo');
        this.afterImage = document.getElementById('afterImage');
        this.afterVideo = document.getElementById('afterVideo');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.processAnotherBtn = document.getElementById('processAnotherBtn');
        this.comparisonWrapper = document.getElementById('comparisonWrapper');
        this.comparisonSlider = document.getElementById('comparisonSlider');
    }

    initEventListeners() {
        // File upload
        this.uploadZone.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files[0]));

        // Drag and drop
        this.uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadZone.classList.add('dragover');
        });

        this.uploadZone.addEventListener('dragleave', () => {
            this.uploadZone.classList.remove('dragover');
        });

        this.uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) this.handleFileSelect(file);
        });

        // Remove file
        this.removeFileBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.resetUpload();
        });

        // Process button
        this.processBtn.addEventListener('click', () => this.processFile());

        // Results actions
        this.downloadBtn.addEventListener('click', () => this.downloadResult());
        this.processAnotherBtn.addEventListener('click', () => this.reset());

        // Comparison slider
        this.initComparisonSlider();
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        const validImageTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        const validVideoTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];

        if (!validImageTypes.includes(file.type) && !validVideoTypes.includes(file.type)) {
            alert('Please upload a valid image (JPG, PNG) or video (MP4, MOV) file.');
            return;
        }

        // Validate file size (100MB)
        if (file.size > 100 * 1024 * 1024) {
            alert('File size must be less than 100MB.');
            return;
        }

        this.currentFile = file;
        this.isVideo = validVideoTypes.includes(file.type);
        this.showPreview(file);
    }

    showPreview(file) {
        const reader = new FileReader();

        reader.onload = (e) => {
            this.uploadContent.style.display = 'none';
            this.uploadPreview.style.display = 'flex';

            if (this.isVideo) {
                this.previewImage.style.display = 'none';
                this.previewVideo.style.display = 'block';
                this.previewVideo.src = e.target.result;
            } else {
                this.previewVideo.style.display = 'none';
                this.previewImage.style.display = 'block';
                this.previewImage.src = e.target.result;
            }

            this.previewFilename.textContent = file.name;
            this.processBtn.disabled = false;
        };

        reader.readAsDataURL(file);
    }

    resetUpload() {
        this.currentFile = null;
        this.fileInput.value = '';
        this.uploadContent.style.display = 'flex';
        this.uploadPreview.style.display = 'none';
        this.previewImage.src = '';
        this.previewVideo.src = '';
        this.processBtn.disabled = true;
    }

    async processFile() {
        if (!this.currentFile) return;

        // Show processing UI
        this.processingCard.style.display = 'block';
        this.processBtn.disabled = true;

        // Scroll to processing section
        this.processingCard.scrollIntoView({ behavior: 'smooth', block: 'center' });

        // Create FormData
        const formData = new FormData();
        formData.append('file', this.currentFile);

        try {
            // Simulate progress for better UX
            this.updateProgress(10);

            const response = await fetch('/api/remove-watermark', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Processing failed');
            }

            this.updateProgress(50);

            // Get the processed file as blob
            const blob = await response.blob();
            this.processedFile = blob;

            this.updateProgress(100);

            // Wait a moment before showing results
            setTimeout(() => {
                this.showResults();
            }, 500);

        } catch (error) {
            console.error('Error processing file:', error);
            alert('Failed to process file. Please try again.');
            this.processingCard.style.display = 'none';
            this.processBtn.disabled = false;
        }
    }

    updateProgress(percent) {
        this.progressFill.style.width = percent + '%';
        this.progressText.textContent = Math.round(percent) + '%';
    }

    showResults() {
        // Hide processing
        this.processingCard.style.display = 'none';

        // Setup before/after comparison
        const beforeURL = URL.createObjectURL(this.currentFile);
        const afterURL = URL.createObjectURL(this.processedFile);

        if (this.isVideo) {
            this.beforeImage.style.display = 'none';
            this.afterImage.style.display = 'none';
            this.beforeVideo.style.display = 'block';
            this.afterVideo.style.display = 'block';
            this.beforeVideo.src = beforeURL;
            this.afterVideo.src = afterURL;

            // Sync video playback
            this.beforeVideo.addEventListener('play', () => this.afterVideo.play());
            this.beforeVideo.addEventListener('pause', () => this.afterVideo.pause());
            this.beforeVideo.addEventListener('seeked', () => {
                this.afterVideo.currentTime = this.beforeVideo.currentTime;
            });
        } else {
            this.beforeVideo.style.display = 'none';
            this.afterVideo.style.display = 'none';
            this.beforeImage.style.display = 'block';
            this.afterImage.style.display = 'block';
            this.beforeImage.src = beforeURL;
            this.afterImage.src = afterURL;
        }

        // Show results section
        this.resultsSection.style.display = 'block';
        this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    initComparisonSlider() {
        let isDragging = false;

        const updateSliderPosition = (e) => {
            const rect = this.comparisonWrapper.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const percent = Math.max(0, Math.min(100, (x / rect.width) * 100));

            this.comparisonSlider.style.left = percent + '%';

            const afterImage = this.comparisonWrapper.querySelector('.comparison-image.after');
            afterImage.style.clipPath = `inset(0 0 0 ${percent}%)`;
        };

        this.comparisonSlider.addEventListener('mousedown', (e) => {
            isDragging = true;
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            updateSliderPosition(e);
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
        });

        // Touch support
        this.comparisonSlider.addEventListener('touchstart', (e) => {
            isDragging = true;
            e.preventDefault();
        });

        document.addEventListener('touchmove', (e) => {
            if (!isDragging) return;
            const touch = e.touches[0];
            updateSliderPosition(touch);
        });

        document.addEventListener('touchend', () => {
            isDragging = false;
        });

        // Click to set position
        this.comparisonWrapper.addEventListener('click', (e) => {
            if (e.target === this.comparisonSlider || e.target.closest('.slider-button')) return;
            updateSliderPosition(e);
        });
    }

    downloadResult() {
        if (!this.processedFile) return;

        const url = URL.createObjectURL(this.processedFile);
        const a = document.createElement('a');
        a.href = url;

        const extension = this.isVideo ? 'mp4' : 'png';
        const timestamp = new Date().getTime();
        a.download = `watermark-removed-${timestamp}.${extension}`;

        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    reset() {
        // Reset all states
        this.resetUpload();
        this.processedFile = null;
        this.isVideo = false;

        // Hide results
        this.resultsSection.style.display = 'none';
        this.processingCard.style.display = 'none';

        // Reset progress
        this.updateProgress(0);

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new WatermarkRemover();
});
