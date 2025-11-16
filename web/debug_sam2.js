/**
 * SAM2 Selector Debug & Test Utilities
 * Paste this entire file into browser console (F12) to enable debugging
 */

class SAM2Debugger {
    constructor() {
        this.enabled = false;
        this.clickLog = [];
        this.stats = {
            totalClicks: 0,
            offsetErrors: [],
            avgError: 0,
            maxError: 0
        };
    }

    /**
     * Enable debug mode with visual overlays
     */
    enable() {
        this.enabled = true;
        this.addVisualOverlays();
        this.interceptClicks();
        console.log('%c[SAM2 Debug] Debug mode ENABLED', 'color: #00ff00; font-weight: bold; font-size: 14px');
        console.log('Click on the video to see debug info');
    }

    /**
     * Disable debug mode
     */
    disable() {
        this.enabled = false;
        this.removeVisualOverlays();
        console.log('%c[SAM2 Debug] Debug mode DISABLED', 'color: #ff0000; font-weight: bold; font-size: 14px');
    }

    /**
     * Add visual debug overlays
     */
    addVisualOverlays() {
        const canvas = document.getElementById('maskCanvas');
        const video = document.getElementById('previewVideo');

        if (!canvas || !video) {
            console.error('[SAM2 Debug] Canvas or video not found');
            return;
        }

        const sam2 = window.sam2Selector;
        if (!sam2) {
            console.error('[SAM2 Debug] SAM2Selector not initialized');
            return;
        }

        // Create debug canvas overlay
        if (!this.debugCanvas) {
            this.debugCanvas = document.createElement('canvas');
            this.debugCanvas.id = 'sam2-debug-overlay';
            this.debugCanvas.style.cssText = `
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 10000;
            `;
            canvas.parentElement.appendChild(this.debugCanvas);
            this.debugCtx = this.debugCanvas.getContext('2d');
        }

        this.updateDebugOverlay();
    }

    /**
     * Update debug overlay visualization
     */
    updateDebugOverlay() {
        const sam2 = window.sam2Selector;
        if (!sam2 || !this.debugCanvas) return;

        const canvas = document.getElementById('maskCanvas');
        const rect = canvas.getBoundingClientRect();

        // Resize debug canvas to match
        this.debugCanvas.width = rect.width;
        this.debugCanvas.height = rect.height;

        const ctx = this.debugCtx;
        ctx.clearRect(0, 0, this.debugCanvas.width, this.debugCanvas.height);

        // Draw container boundary (blue)
        ctx.strokeStyle = '#0088ff';
        ctx.lineWidth = 2;
        ctx.strokeRect(1, 1, this.debugCanvas.width - 2, this.debugCanvas.height - 2);
        ctx.fillStyle = '#0088ff';
        ctx.font = '12px monospace';
        ctx.fillText('Container', 5, 15);

        // Draw video render area boundary (red)
        if (sam2.renderWidth && sam2.renderHeight) {
            ctx.strokeStyle = '#ff0000';
            ctx.lineWidth = 3;
            ctx.strokeRect(
                sam2.offsetX,
                sam2.offsetY,
                sam2.renderWidth,
                sam2.renderHeight
            );
            ctx.fillStyle = '#ff0000';
            ctx.fillText('Video Render Area', sam2.offsetX + 5, sam2.offsetY + 15);

            // Draw letterbox areas with semi-transparent overlay
            ctx.fillStyle = 'rgba(255, 0, 0, 0.1)';

            // Top letterbox
            if (sam2.offsetY > 0) {
                ctx.fillRect(0, 0, this.debugCanvas.width, sam2.offsetY);
            }
            // Bottom letterbox
            if (sam2.offsetY + sam2.renderHeight < this.debugCanvas.height) {
                ctx.fillRect(0, sam2.offsetY + sam2.renderHeight, this.debugCanvas.width, this.debugCanvas.height - (sam2.offsetY + sam2.renderHeight));
            }
            // Left letterbox
            if (sam2.offsetX > 0) {
                ctx.fillRect(0, 0, sam2.offsetX, this.debugCanvas.height);
            }
            // Right letterbox
            if (sam2.offsetX + sam2.renderWidth < this.debugCanvas.width) {
                ctx.fillRect(sam2.offsetX + sam2.renderWidth, 0, this.debugCanvas.width - (sam2.offsetX + sam2.renderWidth), this.debugCanvas.height);
            }
        }

        // Draw stats overlay
        this.drawStatsOverlay();
    }

    /**
     * Draw statistics overlay
     */
    drawStatsOverlay() {
        const sam2 = window.sam2Selector;
        if (!sam2) return;

        const ctx = this.debugCtx;
        const x = 10;
        let y = this.debugCanvas.height - 150;

        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(x - 5, y - 15, 300, 140);

        // Stats text
        ctx.fillStyle = '#00ff00';
        ctx.font = 'bold 14px monospace';
        ctx.fillText('SAM2 DEBUG INFO', x, y);

        ctx.font = '12px monospace';
        y += 20;
        ctx.fillText(`Video: ${sam2.videoWidth}x${sam2.videoHeight}`, x, y);
        y += 18;
        ctx.fillText(`Render: ${sam2.renderWidth?.toFixed(1) || 'N/A'}x${sam2.renderHeight?.toFixed(1) || 'N/A'}`, x, y);
        y += 18;
        ctx.fillText(`Offset: (${sam2.offsetX?.toFixed(1) || 'N/A'}, ${sam2.offsetY?.toFixed(1) || 'N/A'})`, x, y);
        y += 18;
        ctx.fillText(`Clicks: ${this.stats.totalClicks}`, x, y);
        y += 18;
        ctx.fillText(`Avg Error: ${this.stats.avgError.toFixed(2)}px`, x, y);
        y += 18;
        ctx.fillText(`Max Error: ${this.stats.maxError.toFixed(2)}px`, x, y);
    }

    /**
     * Intercept clicks to log debug info
     */
    interceptClicks() {
        const canvas = document.getElementById('maskCanvas');
        if (!canvas) return;

        canvas.addEventListener('click', (e) => {
            if (!this.enabled) return;
            this.logClickDebug(e);
        }, true);
    }

    /**
     * Log detailed click debug information
     */
    logClickDebug(e) {
        const sam2 = window.sam2Selector;
        if (!sam2) return;

        const canvas = document.getElementById('maskCanvas');
        const rect = canvas.getBoundingClientRect();

        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;

        const relativeX = clickX - sam2.offsetX;
        const relativeY = clickY - sam2.offsetY;

        const inVideoArea = relativeX >= 0 && relativeY >= 0 &&
                           relativeX <= sam2.renderWidth && relativeY <= sam2.renderHeight;

        let videoX = null, videoY = null;
        if (inVideoArea) {
            videoX = Math.floor((relativeX / sam2.renderWidth) * sam2.canvas.width);
            videoY = Math.floor((relativeY / sam2.renderHeight) * sam2.canvas.height);
        }

        const debugInfo = {
            click: { x: clickX.toFixed(2), y: clickY.toFixed(2) },
            relative: { x: relativeX.toFixed(2), y: relativeY.toFixed(2) },
            video: videoX !== null ? { x: videoX, y: videoY } : 'OUTSIDE',
            renderBounds: {
                width: sam2.renderWidth.toFixed(2),
                height: sam2.renderHeight.toFixed(2),
                offset: { x: sam2.offsetX.toFixed(2), y: sam2.offsetY.toFixed(2) }
            },
            inVideoArea
        };

        console.log('%c[SAM2 Debug] Click Event', 'color: #00ffff; font-weight: bold');
        console.table(debugInfo);

        this.stats.totalClicks++;
        this.clickLog.push(debugInfo);

        // Update overlay
        this.updateDebugOverlay();
        this.drawClickMarker(clickX, clickY, inVideoArea);
    }

    /**
     * Draw click marker on debug overlay
     */
    drawClickMarker(x, y, inVideoArea) {
        const ctx = this.debugCtx;

        ctx.strokeStyle = inVideoArea ? '#00ff00' : '#ff0000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.stroke();

        // Crosshair
        ctx.beginPath();
        ctx.moveTo(x - 15, y);
        ctx.lineTo(x + 15, y);
        ctx.moveTo(x, y - 15);
        ctx.lineTo(x, y + 15);
        ctx.stroke();

        // Fade out after 2 seconds
        setTimeout(() => this.updateDebugOverlay(), 2000);
    }

    /**
     * Remove debug overlays
     */
    removeVisualOverlays() {
        if (this.debugCanvas) {
            this.debugCanvas.remove();
            this.debugCanvas = null;
            this.debugCtx = null;
        }
    }

    /**
     * Run automated accuracy test
     */
    async runAccuracyTest(numTests = 10) {
        console.log(`%c[SAM2 Debug] Running accuracy test with ${numTests} points...`, 'color: #ffff00; font-weight: bold');

        const sam2 = window.sam2Selector;
        if (!sam2) {
            console.error('[SAM2 Debug] SAM2Selector not initialized');
            return;
        }

        const results = [];
        const canvas = document.getElementById('maskCanvas');
        const rect = canvas.getBoundingClientRect();

        // Test points in video area
        for (let i = 0; i < numTests; i++) {
            const testX = sam2.offsetX + Math.random() * sam2.renderWidth;
            const testY = sam2.offsetY + Math.random() * sam2.renderHeight;

            // Expected video coordinate
            const expectedVideoX = Math.floor(((testX - sam2.offsetX) / sam2.renderWidth) * sam2.canvas.width);
            const expectedVideoY = Math.floor(((testY - sam2.offsetY) / sam2.renderHeight) * sam2.canvas.height);

            // Simulate click and get actual result
            const mockEvent = {
                clientX: rect.left + testX,
                clientY: rect.top + testY
            };
            const point = sam2.getCanvasPoint(mockEvent);

            if (point) {
                const errorX = Math.abs(point.x - expectedVideoX);
                const errorY = Math.abs(point.y - expectedVideoY);
                const totalError = Math.sqrt(errorX * errorX + errorY * errorY);

                results.push({
                    test: i + 1,
                    click: `(${testX.toFixed(1)}, ${testY.toFixed(1)})`,
                    expected: `(${expectedVideoX}, ${expectedVideoY})`,
                    actual: `(${point.x}, ${point.y})`,
                    error: totalError.toFixed(2) + 'px'
                });

                this.stats.offsetErrors.push(totalError);
            }
        }

        this.stats.avgError = this.stats.offsetErrors.reduce((a, b) => a + b, 0) / this.stats.offsetErrors.length;
        this.stats.maxError = Math.max(...this.stats.offsetErrors);

        console.table(results);
        console.log(`%c[SAM2 Debug] Test complete! Avg error: ${this.stats.avgError.toFixed(2)}px, Max error: ${this.stats.maxError.toFixed(2)}px`,
                    'color: #00ff00; font-weight: bold');

        if (this.stats.avgError < 1) {
            console.log('%c✅ EXCELLENT - Coordinate mapping is pixel-perfect!', 'color: #00ff00; font-weight: bold; font-size: 16px');
        } else if (this.stats.avgError < 5) {
            console.log('%c⚠️ GOOD - Minor offset detected, acceptable for most use cases', 'color: #ffaa00; font-weight: bold');
        } else {
            console.log('%c❌ POOR - Significant offset detected, needs debugging', 'color: #ff0000; font-weight: bold');
        }

        return results;
    }

    /**
     * Benchmark performance
     */
    benchmark(iterations = 1000) {
        console.log(`%c[SAM2 Debug] Running performance benchmark...`, 'color: #ffff00; font-weight: bold');

        const sam2 = window.sam2Selector;
        if (!sam2) {
            console.error('[SAM2 Debug] SAM2Selector not initialized');
            return;
        }

        const canvas = document.getElementById('maskCanvas');
        const rect = canvas.getBoundingClientRect();

        // Benchmark coordinate transformation
        const start = performance.now();
        for (let i = 0; i < iterations; i++) {
            const mockEvent = {
                clientX: rect.left + Math.random() * rect.width,
                clientY: rect.top + Math.random() * rect.height
            };
            sam2.getCanvasPoint(mockEvent);
        }
        const end = performance.now();

        const avgTime = (end - start) / iterations;
        const opsPerSecond = 1000 / avgTime;

        console.log(`%c[SAM2 Debug] Benchmark Results:`, 'color: #00ff00; font-weight: bold');
        console.log(`  Iterations: ${iterations}`);
        console.log(`  Total time: ${(end - start).toFixed(2)}ms`);
        console.log(`  Avg time per op: ${avgTime.toFixed(4)}ms`);
        console.log(`  Operations/sec: ${opsPerSecond.toFixed(0)}`);

        return { avgTime, opsPerSecond };
    }

    /**
     * Print current state
     */
    printState() {
        const sam2 = window.sam2Selector;
        if (!sam2) {
            console.error('[SAM2 Debug] SAM2Selector not initialized');
            return;
        }

        console.log('%c[SAM2 Debug] Current State:', 'color: #00ffff; font-weight: bold');
        console.table({
            'Video Resolution': `${sam2.videoWidth}x${sam2.videoHeight}`,
            'Canvas Internal': `${sam2.canvas.width}x${sam2.canvas.height}`,
            'Canvas Display': `${sam2.canvas.getBoundingClientRect().width.toFixed(1)}x${sam2.canvas.getBoundingClientRect().height.toFixed(1)}`,
            'Render Size': `${sam2.renderWidth?.toFixed(1)}x${sam2.renderHeight?.toFixed(1)}`,
            'Offset': `(${sam2.offsetX?.toFixed(1)}, ${sam2.offsetY?.toFixed(1)})`,
            'Selections': sam2.selections?.length || 0,
            'Total Points': sam2.selections?.reduce((sum, s) => sum + s.points.length, 0) || 0
        });
    }

    /**
     * Show help
     */
    help() {
        console.log(`%c
╔═══════════════════════════════════════════════════════════════╗
║              SAM2 DEBUGGER - COMMAND REFERENCE                ║
╚═══════════════════════════════════════════════════════════════╝

debug.enable()           - Enable debug mode with visual overlays
debug.disable()          - Disable debug mode
debug.printState()       - Print current SAM2 state
debug.runAccuracyTest()  - Run automated accuracy test (10 points)
debug.runAccuracyTest(N) - Run accuracy test with N points
debug.benchmark()        - Run performance benchmark (1000 ops)
debug.benchmark(N)       - Run benchmark with N iterations
debug.help()             - Show this help

VISUAL OVERLAYS:
- Blue border   = Canvas container boundary
- Red border    = Actual video render area
- Red tint      = Letterbox areas (clicks ignored)
- Green circle  = Valid click inside video
- Red circle    = Invalid click in letterbox

        `, 'font-family: monospace; color: #00ff00');
    }
}

// Auto-initialize
window.sam2Debug = new SAM2Debugger();
console.log('%c[SAM2 Debug] Debugger loaded! Type "debug.help()" for commands', 'color: #00ff00; font-weight: bold; font-size: 14px');
console.log('Quick start: debug.enable()');

// Alias for convenience
window.debug = window.sam2Debug;
