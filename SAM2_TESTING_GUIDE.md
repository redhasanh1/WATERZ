# SAM2 Click Position Testing & Debug Guide

## Quick Test (30 seconds)

### Option 1: Load Debug Script (Recommended)

1. Open https://markremoverai.com/
2. Upload a video
3. Press `F12` to open Developer Console
4. Paste this one line:

```javascript
fetch('/debug_sam2.js').then(r=>r.text()).then(eval).then(()=>debug.enable())
```

5. Click on the video - you'll see:
   - Red box around the actual video area
   - Blue box around the container
   - Green circles where you click (if inside video)
   - Red circles if you click in letterbox
   - Debug stats in bottom-left

6. Run automated test:
```javascript
debug.runAccuracyTest(20)
```

**Expected result:** Avg error < 1px = PERFECT ✅

---

### Option 2: Manual Console Test (No external files)

Paste this into console (F12):

```javascript
// Quick visual test
const sam2 = window.sam2Selector;
const canvas = document.getElementById('maskCanvas');
const ctx = canvas.getContext('2d');

// Draw debug borders
ctx.strokeStyle = '#ff0000';
ctx.lineWidth = 5;
ctx.strokeRect(
    sam2.offsetX,
    sam2.offsetY,
    sam2.renderWidth,
    sam2.renderHeight
);

console.log('RED BOX = actual video area');
console.log('Render size:', sam2.renderWidth, 'x', sam2.renderHeight);
console.log('Offset:', sam2.offsetX, sam2.offsetY);

// Add click logger
canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const relX = clickX - sam2.offsetX;
    const relY = clickY - sam2.offsetY;

    console.log('Click:', {
        screen: [clickX.toFixed(1), clickY.toFixed(1)],
        relative: [relX.toFixed(1), relY.toFixed(1)],
        inVideo: relX >= 0 && relY >= 0 && relX <= sam2.renderWidth && relY <= sam2.renderHeight
    });
});

console.log('✅ Click the video to see coordinate mapping');
```

---

## Full Debug Mode Commands

After loading debug script:

```javascript
// Enable debug overlays
debug.enable()

// Run 50-point accuracy test
debug.runAccuracyTest(50)

// Performance benchmark
debug.benchmark(1000)

// Print current state
debug.printState()

// Disable debug mode
debug.disable()

// Show help
debug.help()
```

---

## What You Should See

### ✅ CORRECT (Fixed)
- Green dot appears **exactly** where you click
- Console shows: `Avg error: 0.34px` (< 1px)
- Red box perfectly outlines the video (no gaps)
- Clicks in black bars are ignored

### ❌ WRONG (Still broken)
- Green dot appears offset (up/left by ~7cm)
- Console shows: `Avg error: 25.67px` (> 5px)
- Red box doesn't match video edges
- Clicks in black bars create dots

---

## Testing Different Scenarios

### Test 1: Normal Video (16:9 in wide container)
- Upload horizontal video
- Should have letterbox bars on top/bottom OR none
- offsetY should be > 0 or offsetY = 0

### Test 2: Portrait Video (9:16 in wide container)
- Upload vertical/portrait video
- Should have pillarbox bars on left/right
- offsetX should be > 0

### Test 3: Window Resize
- Click video → note dot position
- Resize browser window
- Click same spot → dot should still be accurate

### Test 4: Different Aspect Ratios
- Test with: 16:9, 4:3, 1:1, 9:16 videos
- All should work perfectly

---

## Interpreting Results

### Accuracy Test Output

```javascript
debug.runAccuracyTest(10)
```

**Result Table:**
| test | click | expected | actual | error |
|------|-------|----------|--------|-------|
| 1 | (450.2, 300.5) | (720, 405) | (720, 405) | 0.00px |
| 2 | (380.7, 250.1) | (610, 337) | (610, 337) | 0.00px |

**Summary:**
- `Avg error: 0.00px` = ✅ PIXEL PERFECT
- `Avg error: 0.5-2px` = ✅ EXCELLENT (rounding variance)
- `Avg error: 2-5px` = ⚠️ ACCEPTABLE (minor issue)
- `Avg error: > 5px` = ❌ BROKEN (needs fix)

---

## Performance Benchmark

```javascript
debug.benchmark(1000)
```

**Expected output:**
```
Iterations: 1000
Total time: 15.23ms
Avg time per op: 0.0152ms
Operations/sec: 65,000
```

**Good performance:** > 10,000 ops/sec

---

## Common Issues & Solutions

### Issue: "SAM2Selector not initialized"
**Solution:** Upload a video first, wait for it to load

### Issue: Debug overlay not showing
**Solution:** Make sure you called `debug.enable()`

### Issue: Red box doesn't appear
**Solution:**
```javascript
sam2.calculateVideoRenderBounds()
debug.updateDebugOverlay()
```

### Issue: Still seeing offset
**Solution:**
1. Hard refresh: `Ctrl+Shift+R`
2. Clear cache in Railway deployment
3. Check commit is deployed: `cb9e01f5`

---

## Automated Testing Script

Run this complete test suite:

```javascript
async function runFullTest() {
    console.log('🧪 Running full SAM2 test suite...\n');

    // 1. State check
    console.log('1️⃣ Checking state...');
    debug.printState();

    // 2. Accuracy test
    console.log('\n2️⃣ Running accuracy test...');
    await debug.runAccuracyTest(20);

    // 3. Performance test
    console.log('\n3️⃣ Running performance test...');
    debug.benchmark(1000);

    // 4. Visual test
    console.log('\n4️⃣ Enabling visual debug...');
    debug.enable();

    console.log('\n✅ Test suite complete! Click the video to verify visually.');
}

runFullTest();
```

---

## Reporting Results

If the fix works, you should see:
```
✅ EXCELLENT - Coordinate mapping is pixel-perfect!
Avg error: 0.34px, Max error: 1.00px
```

If still broken:
```
❌ POOR - Significant offset detected
Avg error: 25.67px, Max error: 45.23px
```

Share the console output and I'll debug further!

---

## Quick Fixes

### If offset persists after deployment:

```javascript
// Force recalculation
sam2Selector.calculateVideoRenderBounds()
sam2Selector.draw()

// Check values
console.log('Render:', sam2Selector.renderWidth, sam2Selector.renderHeight)
console.log('Offset:', sam2Selector.offsetX, sam2Selector.offsetY)
```

### If letterbox detection wrong:

```javascript
// Manual override (example values)
sam2Selector.renderWidth = 800
sam2Selector.renderHeight = 450
sam2Selector.offsetX = 0
sam2Selector.offsetY = 225
sam2Selector.draw()
```

---

## Expected Console Logs

When working correctly:

```
[SAM2Selector] Initialized
[SAM2Selector] Canvas: 1920x1080, Display: 800x900
[SAM2Selector] Video render: 800.0x450.0, Offset: (0.0, 225.0)
[SAM2 Debug] Click Event
  click: { x: 400, y: 450 }
  relative: { x: 400, y: 225 }
  video: { x: 960, y: 540 }
  inVideoArea: true
```

When broken:

```
[SAM2Selector] Video render: undefined x undefined, Offset: (undefined, undefined)
[SAM2 Debug] Click Event
  relative: { x: NaN, y: NaN }
  inVideoArea: false
```
