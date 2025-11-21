#!/usr/bin/env python3
"""Add test button to index.html"""

with open('web/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the Clear Points button and add Test button after it
old_buttons = '''                                <button id="clearPointsBtn" style="padding: 0.8rem 2rem; border-radius: 12px; border: none; background: var(--bg-gradient-2); color: white; cursor: pointer; font-size: 1rem; font-weight: 600; box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3); transition: all 0.3s ease;" disabled>
                                    Clear Points
                                </button>'''

new_buttons = '''                                <button id="clearPointsBtn" style="padding: 0.8rem 2rem; border-radius: 12px; border: none; background: var(--bg-gradient-2); color: white; cursor: pointer; font-size: 1rem; font-weight: 600; box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3); transition: all 0.3s ease;" disabled>
                                    Clear Points
                                </button>
                                <button id="testSAM2Btn" style="display:none; padding: 0.8rem 2rem; border-radius: 12px; border: none; background: var(--bg-gradient-3); color: white; cursor: pointer; font-size: 1rem; font-weight: 600; box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3); transition: all 0.3s ease;">
                                    🧪 Test SAM2 Connection
                                </button>'''

if old_buttons in content:
    content = content.replace(old_buttons, new_buttons)
    print("[OK] Added test button to HTML")
else:
    print("[ERROR] Could not find button section")
    exit(1)

# Now add the JavaScript test function after the enable button handler
old_handler = '''                // Enable SAM2 selector
                if (sam2Selector) {
                    sam2Selector.enabled = true;
                    console.log('[SAM2] Selection mode enabled');
                }
            } else {
                enableBtn.textContent = 'Enable Selection Mode';
                enableBtn.style.background = 'var(--gradient-primary)';
                canvas.style.pointerEvents = 'none';

                // Disable SAM2 selector
                if (sam2Selector) {
                    sam2Selector.enabled = false;
                    console.log('[SAM2] Selection mode disabled');
                }
            }
        });'''

new_handler = '''                // Enable SAM2 selector
                if (sam2Selector) {
                    sam2Selector.enabled = true;
                    console.log('[SAM2] Selection mode enabled');
                }

                // Show test button
                document.getElementById('testSAM2Btn').style.display = 'inline-block';
            } else {
                enableBtn.textContent = 'Enable Selection Mode';
                enableBtn.style.background = 'var(--gradient-primary)';
                canvas.style.pointerEvents = 'none';

                // Disable SAM2 selector
                if (sam2Selector) {
                    sam2Selector.enabled = false;
                    console.log('[SAM2] Selection mode disabled');
                }

                // Hide test button
                document.getElementById('testSAM2Btn').style.display = 'none';
            }
        });

        // Test SAM2 connection button handler
        const testSAM2Btn = document.getElementById('testSAM2Btn');
        if (testSAM2Btn) {
            testSAM2Btn.addEventListener('click', async function() {
                console.log('[TEST] 🧪 Sending test request to SAM2 worker...');

                // Create 100x100 test frame (white square)
                const testCanvas = document.createElement('canvas');
                testCanvas.width = 100;
                testCanvas.height = 100;
                const testCtx = testCanvas.getContext('2d');
                testCtx.fillStyle = 'white';
                testCtx.fillRect(0, 0, 100, 100);
                const testFrame = testCanvas.toDataURL('image/png').split(',')[1];

                console.log('[TEST] 📤 Sending to /api/sam2/select-object');

                try {
                    const response = await fetch('/api/sam2/select-object', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            frame_data: testFrame,
                            points: [{x: 50, y: 50, label: 1}],
                            video_width: 100,
                            video_height: 100
                        })
                    });

                    const data = await response.json();
                    console.log('[TEST] 📥 Response:', data);

                    if (data.status === 'success') {
                        alert('✅ SAM2 worker connected!\\n\\nMask received successfully!');
                    } else {
                        alert('❌ SAM2 worker failed:\\n\\n' + data.message);
                    }
                } catch (error) {
                    console.error('[TEST] ❌ Error:', error);
                    alert('❌ Network error:\\n\\n' + error.message);
                }
            });
        }'''

if old_handler in content:
    content = content.replace(old_handler, new_handler)
    print("[OK] Added test button handler")
else:
    print("[ERROR] Could not find handler section")
    exit(1)

with open('web/index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Successfully added test button and handler to index.html")
