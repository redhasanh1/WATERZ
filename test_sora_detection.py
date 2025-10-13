#!/usr/bin/env python3
"""
Test Sora watermark detection to diagnose why watermarks aren't being removed
"""

import cv2
import os
import sys

# Test with the actual YOLO detector
from yolo_detector import YOLOWatermarkDetector

def test_sora_detection():
    """Test detection on Sora sample frame"""
    
    # Load test frame
    test_frame_path = 'sora_test_frame.jpg'
    if not os.path.exists(test_frame_path):
        print(f"❌ Test frame not found: {test_frame_path}")
        return
    
    image = cv2.imread(test_frame_path)
    if image is None:
        print(f"❌ Could not load image: {test_frame_path}")
        return
    
    print(f"✅ Loaded test frame: {image.shape}")
    
    # Initialize YOLO detector
    print("\n" + "="*50)
    print("INITIALIZING SORA YOLO DETECTOR")
    print("="*50)
    
    detector = YOLOWatermarkDetector()
    
    # Test with different confidence thresholds
    confidence_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    print(f"\n" + "="*50)
    print("TESTING DETECTION AT DIFFERENT CONFIDENCE LEVELS")
    print("="*50)
    
    for conf in confidence_levels:
        print(f"\n🔍 Testing confidence threshold: {conf}")
        
        detections = detector.detect(image, confidence_threshold=conf, padding=30)
        
        print(f"   Found {len(detections)} detections")
        
        if detections:
            for i, det in enumerate(detections):
                bbox = det['bbox']
                confidence = det['confidence']
                x1, y1, x2, y2 = bbox
                w, h = x2-x1, y2-y1
                print(f"   Detection {i+1}: bbox=({x1},{y1},{x2},{y2}) size={w}x{h} conf={confidence:.3f}")
        
        # Save visualization for this confidence level
        if detections:
            vis = image.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                confidence = det['confidence']
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(vis, f"{confidence:.2%}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            output_path = f'sora_detection_conf_{conf:.2f}.jpg'
            cv2.imwrite(output_path, vis)
            print(f"   💾 Saved visualization: {output_path}")
            
            # Create mask for this confidence
            mask = detector.create_mask(image, detections, expand_ratio=0.1, feather_pixels=21)
            mask_path = f'sora_mask_conf_{conf:.2f}.png'
            cv2.imwrite(mask_path, mask)
            print(f"   💾 Saved mask: {mask_path}")
    
    print(f"\n" + "="*50)
    print("DETECTION TEST COMPLETE")
    print("="*50)

if __name__ == "__main__":
    test_sora_detection()