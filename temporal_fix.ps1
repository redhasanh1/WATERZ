$file = "D:\watermarkz\celery_3070_worker.py"
$content = Get-Content $file -Raw

# Fix 1: Get total_frames before the copy loops
$oldCode1 = @"
            # Copy just this segment's frames from the cache
            seg_frames_dir = os.path.join(work_dir, 'frames')
            os.makedirs(seg_frames_dir, exist_ok=True)

            for frame_idx in range(start_frame, end_frame + 1):
"@

$newCode1 = @"
            # Copy ALL frames from the cache (for temporal context)
            seg_frames_dir = os.path.join(work_dir, 'frames')
            os.makedirs(seg_frames_dir, exist_ok=True)

            total_frames = len([f for f in os.listdir(all_frames_dir) if f.endswith('.png')])
            print(f"[3070] Copying ALL {total_frames} frames for temporal context...")

            for frame_idx in range(total_frames):
"@

$content = $content.Replace($oldCode1, $newCode1)

# Fix 2: Change destination to keep original numbering (frames)
$oldCode2 = @"
                # Dest: local index within segment
                local_idx = frame_idx - start_frame
                dst = os.path.join(seg_frames_dir, f"{local_idx:04d}.png")
"@

$newCode2 = @"
                # Dest: keep original frame numbering
                dst = os.path.join(seg_frames_dir, f"{frame_idx:04d}.png")
"@

$content = $content.Replace($oldCode2, $newCode2)

# Fix 3: Copy ALL masks
$oldCode3 = @"
            # Copy just this segment's masks from the cache
            seg_masks_dir = os.path.join(work_dir, 'masks')
            os.makedirs(seg_masks_dir, exist_ok=True)

            for frame_idx in range(start_frame, end_frame + 1):
                # Source: original frame index in all_masks
                src = os.path.join(all_masks_dir, f"{frame_idx:04d}.png")
                # Dest: local index within segment
                local_idx = frame_idx - start_frame
                dst = os.path.join(seg_masks_dir, f"{local_idx:04d}.png")
"@

$newCode3 = @"
            # Copy ALL masks from the cache (for temporal context)
            seg_masks_dir = os.path.join(work_dir, 'masks')
            os.makedirs(seg_masks_dir, exist_ok=True)

            for frame_idx in range(total_frames):
                # Source: original frame index in all_masks
                src = os.path.join(all_masks_dir, f"{frame_idx:04d}.png")
                # Dest: keep original frame numbering
                dst = os.path.join(seg_masks_dir, f"{frame_idx:04d}.png")
"@

$content = $content.Replace($oldCode3, $newCode3)

Set-Content $file $content -NoNewline
Write-Host "Done - Applied temporal fix (copy ALL frames, keep original numbering)"
