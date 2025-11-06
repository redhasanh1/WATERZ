"""
Direct watermark removal test - no Celery, just pure processing
"""
import os
import sys
import time
os.chdir(os.path.join(os.path.dirname(__file__), 'faster-propainter-main'))
sys.path.insert(0, os.getcwd())

# Set environment variables
os.environ['FORCE_TRT_RAFT'] = '1'
os.environ['USE_NEUFLOW'] = '0'
os.environ['ENABLE_FLASH_ATTENTION'] = '1'
os.environ['YOLO_REQUIRE_TENSORRT'] = '1'

from watermark import remove_watermark

def test_watermark_removal(video_path, output_path):
    """Test watermark removal with timing"""
    print(f"\n{'='*60}")
    print(f"TESTING WATERMARK REMOVAL")
    print(f"{'='*60}")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        # Run watermark removal
        remove_watermark(
            video_path=video_path,
            output_path=output_path,
            neighbor_length=10,
            raft_iter=10
        )

        elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"SUCCESS! Total time: {elapsed:.2f} seconds")
        print(f"{'='*60}")
        print(f"Output saved to: {output_path}")

        return output_path

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"ERROR after {elapsed:.2f} seconds: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_watermark_direct.py <input_video> [output_video]")
        print("Example: python test_watermark_direct.py test_input.mp4 test_output.mp4")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else "output_direct_test.mp4"

    # Make paths absolute
    if not os.path.isabs(input_video):
        input_video = os.path.join(os.path.dirname(__file__), input_video)
    if not os.path.isabs(output_video):
        output_video = os.path.join(os.path.dirname(__file__), output_video)

    if not os.path.exists(input_video):
        print(f"ERROR: Input video not found: {input_video}")
        sys.exit(1)

    test_watermark_removal(input_video, output_video)
