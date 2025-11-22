"""
SAM2 Worker Stress Test
Simulates multiple concurrent users hitting the SAM2 select-object API
"""

import argparse
import base64
import io
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from PIL import Image

# Configuration
DEFAULT_API_URL = "http://localhost:9000/api/sam2/select-object"
DEFAULT_USERS = 10
DEFAULT_REQUESTS_PER_USER = 1  # Each user makes 1 request (realistic scenario)
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480


def create_test_image(width=640, height=480):
    """Create a simple test image with some color variation"""
    # Create a gradient image for testing
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    for y in range(height):
        for x in range(width):
            r = int((x / width) * 255)
            g = int((y / height) * 255)
            b = int(((x + y) / (width + height)) * 255)
            pixels[x, y] = (r, g, b)

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def generate_random_points(width, height, num_points=1):
    """Generate random point coordinates for testing"""
    points = []
    for i in range(num_points):
        points.append({
            "x": random.randint(50, width - 50),
            "y": random.randint(50, height - 50),
            "label": 1 if i == 0 else random.choice([0, 1])  # First point always positive
        })
    return points


def make_request(api_url, frame_data, points, width, height, request_id):
    """Make a single request to the SAM2 API"""
    payload = {
        "frame_data": frame_data,
        "points": points,
        "video_width": width,
        "video_height": height
    }

    start_time = time.time()

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        elapsed_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            return {
                "request_id": request_id,
                "status": "success",
                "response_status": result.get("status", "unknown"),
                "elapsed_ms": elapsed_ms,
                "score": result.get("score", 0),
                "has_mask": "mask" in result
            }
        else:
            return {
                "request_id": request_id,
                "status": "error",
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "elapsed_ms": elapsed_ms
            }

    except requests.exceptions.Timeout:
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            "request_id": request_id,
            "status": "timeout",
            "error": "Request timed out after 30s",
            "elapsed_ms": elapsed_ms
        }
    except requests.exceptions.ConnectionError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            "request_id": request_id,
            "status": "connection_error",
            "error": str(e)[:200],
            "elapsed_ms": elapsed_ms
        }
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            "request_id": request_id,
            "status": "error",
            "error": str(e)[:200],
            "elapsed_ms": elapsed_ms
        }


def user_session(user_id, api_url, frame_data, width, height, num_requests, stagger_ms=0):
    """Simulate a single user making multiple requests"""
    results = []

    # Stagger user arrivals (realistic traffic pattern)
    if stagger_ms > 0:
        time.sleep(random.uniform(0, stagger_ms / 1000))

    for req_num in range(num_requests):
        request_id = f"user{user_id}_req{req_num}"
        points = generate_random_points(width, height, num_points=random.randint(1, 3))

        result = make_request(api_url, frame_data, points, width, height, request_id)
        result["user_id"] = user_id
        results.append(result)

        # Small random delay between requests (50-200ms) to simulate real usage
        if num_requests > 1:
            time.sleep(random.uniform(0.05, 0.2))

    return results


def run_stress_test(api_url, num_users, requests_per_user, image_width, image_height, stagger_ms=0):
    """Run the stress test with concurrent users"""

    print("=" * 60)
    print("SAM2 WORKER STRESS TEST")
    print("=" * 60)
    print(f"Target API: {api_url}")
    print(f"Concurrent users: {num_users}")
    print(f"Requests per user: {requests_per_user}")
    print(f"Total requests: {num_users * requests_per_user}")
    print(f"Test image size: {image_width}x{image_height}")
    if stagger_ms > 0:
        print(f"User stagger: 0-{stagger_ms}ms random delay (realistic)")
    else:
        print(f"User stagger: None (burst mode)")
    print("=" * 60)

    # Generate test image once (all users use same image)
    print("\nGenerating test image...", end=" ")
    frame_data = create_test_image(image_width, image_height)
    print(f"Done ({len(frame_data)} bytes base64)")

    # Run concurrent user sessions
    print(f"\nStarting stress test at {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)

    all_results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = {
            executor.submit(
                user_session,
                user_id,
                api_url,
                frame_data,
                image_width,
                image_height,
                requests_per_user,
                stagger_ms
            ): user_id
            for user_id in range(num_users)
        }

        for future in as_completed(futures):
            user_id = futures[future]
            try:
                user_results = future.result()
                all_results.extend(user_results)

                # Print progress
                successes = sum(1 for r in user_results if r["status"] == "success")
                print(f"  User {user_id}: {successes}/{len(user_results)} successful")

            except Exception as e:
                print(f"  User {user_id}: ERROR - {e}")

    total_time = time.time() - start_time

    # Analyze results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    total_requests = len(all_results)
    successful = [r for r in all_results if r["status"] == "success"]
    failed = [r for r in all_results if r["status"] != "success"]

    print(f"\nTotal requests:     {total_requests}")
    print(f"Successful:         {len(successful)} ({len(successful)/total_requests*100:.1f}%)")
    print(f"Failed:             {len(failed)} ({len(failed)/total_requests*100:.1f}%)")
    print(f"Total test time:    {total_time:.2f}s")
    print(f"Throughput:         {total_requests/total_time:.2f} requests/sec")

    if successful:
        response_times = [r["elapsed_ms"] for r in successful]
        scores = [r.get("score", 0) for r in successful if r.get("score")]

        print("\nResponse Time Statistics (successful requests):")
        print(f"  Min:              {min(response_times):.1f}ms")
        print(f"  Max:              {max(response_times):.1f}ms")
        print(f"  Average:          {statistics.mean(response_times):.1f}ms")
        print(f"  Median:           {statistics.median(response_times):.1f}ms")

        if len(response_times) > 1:
            print(f"  Std Dev:          {statistics.stdev(response_times):.1f}ms")

            # Calculate percentiles
            sorted_times = sorted(response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p99_index = int(len(sorted_times) * 0.99)
            print(f"  P95:              {sorted_times[min(p95_index, len(sorted_times)-1)]:.1f}ms")
            print(f"  P99:              {sorted_times[min(p99_index, len(sorted_times)-1)]:.1f}ms")

        if scores:
            print(f"\nMask Scores:")
            print(f"  Average score:    {statistics.mean(scores):.3f}")
            print(f"  Min score:        {min(scores):.3f}")
            print(f"  Max score:        {max(scores):.3f}")

    if failed:
        print("\nError Breakdown:")
        error_types = {}
        for r in failed:
            error_type = r["status"]
            error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")

        # Show first few error messages
        print("\nSample errors:")
        for r in failed[:3]:
            print(f"  - {r.get('error', 'Unknown error')[:100]}")

    print("\n" + "=" * 60)

    return {
        "total_requests": total_requests,
        "successful": len(successful),
        "failed": len(failed),
        "total_time_seconds": total_time,
        "throughput_rps": total_requests / total_time,
        "avg_response_ms": statistics.mean([r["elapsed_ms"] for r in successful]) if successful else 0
    }


def main():
    parser = argparse.ArgumentParser(description="SAM2 Worker Stress Test")
    parser.add_argument("--url", default=DEFAULT_API_URL,
                        help=f"API endpoint URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--users", type=int, default=DEFAULT_USERS,
                        help=f"Number of concurrent users (default: {DEFAULT_USERS})")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS_PER_USER,
                        help=f"Requests per user (default: {DEFAULT_REQUESTS_PER_USER})")
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH,
                        help=f"Test image width (default: {DEFAULT_IMAGE_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT,
                        help=f"Test image height (default: {DEFAULT_IMAGE_HEIGHT})")
    parser.add_argument("--stagger", type=int, default=0,
                        help="Max random delay in ms before each user starts (default: 0 = burst mode)")

    args = parser.parse_args()

    run_stress_test(
        api_url=args.url,
        num_users=args.users,
        requests_per_user=args.requests,
        image_width=args.width,
        image_height=args.height,
        stagger_ms=args.stagger
    )


if __name__ == "__main__":
    main()
