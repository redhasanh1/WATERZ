#!/usr/bin/env python3
"""
Flask Backend for AI Watermark Remover
Handles file uploads and watermark removal processing
"""

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'python_packages'))

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
import mimetypes
from pathlib import Path
from urllib.parse import urljoin
import stripe

# Import watermark removal modules
try:
    from yolo_detector import YOLOWatermarkDetector
    from lama_inpaint_local import LamaInpainter
except ImportError as e:
    print(f"Warning: Could not import watermark removal modules: {e}")
    print("Make sure you're running from the watermarkz directory")

app = Flask(__name__, static_folder='.')
CORS(app)

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Stripe configuration
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_PRICE_LOOKUP = {
    'pro': os.environ.get('STRIPE_PRICE_ID_PRO', ''),
    'enterprise': os.environ.get('STRIPE_PRICE_ID_ENTERPRISE', '')
}
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET', '')

# Initialize detector and inpainter (lazy loading)
detector = None
inpainter = None


def get_detector():
    """Lazy load YOLO detector"""
    global detector
    if detector is None:
        print("Initializing YOLO detector...")
        detector = YOLOWatermarkDetector()
    return detector


def get_inpainter():
    """Lazy load LaMa inpainter"""
    global inpainter
    if inpainter is None:
        print("Initializing LaMa inpainter...")
        inpainter = LamaInpainter()
    return inpainter


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    """Check if file is a video"""
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in {'mp4', 'mov', 'avi'}


def process_image(image_path):
    """Process a single image to remove watermark"""
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError("Failed to read image")

    # Get detector and inpainter
    det = get_detector()
    inp = get_inpainter()

    # Detect watermark
    detections = det.detect(frame, confidence_threshold=0.3, padding=30)

    if not detections:
        print("No watermark detected, returning original image")
        return image_path

    # Create mask
    mask = det.create_mask(frame, detections)

    # Remove watermark
    result = inp.inpaint_region(frame, mask)

    # Save result
    output_path = image_path.replace('.', '_processed.')
    cv2.imwrite(output_path, result)

    return output_path


def process_video(video_path):
    """Process video to remove watermark from all frames"""
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Setup output
    output_path = video_path.replace('.', '_processed.')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError("Failed to create video writer")

    # Get detector and inpainter
    det = get_detector()
    inp = get_inpainter()

    # Load template for fallback
    template_path = os.path.join(os.path.dirname(video_path), '..', 'watermark_template.png')
    template = None
    if os.path.exists(template_path):
        template = cv2.imread(template_path)

    last_valid_bbox = None

    # Process frames
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect watermark
        detections = det.detect(frame, confidence_threshold=0.3, padding=30)

        # Fallback: template matching if YOLO missed
        if not detections and template is not None and last_valid_bbox:
            th, tw = template.shape[:2]
            result_match = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_match)

            if max_val > 0.6:
                x1, y1 = max_loc
                x2, y2 = x1 + tw, y1 + th
                x1 = max(0, x1 - 30)
                y1 = max(0, y1 - 30)
                x2 = min(width, x2 + 30)
                y2 = min(height, y2 + 30)
                detections = [{'bbox': (x1, y1, x2, y2), 'confidence': max_val}]

        # Use last known position as fallback (temporal consistency)
        if not detections and last_valid_bbox:
            detections = [{'bbox': last_valid_bbox, 'confidence': 0.0}]

        if detections:
            # Update last known position
            if detections[0]['confidence'] > 0.3:
                last_valid_bbox = detections[0]['bbox']

            # Create mask and remove watermark
            mask = det.create_mask(frame, detections)
            try:
                processed_frame = inp.inpaint_region(frame, mask)
                out.write(processed_frame)
            except Exception as e:
                print(f"Frame {frame_num} inpainting failed: {e}")
                out.write(frame)
        else:
            out.write(frame)

        frame_num += 1
        if frame_num % 30 == 0:
            print(f"Processed {frame_num}/{total_frames} frames ({int(frame_num/total_frames*100)}%)")

    # Cleanup
    cap.release()
    out.release()

    print(f"Video processing complete: {output_path}")
    return output_path


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_file('index.html')


@app.route('/success.html')
def success_page():
    return send_file('success.html')


@app.route('/cancel.html')
def cancel_page():
    return send_file('cancel.html')


@app.route('/premium.html')
def premium_page():
    return send_file('premium.html')


@app.route('/login.html')
def login_page():
    return send_file('login.html')


@app.route('/terms.html')
def terms_page():
    return send_file('terms.html')


@app.route('/privacy.html')
def privacy_page():
    return send_file('privacy.html')


@app.route('/css/<path:path>')
def serve_css(path):
    """Serve CSS files"""
    return send_file(f'css/{path}')


@app.route('/js/<path:path>')
def serve_js(path):
    """Serve JavaScript files"""
    return send_file(f'js/{path}')


@app.route('/api/remove-watermark', methods=['POST'])
def remove_watermark():
    """API endpoint to process uploaded file"""
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"Processing file: {filename}")

        # Process based on file type
        if is_video(filename):
            result_path = process_video(filepath)
        else:
            result_path = process_image(filepath)

        # Determine content type
        content_type = mimetypes.guess_type(result_path)[0] or 'application/octet-stream'

        # Send the processed file
        return send_file(
            result_path,
            mimetype=content_type,
            as_attachment=True,
            download_name=f"removed_{filename}"
        )

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        # Cleanup uploaded file (keep processed file for download)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass


def _default_url(path: str) -> str:
    """Build an absolute URL for redirect targets."""
    base_url = request.headers.get('Origin') or request.host_url
    return urljoin(base_url.rstrip('/') + '/', path.lstrip('/'))


@app.route('/api/billing/create-checkout-session', methods=['POST'])
def create_checkout_session():
    """Create a Stripe Checkout session for a subscription purchase."""
    if not stripe.api_key:
        return jsonify({'error': 'Stripe is not configured on the server.'}), 503

    data = request.get_json(silent=True) or {}
    plan = data.get('plan', 'pro').lower()
    price_id = STRIPE_PRICE_LOOKUP.get(plan)

    if not price_id:
        return jsonify({'error': f'Unsupported plan "{plan}".'}), 400

    success_url = data.get('success_url') or _default_url('success.html?session_id={CHECKOUT_SESSION_ID}')
    cancel_url = data.get('cancel_url') or _default_url('cancel.html')
    customer_email = data.get('email') or None

    try:
        session = stripe.checkout.Session.create(
            mode='subscription',
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1
            }],
            success_url=success_url,
            cancel_url=cancel_url,
            customer_email=customer_email,
            allow_promotion_codes=True,
            metadata={'plan': plan}
        )
        return jsonify({'url': session.url})
    except stripe.error.StripeError as exc:
        print(f"Stripe error: {exc}")
        return jsonify({'error': str(exc)}), 502
    except Exception as exc:
        print(f"Unexpected error creating checkout session: {exc}")
        return jsonify({'error': 'Unable to create checkout session.'}), 500


@app.route('/api/billing/create-portal-session', methods=['POST'])
def create_portal_session():
    """Create a Stripe billing portal session for existing customers."""
    if not stripe.api_key:
        return jsonify({'error': 'Stripe is not configured on the server.'}), 503

    data = request.get_json(silent=True) or {}
    customer_id = data.get('customer_id')
    session_id = data.get('session_id')

    if not customer_id and session_id:
        try:
            checkout_session = stripe.checkout.Session.retrieve(session_id)
            customer_id = checkout_session.customer
        except stripe.error.StripeError as exc:
            return jsonify({'error': str(exc)}), 400

    if not customer_id:
        return jsonify({'error': 'customer_id or session_id is required.'}), 400

    try:
        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=_default_url('premium.html#pricing')
        )
        return jsonify({'url': portal_session.url})
    except stripe.error.StripeError as exc:
        return jsonify({'error': str(exc)}), 502


@app.route('/api/billing/webhook', methods=['POST'])
def stripe_webhook():
    """Process Stripe webhook events."""
    if not STRIPE_WEBHOOK_SECRET:
        return '', 200  # Webhooks disabled

    payload = request.data
    sig_header = request.headers.get('Stripe-Signature', '')

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=STRIPE_WEBHOOK_SECRET
        )
    except ValueError as exc:
        print(f"Invalid payload: {exc}")
        return 'Invalid payload', 400
    except stripe.error.SignatureVerificationError as exc:
        print(f"Invalid signature: {exc}")
        return 'Invalid signature', 400

    event_type = event['type']
    print(f"Received Stripe event: {event_type}")

    # Placeholder for future business logic
    if event_type == 'checkout.session.completed':
        session = event['data']['object']
        print(f"Checkout completed for customer {session.get('customer')}")
    elif event_type == 'customer.subscription.deleted':
        subscription = event['data']['object']
        print(f"Subscription canceled: {subscription.get('id')}")

    return '', 200


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'detector_loaded': detector is not None,
        'inpainter_loaded': inpainter is not None
    })


if __name__ == '__main__':
    print("=" * 60)
    print("AI Watermark Remover - Flask Backend")
    print("=" * 60)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Max file size: {MAX_FILE_SIZE / 1024 / 1024}MB")
    print("=" * 60)

    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
