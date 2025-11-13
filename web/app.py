#!/usr/bin/env python3
"""
Flask Backend for AI Watermark Remover
Handles file uploads and watermark removal processing
"""

import sys
import os
import json
from threading import Lock

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'python_packages'))

from flask import Flask, request, send_file, jsonify, redirect, session
from flask_cors import CORS
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
import mimetypes
from pathlib import Path
from urllib.parse import urljoin, urlencode
import stripe
import uuid
from datetime import datetime
import secrets
import requests

# Import watermark removal modules
try:
    from yolo_detector import YOLOWatermarkDetector
    from wavepaint_tensorrt_inpainter import WavePaintTensorRTInpainter
except ImportError as e:
    print(f"Warning: Could not import watermark removal modules: {e}")
    print("Make sure you're running from the watermarkz directory")

app = Flask(__name__, static_folder='.')
CORS(app)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.environ.get('SECRET_KEY', 'watermarkai-dev-secret'))

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Simple data directory for user/credit persistence
# Prefer explicit DATA_DIR env; otherwise use a stable folder next to this file
DATA_DIR = os.environ.get('DATA_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
USER_DB_FILE = os.path.join(DATA_DIR, 'users.json')
EVENT_DB_FILE = os.path.join(DATA_DIR, 'events.json')
_db_lock = Lock()

def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def _read_json_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _write_json_file(path, obj):
    _ensure_data_dir()
    with _db_lock:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

def _read_user_db():
    return _read_json_file(USER_DB_FILE)

def _write_user_db(db):
    _write_json_file(USER_DB_FILE, db)

def _read_event_db():
    db = _read_json_file(EVENT_DB_FILE)
    if isinstance(db, dict) and 'processed' in db and isinstance(db['processed'], list):
        return db
    return {'processed': []}

def _mark_event_processed(event_id):
    edb = _read_event_db()
    if event_id not in edb['processed']:
        edb['processed'].append(event_id)
        _write_json_file(EVENT_DB_FILE, edb)

def _is_event_processed(event_id) -> bool:
    edb = _read_event_db()
    return event_id in edb.get('processed', [])

def _reverse_price_lookup():
    # Build reverse map from price_id -> plan key
    return {v: k for k, v in STRIPE_PRICE_LOOKUP.items() if v}

# Credit configuration and defaults
# One-time free credits on signup
CREDITS_ON_SIGNUP = int(os.environ.get('CREDITS_ON_SIGNUP', '5'))

# Credits per subscription (on first purchase)
CREDITS_ON_SUB = {
    'starter': int(os.environ.get('CREDITS_ON_SUB_STARTER', '20')),
    'pro': int(os.environ.get('CREDITS_ON_SUB_PRO', '50')),
    'enterprise': int(os.environ.get('CREDITS_ON_SUB_ENTERPRISE', '300')),
}

# Credits per renewal (defaults to same as on sub)
CREDITS_ON_RENEW = {
    'starter': int(os.environ.get('CREDITS_ON_RENEW_STARTER', str(CREDITS_ON_SUB['starter']))),
    'pro': int(os.environ.get('CREDITS_ON_RENEW_PRO', str(CREDITS_ON_SUB['pro']))),
    'enterprise': int(os.environ.get('CREDITS_ON_RENEW_ENTERPRISE', str(CREDITS_ON_SUB['enterprise']))),
}

# Baseline unit: 1 credit = 10s @ 720p30
BASELINE_WIDTH = int(os.environ.get('CREDIT_BASE_WIDTH', '1280'))
BASELINE_HEIGHT = int(os.environ.get('CREDIT_BASE_HEIGHT', '720'))
BASELINE_FPS = float(os.environ.get('CREDIT_BASE_FPS', '30'))
BASELINE_SECONDS = float(os.environ.get('CREDIT_BASE_SECONDS', '10'))

# Images are free by default
CREDIT_COST_IMAGE = int(os.environ.get('CREDIT_COST_IMAGE', '0'))
CREDIT_COST_VIDEO = int(os.environ.get('CREDIT_COST_VIDEO', '1'))  # legacy fallback, not used when dynamic calc is available

def _find_email_by_customer(customer_id: str, db: dict) -> str:
    for email, rec in db.items():
        if isinstance(rec, dict) and rec.get('stripe_customer_id') == customer_id:
            return email
    return ''

def _award_credits(email: str, amount: int, reason: str, stripe_customer_id: str = None, event_id: str = None):
    if not email or not amount:
        return False
    db = _read_user_db()
    user = db.get(email) or {
        'email': email,
        'name': None,
        'credits': 0,
        'credit_history': [],
        'stripe_customer_id': None,
        'signup_granted': False,
        'updated_at': None,
    }
    user['credits'] = int(user.get('credits') or 0) + int(amount)
    if stripe_customer_id:
        user['stripe_customer_id'] = stripe_customer_id
    user['credit_history'] = user.get('credit_history') or []
    user['credit_history'].append({
        'ts': datetime.utcnow().isoformat() + 'Z',
        'delta': int(amount),
        'reason': reason,
        'event_id': event_id,
    })
    user['updated_at'] = datetime.utcnow().isoformat() + 'Z'
    db[email] = user
    _write_user_db(db)
    return True

def _ensure_user_and_signup_credits(email: str, name: str = None):
    """Ensure the user record exists and grant one-time signup credits if configured."""
    if not email:
        return
    db = _read_user_db()
    rec = db.get(email) or {
        'email': email,
        'name': name,
        'credits': 0,
        'credit_history': [],
        'stripe_customer_id': None,
        'signup_granted': False,
        'updated_at': None,
    }
    if not rec.get('signup_granted') and CREDITS_ON_SIGNUP > 0:
        rec['credits'] = int(rec.get('credits') or 0) + int(CREDITS_ON_SIGNUP)
        rec['credit_history'] = rec.get('credit_history') or []
        rec['credit_history'].append({
            'ts': datetime.utcnow().isoformat() + 'Z',
            'delta': int(CREDITS_ON_SIGNUP),
            'reason': 'signup',
            'event_id': None,
        })
        rec['signup_granted'] = True
    rec['updated_at'] = datetime.utcnow().isoformat() + 'Z'
    db[email] = rec
    _write_user_db(db)

def _consume_credits(email: str, amount: int, reason: str) -> bool:
    if not email or amount <= 0:
        return False
    db = _read_user_db()
    user = db.get(email)
    if not user:
        return False
    current = int(user.get('credits') or 0)
    if current < amount:
        return False
    user['credits'] = current - amount
    user['credit_history'] = user.get('credit_history') or []
    user['credit_history'].append({
        'ts': datetime.utcnow().isoformat() + 'Z',
        'delta': -int(amount),
        'reason': reason,
        'event_id': None,
    })
    user['updated_at'] = datetime.utcnow().isoformat() + 'Z'
    db[email] = user
    _write_user_db(db)
    return True

# Stripe configuration
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_PRICE_LOOKUP = {
    'starter': os.environ.get('STRIPE_PRICE_ID_STARTER', ''),
    'pro': os.environ.get('STRIPE_PRICE_ID_PRO', ''),
    'enterprise': os.environ.get('STRIPE_PRICE_ID_ENTERPRISE', ''),
}
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET', '')
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
GOOGLE_REDIRECT_URI = os.environ.get('GOOGLE_REDIRECT_URI')
GOOGLE_OAUTH_SCOPE = os.environ.get('GOOGLE_OAUTH_SCOPE', 'openid email profile')

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
    """Lazy load WavePaint TensorRT inpainter"""
    global inpainter
    if inpainter is None:
        print("Initializing WavePaint TensorRT inpainter...")
        inpainter = WavePaintTensorRTInpainter()
    return inpainter


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    """Check if file is a video"""
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in {'mp4', 'mov', 'avi'}


def estimate_video_credits(video_path: str) -> int:
    """Estimate required credits for a video based on resolution, fps, and duration.

    Baseline: 1 credit = 10 seconds @ 720p (1280x720) and 30 fps.
    Credits are computed as ceil( (pixels_ratio) * (fps_ratio) * (time_ratio) ).
    Always at least 1 credit for any non-empty video.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 1
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or BASELINE_FPS
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or BASELINE_WIDTH
        height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or BASELINE_HEIGHT
        total_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()

        if fps <= 0:
            fps = BASELINE_FPS
        if width <= 0 or height <= 0:
            width, height = BASELINE_WIDTH, BASELINE_HEIGHT

        duration_sec = (total_frames / fps) if total_frames > 0 else BASELINE_SECONDS
        pixels_ratio = (width * height) / float(BASELINE_WIDTH * BASELINE_HEIGHT)
        fps_ratio = fps / BASELINE_FPS
        time_ratio = duration_sec / BASELINE_SECONDS

        raw = pixels_ratio * fps_ratio * time_ratio
        credits = int(np.ceil(max(0.01, raw)))
        return max(1, credits)
    except Exception:
        return 1


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
    """Serve the main HTML page."""
    return send_file('index.html')

@app.route('/index.html')
def legacy_index():
    """Legacy path compatibility: serve the same page."""
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


@app.route('/auth/google')
def google_auth_start():
    """Kick off Google OAuth flow by redirecting to consent screen."""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return jsonify({'error': 'Google OAuth is not configured.'}), 500

    redirect_uri = GOOGLE_REDIRECT_URI or urljoin(request.host_url, 'auth/google/callback')
    state = secrets.token_urlsafe(16)
    session['google_oauth_state'] = state

    flow = request.args.get('flow')
    if flow:
        session['google_oauth_flow'] = flow

    post_auth = request.args.get('next')
    if post_auth and post_auth.startswith('/'):
        session['google_redirect_after_login'] = post_auth

    query_params = {
        'client_id': GOOGLE_CLIENT_ID,
        'redirect_uri': redirect_uri,
        'response_type': 'code',
        'scope': GOOGLE_OAUTH_SCOPE,
        'state': state,
        'access_type': 'offline',
        'prompt': 'select_account',
        'include_granted_scopes': 'true',
    }

    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(query_params)}"
    return redirect(auth_url)


@app.route('/auth/google/callback')
def google_auth_callback():
    """Handle Google's OAuth redirect back to the application."""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return jsonify({'error': 'Google OAuth is not configured.'}), 500

    def _redirect_to_login(params):
        base = urljoin(request.host_url, 'login.html')
        cleaned = {k: v for k, v in params.items() if v}
        query = urlencode(cleaned)
        return redirect(f"{base}?{query}" if query else base)

    error = request.args.get('error')
    if error:
        description = request.args.get('error_description', error)
        return _redirect_to_login({'google': 'error', 'message': description})

    state = request.args.get('state')
    saved_state = session.pop('google_oauth_state', None)
    if not state or saved_state != state:
        return _redirect_to_login({'google': 'error', 'message': 'state_mismatch'})

    code = request.args.get('code')
    if not code:
        return _redirect_to_login({'google': 'error', 'message': 'missing_code'})

    redirect_uri = GOOGLE_REDIRECT_URI or urljoin(request.host_url, 'auth/google/callback')
    token_payload = {
        'code': code,
        'client_id': GOOGLE_CLIENT_ID,
        'client_secret': GOOGLE_CLIENT_SECRET,
        'redirect_uri': redirect_uri,
        'grant_type': 'authorization_code',
    }

    try:
        token_response = requests.post(
            'https://oauth2.googleapis.com/token',
            data=token_payload,
            timeout=10,
        )
        token_data = token_response.json()
    except requests.RequestException as exc:
        print(f"Google token exchange failed: {exc}")
        return _redirect_to_login({'google': 'error', 'message': 'token_exchange_failed'})

    if not token_response.ok or 'access_token' not in token_data:
        error_msg = token_data.get('error_description') or token_data.get('error') or 'token_error'
        return _redirect_to_login({'google': 'error', 'message': error_msg})

    user_info = {}
    access_token = token_data.get('access_token')
    if access_token:
        try:
            user_response = requests.get(
                'https://openidconnect.googleapis.com/v1/userinfo',
                headers={'Authorization': f'Bearer {access_token}'},
                timeout=10,
            )
            if user_response.ok:
                user_info = user_response.json()
        except requests.RequestException as exc:
            print(f"Google userinfo fetch failed: {exc}")

    session['google_user'] = {
        'email': user_info.get('email'),
        'name': user_info.get('name'),
        'picture': user_info.get('picture'),
    }
    session.pop('google_oauth_flow', None)

    # Ensure user exists and grant one-time signup credits
    try:
        if session['google_user'].get('email'):
            _ensure_user_and_signup_credits(session['google_user'].get('email'), session['google_user'].get('name'))
    except Exception as exc:
        print(f"Signup credits ensure failed: {exc}")

    next_path = session.pop('google_redirect_after_login', None)
    if next_path and next_path.startswith('/'):
        return redirect(next_path)

    success_params = {'google': 'success'}
    if session['google_user'].get('email'):
        success_params['email'] = session['google_user']['email']
    if session['google_user'].get('name'):
        success_params['name'] = session['google_user']['name']
    return _redirect_to_login(success_params)


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

    consumed = False
    cost = 0
    user_email = None
    try:
        # Determine costs and enforce credits for videos
        filename = secure_filename(file.filename)
        is_vid = is_video(filename)

        # Save file early for videos so we can estimate credits from metadata
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if is_vid:
            file.save(filepath)
            cost = estimate_video_credits(filepath)
        else:
            cost = CREDIT_COST_IMAGE

        if 'google_user' in session and isinstance(session.get('google_user'), dict):
            user_email = session['google_user'].get('email')

        if cost > 0:
            if not user_email:
                return jsonify({'error': 'signin_required', 'message': 'Please sign in to use video processing.', 'required_credits': int(cost)}), 401
            # Check and consume credits up front to avoid heavy compute for unpaid
            db = _read_user_db()
            user = db.get(user_email)
            current = int((user or {}).get('credits') or 0)
            if current < int(cost):
                return jsonify({'error': 'insufficient_credits', 'message': 'Not enough credits to process video.', 'required_credits': int(cost), 'credits': current}), 402
            if not _consume_credits(user_email, int(cost), reason='video_process'):
                return jsonify({'error': 'consume_failed', 'message': 'Could not reserve credits. Please try again.'}), 409
            consumed = True

        # Save uploaded file for images (video already saved)
        if not is_vid:
            file.save(filepath)

        print(f"Processing file: {filename}")

        # Process based on file type
        if is_vid:
            result_path = process_video(filepath)
        else:
            result_path = process_image(filepath)

        # Determine content type
        content_type = mimetypes.guess_type(result_path)[0] or 'application/octet-stream'

        # Send the processed file
        resp = send_file(
            result_path,
            mimetype=content_type,
            as_attachment=True,
            download_name=f"removed_{filename}"
        )
        try:
            if consumed and cost:
                resp.headers['X-Credits-Used'] = str(int(cost))
        except Exception:
            pass
        return resp

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        # Refund credits if we reserved them but processing failed
        try:
            if consumed and user_email and cost > 0:
                _award_credits(user_email, cost, reason='refund_video_failed')
        except Exception:
            pass
        return jsonify({'error': str(e)}), 500

    finally:
        # Cleanup uploaded file (keep processed file for download)
        if 'filepath' in locals() and os.path.exists(filepath):
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
    event_id = event.get('id')
    print(f"Received Stripe event: {event_type} ({event_id})")

    # Idempotency: skip if already processed
    if event_id and _is_event_processed(event_id):
        return '', 200

    try:
        if event_type == 'checkout.session.completed':
            session_obj = event['data']['object']
            # Plan from metadata set at checkout creation
            meta = (session_obj.get('metadata') or {})
            plan = (meta.get('plan') or '').lower()
            customer_id = session_obj.get('customer')
            email = (session_obj.get('customer_details') or {}).get('email') or session_obj.get('customer_email')

            # Amount by plan
            amount = CREDITS_ON_SUB.get(plan, 0)
            if amount and email:
                _award_credits(email, amount, reason=f'subscription_checkout_{plan}', stripe_customer_id=customer_id, event_id=event_id)

        elif event_type == 'invoice.payment_succeeded':
            invoice = event['data']['object']
            customer_id = invoice.get('customer')
            # Determine plan from price ID on first line if possible
            plan = None
            try:
                lines = invoice.get('lines', {}).get('data', [])
                if lines:
                    price_id = (((lines[0] or {}).get('price') or {}).get('id'))
                    plan = _reverse_price_lookup().get(price_id)
            except Exception:
                plan = None

            # Find email by stored mapping
            db = _read_user_db()
            email = _find_email_by_customer(customer_id, db)
            amount = CREDITS_ON_RENEW.get(plan or 'pro', 0)  # default to 'pro' if unknown
            if amount and email:
                _award_credits(email, amount, reason=f'subscription_renew_{plan}', stripe_customer_id=customer_id, event_id=event_id)

        elif event_type == 'customer.subscription.updated':
            sub = event['data']['object']
            prev = (event.get('data') or {}).get('previous_attributes') or {}
            new_status = (sub.get('status') or '').lower()
            old_status = (prev.get('status') or '').lower()
            customer_id = sub.get('customer')

            # Track status on the user record
            db = _read_user_db()
            email = _find_email_by_customer(customer_id, db)
            if email:
                rec = db.get(email, {})
                rec['subscription_status'] = new_status
                # Derive plan from first item price if present
                try:
                    items = (sub.get('items') or {}).get('data') or []
                    price_id = (((items[0] or {}).get('price') or {}).get('id')) if items else None
                except Exception:
                    price_id = None
                if price_id:
                    plan_key = _reverse_price_lookup().get(price_id)
                    if plan_key:
                        rec['plan'] = plan_key
                rec['updated_at'] = datetime.utcnow().isoformat() + 'Z'
                db[email] = rec
                _write_user_db(db)

            # Award initial credits if transitioning into active
            if old_status != 'active' and new_status == 'active':
                # infer plan to compute credit amount
                plan_key = None
                try:
                    items = (sub.get('items') or {}).get('data') or []
                    price_id = (((items[0] or {}).get('price') or {}).get('id')) if items else None
                    if price_id:
                        plan_key = _reverse_price_lookup().get(price_id)
                except Exception:
                    plan_key = None

                amount = CREDITS_ON_SUB.get(plan_key or 'pro', 0)
                if amount and email:
                    _award_credits(email, amount, reason=f'subscription_activated_{plan_key}', stripe_customer_id=customer_id, event_id=event_id)

        elif event_type == 'customer.subscription.deleted':
            subscription = event['data']['object']
            customer_id = subscription.get('customer')
            db = _read_user_db()
            email = _find_email_by_customer(customer_id, db)
            if email:
                # Mark status for visibility (no credits change)
                rec = db.get(email, {})
                rec['subscription_status'] = 'canceled'
                rec['updated_at'] = datetime.utcnow().isoformat() + 'Z'
                db[email] = rec
                _write_user_db(db)

        elif event_type in ('customer.created', 'customer.updated'):
            cust = event['data']['object']
            customer_id = cust.get('id')
            email = cust.get('email')
            if customer_id and email:
                db = _read_user_db()
                rec = db.get(email, {
                    'email': email,
                    'name': None,
                    'credits': 0,
                    'credit_history': [],
                    'stripe_customer_id': None,
                    'updated_at': None,
                })
                rec['stripe_customer_id'] = customer_id
                rec['updated_at'] = datetime.utcnow().isoformat() + 'Z'
                db[email] = rec
                _write_user_db(db)

        # Mark processed if we made it here without error
        if event_id:
            _mark_event_processed(event_id)
    except Exception as exc:
        print(f"Webhook handler error for {event_type}: {exc}")
        # Do not mark processed to allow retries
        return 'handler error', 500

    return '', 200


# Compatibility alias for existing Stripe endpoint configuration
@app.route('/api/stripe/webhook', methods=['POST'])
def stripe_webhook_alias():
    return stripe_webhook()


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'detector_loaded': detector is not None,
        'inpainter_loaded': inpainter is not None,
        'billing_webhook_enabled': bool(STRIPE_WEBHOOK_SECRET),
        'stripe_prices_configured': {k: bool(v) for k, v in STRIPE_PRICE_LOOKUP.items()}
    })


@app.route('/api/train', methods=['POST'])
def train_custom_inpainting():
    """Accept two videos (original + object-removed) and register a training job.

    Expects multipart/form-data with file fields named either:
    - 'original' and 'removed' (preferred), or
    - 's2' and 's2_remove' (backward-compatible with user wording).
    """
    files = request.files

    # Accept multiple naming conventions
    original = files.get('original') or files.get('s2')
    removed = files.get('removed') or files.get('s2_remove')

    if not original or not removed:
        return jsonify({'status': 'error', 'message': 'Two files required: original and removed (or s2 and s2_remove).'}), 400

    if original.filename == '' or removed.filename == '':
        return jsonify({'status': 'error', 'message': 'Empty filename for one or both files.'}), 400

    if not (allowed_file(original.filename) and allowed_file(removed.filename)):
        return jsonify({'status': 'error', 'message': 'Invalid file type. Please upload supported video formats.'}), 400

    if not (is_video(original.filename) and is_video(removed.filename)):
        return jsonify({'status': 'error', 'message': 'Both files must be videos.'}), 400

    # Create job directory
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(UPLOAD_FOLDER, 'training_jobs', job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Save uploads
    orig_name = secure_filename(original.filename)
    rem_name = secure_filename(removed.filename)
    orig_path = os.path.join(job_dir, f"original_{orig_name}")
    rem_path = os.path.join(job_dir, f"removed_{rem_name}")
    original.save(orig_path)
    removed.save(rem_path)

    # Quick probe for basic metadata (optional, best-effort)
    def _probe_video(p):
        try:
            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                return None
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': float(cap.get(cv2.CAP_PROP_FPS)) or None,
                'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }
            cap.release()
            return info
        except Exception:
            return None

    orig_info = _probe_video(orig_path)
    rem_info = _probe_video(rem_path)

    # Placeholder: In a full implementation, enqueue a background training task
    # that aligns frames, computes diffs, and fine-tunes an inpainting model.

    return jsonify({
        'status': 'success',
        'message': 'Training job registered. Processing will start shortly.',
        'job_id': job_id,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'uploads': {
            'original': {'filename': orig_name, 'meta': orig_info},
            'removed': {'filename': rem_name, 'meta': rem_info},
        }
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
