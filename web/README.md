# AI Watermark Remover - Local Web Interface

A modern, AI-powered web application for removing watermarks from images and videos using YOLO detection and LaMa inpainting.

## Features

- 🎯 **AI-Powered Detection**: Automatic watermark detection using YOLOv8
- 🎨 **Smart Removal**: Advanced LaMa inpainting for seamless watermark removal
- 📹 **Video Support**: Process both images (JPG, PNG) and videos (MP4, MOV)
- 🔍 **Before/After Comparison**: Interactive slider to compare results
- 💅 **Modern UI**: Clean, dark-themed interface with purple gradients
- ⚡ **Real-time Progress**: Live processing updates and progress tracking

## Installation

### Prerequisites

1. Python 3.8+
2. YOLO model weights (should be in parent watermarkz directory)
3. LaMa inpainting model (should be configured in parent watermarkz directory)

### Setup

1. Navigate to the web directory:
```bash
cd /workspaces/RoomFinderAI/watermarkz/web
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure the parent directory has the required models:
   - `yolo_detector.py` and YOLOv8 model weights
   - `lama_inpaint_local.py` and LaMa model

## Running the Application

### Start the Flask Backend

```bash
python app.py
```

The server will start on `http://localhost:5000`

### Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

## Billing & Stripe Integration

Stripe Checkout is wired in for the paid plans. Configure the following environment variables before starting the backend:

| Variable | Description |
|----------|-------------|
| `STRIPE_SECRET_KEY` | Your Stripe secret API key (starts with `sk_...`). |
| `STRIPE_PRICE_ID_PRO` | Price ID for the Professional subscription (e.g. `price_123`). |
| `STRIPE_PRICE_ID_ENTERPRISE` | Price ID for the Enterprise subscription. |
| `STRIPE_WEBHOOK_SECRET` | (Optional) Webhook signing secret for the billing webhook endpoint. |

When running locally you can use the Stripe CLI to forward webhooks:

```bash
stripe login
stripe listen --forward-to localhost:5000/api/billing/webhook
```

### Billing Endpoints

- `POST /api/billing/create-checkout-session` – Creates a Stripe Checkout session for a plan. Payload: `{ "plan": "pro" }`.
- `POST /api/billing/create-portal-session` – Creates a Stripe billing portal session. Accepts `{ "customer_id": "cus_..." }` or `{ "session_id": "cs_..." }`.
- `POST /api/billing/webhook` – Receives Stripe webhook events (enabled when `STRIPE_WEBHOOK_SECRET` is provided).

The frontend pricing buttons call these endpoints through `web/js/billing.js`.

## Usage

1. **Upload File**: Drag and drop or click to select an image or video file
2. **Preview**: Review your uploaded file
3. **Process**: Click "Process File" to start watermark removal
4. **Compare**: Use the interactive slider to compare before/after results
5. **Download**: Download your watermark-free file

## API Endpoints

### POST `/api/remove-watermark`
Upload a file for watermark removal.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image or video)

**Response:**
- Returns the processed file with watermarks removed

### GET `/api/health`
Check server health and model status.

**Response:**
```json
{
  "status": "ok",
  "detector_loaded": true,
  "inpainter_loaded": true
}
```

## File Support

### Images
- JPG/JPEG
- PNG
- Max size: 100MB

### Videos
- MP4
- MOV
- AVI
- Max size: 100MB

## Technical Details

### Frontend
- Pure JavaScript (no frameworks)
- Modern CSS with glassmorphism effects
- Responsive design for all screen sizes
- Interactive before/after comparison slider

### Backend
- Flask web framework
- OpenCV for image/video processing
- YOLO for watermark detection
- LaMa for inpainting

### Processing Pipeline
1. File upload and validation
2. YOLO detection of watermark regions
3. Template matching fallback (if available)
4. Temporal consistency for videos
5. LaMa inpainting for seamless removal
6. Return processed file

## Troubleshooting

### Models not loading
Make sure the parent directory contains:
- `yolo_detector.py`
- `lama_inpaint_local.py`
- YOLO model weights (`yolov8n.pt` or similar)
- LaMa model files

### Port already in use
Change the port in `app.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=True)
```

### Large video processing
For videos longer than a few minutes, processing may take time. The progress is logged in the console.

## Notes

- This tool is for educational and fair use purposes only
- Processing time varies based on file size and content
- Video processing is frame-by-frame and may take several minutes
- First request may be slower as models are lazy-loaded

## License

For educational purposes only. Please respect copyright and use responsibly.
