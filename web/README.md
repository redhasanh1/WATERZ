# MarkRemoverAI - Local Web Interface

A modern, AI-powered web application for removing unwanted objects from images and videos using YOLO detection and LaMa inpainting.

## Features

- 🎯 **AI-Powered Detection**: Automatic object detection using YOLOv8
- 🎨 **Smart Removal**: Advanced LaMa inpainting for seamless object removal
- 📹 **Video Support**: Process both images (JPG, PNG) and videos (MP4, MOV)
- 🔍 **Before/After Comparison**: Interactive slider to compare results
- 💅 **Modern UI**: Clean, dark-themed interface with purple gradients
- ⚡ **Real-time Progress**: Live processing updates and progress tracking

## Installation

### Prerequisites

1. Python 3.8+
2. YOLO model weights (should be in parent directory)
3. LaMa inpainting model (should be configured in parent directory)

### Setup

1. Navigate to the web directory:
```bash
cd web
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
3. **Process**: Click "Process File" to start object removal
4. **Compare**: Use the interactive slider to compare before/after results
5. **Download**: Download your clean file

## API Endpoints

### POST `/api/remove-object`
Upload a file for object removal.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image or video)

**Response:**
- Returns the processed file with objects removed

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
- YOLO for object detection
- LaMa for inpainting

### Processing Pipeline
1. File upload and validation
2. YOLO detection of object regions
3. Template matching fallback (if available)
4. Temporal consistency for videos
5. LaMa inpainting for seamless removal
6. Return processed file

## Troubleshooting

### Common Issues

1. **Model not loaded**: Ensure YOLO and LaMa models are properly configured in the parent directory
2. **Upload fails**: Check file size (max 100MB) and format (JPG, PNG, MP4, MOV, AVI)
3. **Processing slow**: Video processing is intensive - consider shorter videos or using GPU acceleration
