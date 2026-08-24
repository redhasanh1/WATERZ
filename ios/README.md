# MarkRemoverAI for iOS

Native SwiftUI client for markremoverai.com. Video only — pick a clip, tap what
should disappear, and the existing Salad GPU workers do the render.

It is a thin client: no model runs on the phone, and no secret ever ships in the
binary. Everything goes through the same public API the website uses.

## Requirements

- Xcode 16 or newer (the project uses file-system-synchronized groups, so new
  files under `MarkRemoverAI/` are picked up without touching the project file)
- iOS 17.0+
- Team `KD3JCFB7N4`, bundle id `com.markremoverai.app`

## Build

```bash
open ios/MarkRemoverAI.xcodeproj
```

Or from the command line:

```bash
xcodebuild -project ios/MarkRemoverAI.xcodeproj -scheme MarkRemoverAI \
  -destination 'platform=iOS Simulator,name=iPhone 17' build
```

## The flow it implements

This mirrors `web/index.html` step for step, against `server_production.py`:

| Step | Call | Notes |
|---|---|---|
| 1 | `POST /api/get-upload-url` | returns a B2 ticket and the `task_id` |
| 2 | `POST` to B2 `upload_url` | direct upload, so Railway sees no ingress; `X-Bz-Content-Sha1: do_not_verify` |
| 3 | `POST /api/upload-complete` | registers the CDN url in Redis, kicks off sprite + HEVC transcode |
| 4 | `POST /api/sam2/select-object` | optional live mask preview; needs the interactive SAM2 worker |
| 5 | `POST /api/sam2/process-video` | enqueues `wsl_sam2` → `propainter`; costs 1 credit |
| 6 | `GET /api/sam2/status/<job_id>` | polled every 3s until `completed` |

Auth is the Flask session cookie, same as the browser. `URLSession` keeps it in
the shared cookie jar, so a login survives relaunches with no token handling.

Point coordinates are sent in **video pixel space**, with `label: 1` for the
object to erase and `label: 0` for something to protect. `VideoFrameExtractor`
applies the track's preferred transform first, so portrait iPhone clips report
their upright size — that is what the worker sees after it decodes.

## Switching hosts

Some school and library networks refuse to resolve `markremoverai.com`. The
login screen has a "Can't connect?" option that switches the client to the
Railway host. For a debug run you can also pass it at launch:

```bash
xcrun simctl launch booted com.markremoverai.app \
  -api_base_url "https://user-interface-ui-production.up.railway.app"
```

## Before it can ship to the App Store

Selling credits for digital content inside an iOS app has to go through
StoreKit in-app purchase — routing to Stripe Checkout is a guideline 3.1.1
rejection. This build deliberately ships **no purchase UI**: it shows the
balance and lets people spend credits they already have. Buying stays on the
website until StoreKit products exist.

Still to do: app icon, launch screen art, and a privacy nutrition label
covering the email address and the uploaded video.
