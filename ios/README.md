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

## Connecting

**The app talks to Railway directly, not through `markremoverai.com`.** Public
and school Wi-Fi filters routinely refuse to resolve the apex domain — it is a
young, uncategorised domain — and the website being unreachable must not take
the app down with it. So:

- Default host is `user-interface-ui-production.up.railway.app`.
- At launch the app probes every known host **concurrently** and keeps the first
  that answers `/api/health`. Racing rather than trying them in order means a
  blackholed DNS lookup can't hold up a host that works.
- If the chosen host dies mid-session, a `cannotFindHost` / `dnsLookupFailed` /
  `cannotConnectToHost` failure triggers one automatic re-probe and the request
  is retried on the winner.
- "Can't connect?" on the login screen still lets you pin one by hand, and
  `-api_base_url <url>` overrides it at launch (now persisted, so it survives
  being reopened from the home screen).

Keep the Railway-generated domain alive on the `USER INTERFACE (UI)` service —
removing it would strip the app of its filter-proof route.

## Console

Every request, sign-in and job step is logged through `AppLog` to the unified
log **and** an in-app ring buffer, because the failures that matter happen on a
real phone on a real network with no Xcode attached. Open it from **Console** on
the login screen or in the account menu: filter by category, show errors only,
share the transcript.

From a Mac:

```bash
xcrun simctl spawn booted log stream --level debug \
  --predicate 'subsystem == "com.markremoverai.app"'
```

`log show` needs `--info --debug` to include anything below error level.

## Sign-in

Three ways in, matching the website:

- **Sign in with Apple** — native, `ASAuthorizationAppleIDProvider`. The app
  posts the identity token to `POST /api/auth/apple`, which validates it against
  Apple's published keys and links or creates the account. Apple only reveals
  the name and address on the *first* authorization, so those are forwarded with
  the token rather than looked up later.
- **Google** — no SDK. `ASWebAuthenticationSession` runs the site's existing
  `/auth/google` flow with `?native=1`. The cookie that web view receives is not
  ours to keep, so the server mints a one-time code (Redis, 5 minutes, deleted on
  first use) and redirects to `markremoverai://auth?code=…`. The app trades it at
  `POST /api/auth/exchange` for a real session. **No new Google OAuth client is
  needed** — this reuses the web credentials already configured.

  One wrinkle: `GOOGLE_REDIRECT_URI` on Railway pins the callback to
  `https://markremoverai.com/auth/google/callback`, which is precisely the host
  filtered Wi-Fi blocks. For `native=1` the server now ignores that pin and
  sends the callback back to whichever host the app reached, so the whole flow
  stays on Railway.
- **Email + password** — the existing `/api/auth/login`.

Sign in with Apple is not optional: offering Google without it is an App Store
rejection under guideline 4.8.

## In-app purchase

Apple requires digital goods to be sold through StoreKit, so Stripe Checkout is
not an option inside the app — that is a 3.1.1 rejection. Credits are sold as
**consumables** priced to match the web packs:

| Product ID | Credits | Price |
|---|---|---|
| `com.markremoverai.app.credits5` | 5 | $2.99 |
| `com.markremoverai.app.credits15` | 15 | $6.99 |
| `com.markremoverai.app.credits60` | 60 | $24.99 |

The flow is StoreKit 2: buy → post `Transaction.jwsRepresentation` to
`POST /api/billing/apple/redeem` → the server verifies the signature against
Apple's root certificates (`certs/apple/`), banks the credits on the same
`users.credits` column Stripe writes to, and only then does the app call
`transaction.finish()`.

Two deliberate choices there:

- **Fail closed.** If the verifier can't load — library missing, certificates
  absent — the endpoint refuses to grant anything rather than trusting a string
  from the internet. It says so plainly to the customer.
- **Unfinished on failure.** If redeeming fails, the transaction is left
  unfinished so StoreKit replays it, instead of someone paying for credits that
  never arrive. `Transaction.unfinished` is drained at launch and by "Restore
  purchases".

Idempotency is the `apple_purchases.transaction_id` unique index — a replayed
receipt collides instead of paying out twice.

`Products.storekit` drives the simulator. It is wired into the scheme, so the
packs appear when you **Run from Xcode** (⌘R); `simctl launch` alone does not
start a StoreKit test session, so products will read as unavailable there.

## Still to do before submitting

1. **Deploy the server half.** `/api/auth/apple`, `/api/auth/exchange` and
   `/api/billing/apple/redeem` live on this branch only. Railway deploys
   `finalbranch`, so they must be merged there to go live, along with the two
   new requirements (`PyJWT[crypto]`, `app-store-server-library`) and the
   `certs/apple/` directory.
2. **App Store Connect** — create the three consumable products with exactly the
   IDs above.
3. **Apple Developer portal** — enable the Sign in with Apple capability on the
   `com.markremoverai.app` App ID.
4. **Google Cloud Console** — add
   `https://user-interface-ui-production.up.railway.app/auth/google/callback`
   to the OAuth client's authorised redirect URIs. Without it Google refuses the
   native flow with `redirect_uri_mismatch`.
5. App icon, launch screen art, and a privacy nutrition label covering the email
   address and the uploaded video.
