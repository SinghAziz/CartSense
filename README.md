# CartSense – Feasibility Prototype

Single HTML file. No backend, no build step, no database, no API keys.

## What it does
- Opens your webcam.
- Detects multiple objects at once using TensorFlow.js + COCO-SSD (pretrained model, loaded from CDN).
- Tracks each physical object across frames (simple IOU tracker) so it isn't
  double-counted every frame.
- Maps detected object classes to fake products:
  - bottle → Coca-Cola → ₹40
  - apple → Apple → ₹20
  - book → Notebook → ₹50
  - cup → Coffee → ₹80
  - banana → Banana → ₹30
- Shows a live cart with quantity and total price.
- "Clear Cart" button resets everything.

An object needs to be seen for a few consecutive frames before it's "confirmed"
into the cart (shown as orange box while pending, green once confirmed). If you
remove it and it stays gone for a short while, it drops out of the cart
automatically.

## How to run on MacBook (Chrome)
You cannot just double-click index.html (file:// URLs block the webcam in
Chrome). Run a tiny local server instead — one command, no installation needed
if you have Python (Mac ships with it):

```
cd cartsense
python3 -m http.server 8000
```

Then open Chrome and go to:
```
http://localhost:8000
```
Allow camera access when prompted. localhost is treated as secure, so no
HTTPS/certificates needed.

## How to run later on Android Chrome
Camera access requires a "secure context" — either `localhost` or `https://`.
Two easy options when you're ready:

1. **Same Wi-Fi + ngrok (fastest):**
   ```
   python3 -m http.server 8000
   ngrok http 8000
   ```
   Open the `https://...ngrok-free.app` URL it gives you, in Chrome on your phone.

2. **Deploy free static hosting** (GitHub Pages, Netlify, Vercel) — just
   upload index.html, you get an https:// URL automatically. No backend needed
   since this is a pure static file.

Either way, no code changes are required — the same index.html works on
desktop or mobile Chrome.

## Requirements checklist
- **Internet required?** Yes — only to download TensorFlow.js and the
  COCO-SSD model from the CDN on first load (and Chrome itself). No internet
  is used for the camera feed or detection after the model has loaded; all
  inference runs locally in the browser.
- **HTTPS required?** Not for localhost. Yes for any other host/IP (including
  phone access over Wi-Fi) — browsers block camera access on plain http://
  except for localhost.
- **API key / credentials required?** None.
- **Backend / database?** None — everything runs client-side in the browser tab.

## Known limitations (expected — this is a feasibility test only)
- Detects generic COCO classes (bottle, cup, apple, banana, book, etc.), not
  real product branding/labels. Your team's real model will replace this later.
- Tracking is a simple box-overlap tracker, not re-identification — if an
  object leaves the frame and a very similar one re-enters in a very different
  position quickly, it may be treated as a new item.
- ~5 detections/second, tune-able in index.html (`setTimeout(loop, 200)`).
