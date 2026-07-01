# E.D.I.T.H. — HUD Server with FULL Real-Time Metrics
import base64
import os, sys, time, threading, re, subprocess, pickle
from pathlib import Path
from datetime import datetime
from flask import Flask, Response, jsonify, request
from flask_socketio import SocketIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import HUD_HOST, HUD_PORT, USER_CITY, USER_STATE, USER_COUNTRY, USER_LAT, USER_LON
from utils.logger import get_logger
log = get_logger("hud")

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config["SECRET_KEY"] = "edith-v9"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
    ping_timeout=20,        # kill dead connections after 20s (was 60s default)
    ping_interval=10,       # heartbeat every 10s
    max_http_buffer_size=1_000_000)  # 1MB max message — prevents memory bloat

@app.errorhandler(Exception)
def _handle_any_error(e):
    """Catch-all so NO route can ever return a bare, bodyless 500 again.
    Always logs the full traceback to edith.log/console and always returns
    a JSON body with the real exception, so the actual cause is visible
    instead of a generic 'INTERNAL SERVER ERROR'."""
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    log.error(f'Unhandled exception on {request.path}: {e}', exc_info=True)
    return jsonify({'error': str(e), 'type': type(e).__name__, 'path': request.path}), 500

_queue       = []
_prev_net    = None
_live_coords = {"lat": None, "lon": None}   # set by browser GPS
_face_auth   = {"loaded": False, "error": None}

FACE_RECO_ROOT         = Path(r"D:\facerego\output")
FACE_MODEL_PATH        = FACE_RECO_ROOT / "best_model.pth"
FACE_LABELS_PATH       = FACE_RECO_ROOT / "label_encoder.pkl"
FACE_ALLOWED_LABEL     = "kamalesh"
FACE_ALLOWED_THRESHOLD = 0.80
IMAGENET_MEAN          = (0.485, 0.456, 0.406)
IMAGENET_STD           = (0.229, 0.224, 0.225)


def _build_face_classifier(num_classes):
    import torch.nn as nn
    from torchvision import models

    class MobileNetV2FaceClassifier(nn.Module):
        def __init__(self, classes):
            super().__init__()
            backbone = models.mobilenet_v2(weights=None)
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(1280, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x).flatten(1)
            return self.head(x)

    return MobileNetV2FaceClassifier(num_classes)


def _ensure_face_auth():
    global _face_auth
    if _face_auth.get("loaded") or _face_auth.get("error"):
        return _face_auth

    try:
        import cv2
        import torch
        from torchvision.transforms import v2

        if not FACE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model: {FACE_MODEL_PATH}")
        if not FACE_LABELS_PATH.exists():
            raise FileNotFoundError(f"Missing labels: {FACE_LABELS_PATH}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(FACE_MODEL_PATH, map_location=device)
        with open(FACE_LABELS_PATH, "rb") as f:
            metadata = pickle.load(f)

        class_names = metadata["class_names"]
        img_size = int(metadata["img_size"])
        model = _build_face_classifier(len(class_names)).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((img_size, img_size)),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            raise RuntimeError("Failed to load OpenCV Haar cascade.")

        _face_auth.update({
            "loaded": True,
            "device": device,
            "model": model,
            "transform": transform,
            "class_names": class_names,
            "cascade": cascade,
        })
        log.info("Face authentication model loaded successfully")
    except Exception as e:
        _face_auth["error"] = str(e)
        log.error(f"Face authentication unavailable: {e}")

    return _face_auth


def _predict_face_auth(frame_bgr):
    state = _ensure_face_auth()
    if state.get("error"):
        return {
            "ok": False,
            "allowed": False,
            "reason": "model_error",
            "message": state["error"],
        }

    import cv2
    import numpy as np
    import torch

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = state["cascade"].detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        return {
            "ok": False,
            "allowed": False,
            "reason": "no_face",
            "message": "No face detected. Center your face in the camera.",
        }

    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    pad = int(0.20 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_bgr.shape[1], x + w + pad)
    y2 = min(frame_bgr.shape[0], y + h + pad)
    face_bgr = frame_bgr[y1:y2, x1:x2]
    if face_bgr.size == 0:
        return {
            "ok": False,
            "allowed": False,
            "reason": "bad_crop",
            "message": "Detected face could not be processed.",
        }

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = state["transform"](face_rgb).unsqueeze(0).to(state["device"])
    with torch.no_grad():
        probs = torch.softmax(state["model"](tensor), dim=1)[0].cpu().numpy()

    best_idx = int(np.argmax(probs))
    best_label = state["class_names"][best_idx]
    confidence = float(probs[best_idx])
    known = confidence >= FACE_ALLOWED_THRESHOLD
    allowed = known and best_label.lower() == FACE_ALLOWED_LABEL

    if allowed:
        message = "Authorized identity confirmed: kamalesh"
    elif known:
        message = f"Access denied. Detected {best_label}."
    else:
        message = "Face detected but confidence is too low."

    return {
        "ok": True,
        "allowed": allowed,
        "known": known,
        "label": best_label,
        "confidence": confidence,
        "reason": "authorized" if allowed else "unauthorized",
        "message": message,
    }

def _reverse_geocode(lat, lon):
    """Turn lat/lon into city name using free Nominatim API."""
    try:
        import requests
        session = requests.Session()
        session.trust_env = False
        url = (f"https://nominatim.openstreetmap.org/reverse"
               f"?lat={lat}&lon={lon}&format=json&zoom=10")
        r = session.get(url, timeout=6,
                         headers={"User-Agent": "EDITH-HUD/9.4 contact@edith.local"})
        if r.status_code == 200:
            d   = r.json()
            addr = d.get("address", {})
            city    = (addr.get("city") or addr.get("town") or
                       addr.get("village") or addr.get("county") or "")
            state   = addr.get("state", "")
            country = addr.get("country", "")
            return {"city": city, "state": state, "country": country,
                    "lat": lat, "lon": lon}
    except Exception as e:
        log.warning(f"Reverse geocode failed: {e}")
    return {}

def _get_current_location():
    """Return best available location: browser GPS > fallback defaults."""
    lat = _live_coords.get("lat")
    lon = _live_coords.get("lon")
    if lat and lon:
        geo = _reverse_geocode(lat, lon)
        if geo:
            return geo
        return {"city": "", "state": "", "country": "",
                "lat": lat, "lon": lon}
    return {}   # no coords yet — caller should handle

@app.route("/")
def index():
    path = os.path.join(os.path.dirname(__file__), "templates", "hud.html")
    with open(path, "r", encoding="utf-8") as f:
        return Response(f.read(), mimetype="text/html")


@app.post("/api/face-auth")
def face_auth():
    try:
        import cv2
        import numpy as np

        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image", "")
        if not image_data:
            return jsonify({
                "ok": False,
                "allowed": False,
                "reason": "missing_image",
                "message": "No webcam frame received.",
            }), 400

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        frame_bytes = base64.b64decode(image_data)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({
                "ok": False,
                "allowed": False,
                "reason": "decode_error",
                "message": "Could not decode webcam frame.",
            }), 400

        result = _predict_face_auth(frame)
        status = 200 if result.get("ok") or result.get("reason") in {"no_face", "unauthorized"} else 503
        return jsonify(result), status
    except Exception as e:
        log.error(f"Face auth request failed: {e}")
        return jsonify({
            "ok": False,
            "allowed": False,
            "reason": "server_error",
            "message": str(e),
        }), 500


@app.route('/api/model')
def api_model():
    """Return cloud task-router status for the requested task type (default 'agent')."""
    try:
        from brain.ai_engine import get_active_model
        task_type = request.args.get('task_type', 'agent')
        return jsonify(get_active_model(task_type))
    except Exception as e:
        log.error(f"Model info request failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/model/usage')
def api_model_usage():
    """Return per-model usage bars + which model was most recently live + why any cooling-down model failed."""
    try:
        from brain.cloud_router import get_model_usage, get_live_model, get_failure_log
        usage = get_model_usage()
        # Attach the most recent failure reason for any model currently on cooldown
        log_entries = get_failure_log(limit=200)
        last_reason = {}
        for ts, task_type, provider, reason in log_entries:
            last_reason[provider] = reason  # log is chronological, so last write wins = most recent
        for label, stats in usage.items():
            if stats.get("on_cooldown"):
                stats["last_failure_reason"] = last_reason.get(label, "unknown")
        return jsonify({
            "usage": usage,
            "live": get_live_model(),
        })
    except Exception as e:
        log.error(f"Model usage request failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/vision', methods=['POST'])
def api_vision():
    """
    Real image analysis — Gemini multimodal (image bytes + text prompt),
    not just filename/metadata. Used by the 📁 attach-file chat feature
    when the attached file is an image.
    """
    try:
        data = request.get_json(force=True) or {}
        image_b64 = data.get('image', '')
        mime_type = data.get('mime', 'image/png')
        prompt    = data.get('prompt') or 'Describe and analyze this image in detail, Sir.'

        # Strip a data: URL prefix if the frontend sent the raw FileReader result
        if image_b64.startswith('data:'):
            image_b64 = image_b64.split(',', 1)[-1]

        if not image_b64:
            return jsonify({'error': 'no image data provided'}), 400

        from brain.ai_engine import query_vision
        response = query_vision(prompt, image_b64, mime_type)
        return jsonify({'response': response})
    except Exception as e:
        log.error(f'/api/vision 500: {e}', exc_info=True)
        return jsonify({'error': str(e), 'response': f"⚠️ Image analysis failed, Sir: {e}"}), 500


@socketio.on("connect")
def on_connect():
    log.info("HUD client connected")
    # Auto-detect real location and push to client
    try:
        from plugins.all_phases import _detect_location
        loc = _detect_location()
        if loc:
            socketio.emit("location", {
                "city":    loc.get("city", USER_CITY),
                "state":   loc.get("region", USER_STATE),
                "country": loc.get("country", USER_COUNTRY),
                "lat":     str(loc.get("lat", USER_LAT)),
                "lon":     str(loc.get("lon", USER_LON)),
                "display": f"{loc.get('city', USER_CITY)}, {loc.get('region', USER_STATE)}, {loc.get('country', USER_COUNTRY)}",
                "source":  "ip"
            })
        else:
            raise ValueError("detection failed")
    except Exception:
        # Fallback to .env values
        socketio.emit("location", {
            "city": USER_CITY, "state": USER_STATE, "country": USER_COUNTRY,
            "lat": USER_LAT, "lon": USER_LON,
            "display": f"{USER_CITY}, {USER_STATE}, {USER_COUNTRY}",
            "source": "fallback"
        })

@socketio.on("set_location")
def on_set_location(data):
    """Browser sends GPS coordinates here."""
    global _live_coords
    lat = data.get("lat")
    lon = data.get("lon")
    if lat and lon:
        _live_coords = {"lat": str(lat), "lon": str(lon)}
        log.info(f"Browser GPS location set: {lat}, {lon}")
        # Reverse geocode and push display name back to client
        geo = _reverse_geocode(lat, lon)
        if geo:
            socketio.emit("location", {
                "city":    geo.get("city", ""),
                "state":   geo.get("state", ""),
                "country": geo.get("country", ""),
                "lat":     str(lat),
                "lon":     str(lon),
                "display": f"{geo.get('city','')}, {geo.get('state','')}, {geo.get('country','')}",
                "source":  "gps"
            })
    else:
        log.info("Browser geolocation denied — will use wttr.in auto-detection")

@socketio.on("get_weather")
def on_get_weather(data=None):
    """Client requested weather — use stored GPS coords if available."""
    try:
        from plugins.all_phases import get_weather
        lat = _live_coords.get("lat", "")
        lon = _live_coords.get("lon", "")
        result = get_weather("", lat=lat, lon=lon)
        socketio.emit("weather_data", {"weather": result})
    except Exception as e:
        socketio.emit("weather_data", {"weather": f"Weather error: {e}"})

@socketio.on("disconnect")
def on_disconnect():
    log.info("HUD client disconnected")

@socketio.on("command")
def on_command(data):
    cmd = data.get("command", "").strip()
    if cmd:
        log.info(f"CMD: {cmd}")
        _queue.append(cmd)

# Cache for slow/expensive metrics — also read by Phase10 bg agent
_metrics_cache: dict = {}

def _get_all_metrics(tick: int = 0):
    import psutil
    global _prev_net, _metrics_cache
    d = {}

    # ── Fast metrics — every tick (5 s) ──────────────────────
    # CPU: non-blocking (interval=None uses last sample, avoids 0.3s sleep)
    d["cpu"]         = round(psutil.cpu_percent(interval=None), 1)
    d["cpu_per_core"]= psutil.cpu_percent(percpu=True, interval=None)

    # RAM
    vm = psutil.virtual_memory()
    d["ram"]         = round(vm.percent, 1)
    d["ram_used"]    = round(vm.used    / 1024**3, 1)
    d["ram_total"]   = round(vm.total   / 1024**3, 1)
    d["ram_free"]    = round(vm.available / 1024**3, 1)

    # Network speed (cheap — just counter diff)
    net_now = psutil.net_io_counters()
    if _prev_net is not None:
        dl = max(0, net_now.bytes_recv - _prev_net.bytes_recv)
        ul = max(0, net_now.bytes_sent - _prev_net.bytes_sent)
        d["net_dl"] = round(dl * 8 / 1_000_000, 2)
        d["net_ul"] = round(ul * 8 / 1_000_000, 2)
    else:
        d["net_dl"] = d["net_ul"] = 0.0
    _prev_net = net_now
    d["net_total_recv"] = round(net_now.bytes_recv / 1024**3, 2)
    d["net_total_sent"] = round(net_now.bytes_sent / 1024**3, 2)

    # Battery (cheap)
    bat = psutil.sensors_battery()
    if bat:
        d["battery"]          = round(bat.percent, 1)
        d["battery_charging"] = bat.power_plugged
        secs = bat.secsleft
        if secs and secs > 0 and not bat.power_plugged:
            h, rem = divmod(secs // 60, 60)
            d["battery_time"] = f"{h}h {rem:02d}m remaining"
        elif bat.power_plugged:
            d["battery_time"] = "Charging"
        else:
            d["battery_time"] = "Calculating..."
    else:
        d["battery"] = 97.0
        d["battery_charging"] = True
        d["battery_time"] = "Desktop"

    # ── Slow metrics — every 6th tick (~30 s) ────────────────
    if tick % 6 == 0:
        # CPU static info (never changes)
        freq = psutil.cpu_freq()
        _metrics_cache["cpu_freq"]    = round(freq.current) if freq else 0
        _metrics_cache["cpu_cores"]   = psutil.cpu_count(logical=False) or 0
        _metrics_cache["cpu_threads"] = psutil.cpu_count(logical=True) or 0

        # Disk usage
        try:
            disk = psutil.disk_usage("C:/")
        except Exception:
            disk = psutil.disk_usage("/")
        _metrics_cache["disk"]       = round(disk.percent, 1)
        _metrics_cache["disk_used"]  = round(disk.used  / 1024**3, 1)
        _metrics_cache["disk_total"] = round(disk.total / 1024**3, 1)
        _metrics_cache["disk_free"]  = round(disk.free  / 1024**3, 1)

        # Disk I/O counters
        try:
            dio = psutil.disk_io_counters()
            _metrics_cache["disk_read_mb"]  = round(dio.read_bytes  / 1024**2, 1)
            _metrics_cache["disk_write_mb"] = round(dio.write_bytes / 1024**2, 1)
        except Exception:
            _metrics_cache.setdefault("disk_read_mb", 0)
            _metrics_cache.setdefault("disk_write_mb", 0)

        # Process count (moderately expensive)
        _metrics_cache["processes"] = len(psutil.pids())

        # Temperature
        try:
            temps = psutil.sensors_temperatures() or {}
            all_t = [r.current for readings in temps.values()
                     for r in readings if r.current and r.current > 0]
            _metrics_cache["temp_cpu"] = round(max(all_t), 1)   if all_t else 45.0
            _metrics_cache["temp_gpu"] = round(sorted(all_t)[-2], 1) if len(all_t) >= 2 else 52.0
            _metrics_cache["temp_arc"] = round(min(all_t), 1)   if all_t else 36.0
        except Exception:
            _metrics_cache.setdefault("temp_cpu", 45.0)
            _metrics_cache.setdefault("temp_gpu", 52.0)
            _metrics_cache.setdefault("temp_arc", 36.0)

        # Uptime
        secs = int(time.time() - psutil.boot_time())
        _metrics_cache["uptime"] = f"{secs//3600}h {(secs%3600)//60:02d}m"

    # ── Ping — every 12th tick (~60 s), run in bg so it never blocks ──
    if tick % 12 == 0:
        def _do_ping():
            try:
                import platform
                if platform.system() == "Windows":
                    cmd_p = ["ping", "-n", "1", "-w", "1000", "8.8.8.8"]
                    pat   = r"Average = (\d+)ms"
                else:
                    cmd_p = ["ping", "-c", "1", "-W", "1", "8.8.8.8"]
                    pat   = r"time[=<]([\d.]+) ?ms"
                res = subprocess.run(cmd_p, capture_output=True, text=True, timeout=3)
                m = re.search(pat, res.stdout)
                _metrics_cache["net_ping"] = round(float(m.group(1))) if m else 0
            except Exception:
                _metrics_cache.setdefault("net_ping", 0)
        import threading as _th
        _th.Thread(target=_do_ping, daemon=True).start()

    # Merge slow-metric cache into result
    d.update(_metrics_cache)

    # Derived
    d.setdefault("cpu_freq", 0); d.setdefault("cpu_cores", 0)
    d.setdefault("disk", 0); d.setdefault("processes", 0)
    d.setdefault("temp_cpu", 45.0); d.setdefault("temp_gpu", 52.0); d.setdefault("temp_arc", 36.0)
    d.setdefault("uptime", "0h 00m"); d.setdefault("net_ping", 0)
    # Uptime
    secs = int(time.time() - psutil.boot_time())
    d["uptime"] = f"{secs//3600}h {(secs%3600)//60:02d}m"
    # Derived
    d["power"]   = d["battery"]
    d["shield"]  = 100.0
    d["overall"] = round(100 - (d["cpu"]*0.3 + d["ram"]*0.4 + d["disk"]*0.3)/3, 1)
    # Location (auto-detected, cached 5 min)
    loc = _get_current_location()
    d["city"]    = loc.get("city",    USER_CITY)
    d["state"]   = loc.get("state",   USER_STATE)
    d["country"] = loc.get("country", USER_COUNTRY)
    d["lat"]     = str(loc.get("lat", USER_LAT))
    d["lon"]     = str(loc.get("lon", USER_LON))
    return d

def _monitor():
    try:
        import psutil
    except:
        log.error("psutil missing — run: pip install psutil")
        return
    log.info("Real-time metrics monitor started")
    # Counters for staggered slow metrics
    _tick = 0
    while True:
        try:
            t0 = time.time()
            m  = _get_all_metrics(_tick)
            m["timestamp"] = datetime.now().isoformat()
            socketio.emit("metrics", m)
            _tick += 1
        except Exception as e:
            log.warning(f"Metrics err: {e}")
        # Poll every 5 s — reduces CPU/RAM load significantly
        time.sleep(max(0, 5.0 - (time.time()-t0)))

def get_command():  return _queue.pop(0) if _queue else None
def send_response(text, command=""):
    import re as _re, json as _json

    # Failsafe: if AI returned [ROUTE:image_search] re-route
    if text and '[ROUTE:image_search]' in text and command:
        try:
            from brain.router import route as _route
            text = _route(command)
        except Exception:
            pass

    # ── Sanitize agent JSON leakage ──────────────────────────
    # If response looks like raw agent JSON (starts with { or ```json),
    # try to extract the "answer" field; fall back to stripping fences.
    if text and text.strip().startswith(('{', '```')):
        clean = _re.sub(r'```[a-z]*\n?', '', text, flags=_re.M)
        clean = _re.sub(r'\n?```', '', clean, flags=_re.M).strip()
        # Try JSON parse
        for candidate in _re.findall(r'\{[\s\S]+?\}', clean + clean[::-1]):
            try:
                obj = _json.loads(candidate)
                if 'answer' in obj and obj['answer']:
                    text = obj['answer']
                    break
            except Exception:
                pass
        else:
            # Remove any remaining JSON blocks, keep prose
            text = _re.sub(r'\{[^{}]*\}', '', clean).strip() or text

    # Strip Ollama meta commentary
    text = _re.sub(r'After performing the action:?', '', text, flags=_re.I).strip()
    text = _re.sub(r'```[a-z]*\n?[\s\S]*?```', '', text, flags=_re.M).strip()

    # ── Safety net: Groq's gpt-oss reasoning-model bug ────────
    # Groq occasionally leaks the model's internal "thinking" straight
    # into the visible reply even with reasoning hidden at the API
    # level (see community.groq.com/t/670), producing text like:
    #   'KamaleshThe user asks "..." According to policy, ... Kamalesh.Kamalesh'
    # If we see Harmony-style channel control tokens, keep only what
    # follows the final channel. This is a last-resort net — the real
    # fix is the reasoning_format="hidden" param set in brain/ai_engine.py
    # and brain/cloud_router.py.
    if text and ('<|channel|>' in text or '<|start|>' in text or 'assistantfinal' in text):
        for _marker in ("<|start|>assistant<|channel|>final<|message|>",
                         "<|channel|>final<|message|>", "assistantfinal"):
            _idx = text.rfind(_marker)
            if _idx != -1:
                text = text[_idx + len(_marker):]
        text = _re.sub(r'<\|[a-z_]+\|>', '', text).strip()

    socketio.emit("response", {"response":text,"command":command,"timestamp":datetime.now().isoformat()})

def send_project_notify(text, command=""):
    """Send project build messages on a SEPARATE event so they never appear in CHAT tab."""
    # Strip [PROJECT] prefix — the frontend adds its own formatting
    clean = text.replace("[PROJECT]", "").strip()
    socketio.emit("project_update", {
        "text":      text,       # full text with [PROJECT] prefix (for PROJECT panel)
        "message":   clean,      # clean version
        "timestamp": datetime.now().isoformat(),
    })

def send_code_stream(filename: str, content: str, file_num: int = 0, total: int = 0):
    """Stream file content to HUD in chunks — creates the live typewriter effect."""
    import time
    # Send header first
    socketio.emit("code_stream_start", {
        "filename": filename,
        "file_num": file_num,
        "total":    total,
        "timestamp": datetime.now().isoformat(),
    })
    # Strip markdown fences before streaming (AI sometimes wraps code in ```)
    import re as _re
    content = _re.sub(r'^[ \t]*```[\w]*[ \t]*\n?', '', content, flags=_re.MULTILINE)
    content = _re.sub(r'\n?[ \t]*```[ \t]*$', '', content, flags=_re.MULTILINE).strip()

    # Stream content in lines (feels natural, like Claude typing)
    lines = content.split("\n")
    chunk = []
    for i, line in enumerate(lines):
        chunk.append(line)
        if len(chunk) >= 3 or i == len(lines) - 1:
            socketio.emit("code_stream_chunk", {
                "filename": filename,
                "lines":    chunk,
                "done":     i == len(lines) - 1,
            })
            chunk = []
            time.sleep(0.04)
    socketio.emit("code_stream_end", {"filename": filename, "lines": len(lines)})


@app.route('/api/health')
def api_health():
    """Self-healing agent status — used by HUD health indicator."""
    try:
        from brain.self_healing import get_status
        return jsonify(get_status())
    except Exception as e:
        return jsonify({"error": str(e), "running": False}), 500



@app.route('/api/project', methods=['GET','POST'])
def api_project():
    from flask import request, jsonify
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        desc = data.get('request','').strip()
        if not desc or len(desc) < 10:
            return jsonify({'error': 'Request must be at least 10 characters'}), 400
        from brain.project_agent import create_project_async

        def stream_fn(event: str, data: dict):
            """
            Receives real-time streaming events from project_agent and
            forwards them as socket events to the HUD live code viewer.
            event: 'stream_start' | 'stream_chunk' | 'stream_end'
            """
            try:
                if event == "stream_start":
                    socketio.emit("code_stream_start", {
                        "filename": data.get("file", "unknown"),
                        "file_num": data.get("num", 0),
                        "total":    data.get("total", 0),
                        "lang":     data.get("lang", "text"),
                    })
                elif event == "stream_chunk":
                    # Split chunk into lines and emit
                    chunk = data.get("chunk", "")
                    lines = chunk.split("\n")
                    socketio.emit("code_stream_chunk", {
                        "filename": data.get("file", "unknown"),
                        "lines":    lines,
                        "done":     False,
                    })
                elif event == "stream_end":
                    socketio.emit("code_stream_end", {
                        "filename": data.get("file", "unknown"),
                        "lines":    data.get("lines", 0),
                    })
            except Exception as _e:
                log.debug(f"stream_fn emit error: {_e}")

        result = create_project_async(desc, notify_fn=send_project_notify, stream_fn=stream_fn)
        return jsonify({'status': 'started', 'message': result})
    return jsonify({'status': 'ready'})

@app.route('/api/project/pause', methods=['POST'])
def api_project_pause():
    from brain.project_agent import toggle_pause, is_paused
    state = toggle_pause()
    return jsonify({'state': state, 'paused': is_paused()})

@app.route('/api/project/status', methods=['GET'])
def api_project_status():
    from brain.project_agent import is_paused, _build_active
    return jsonify({'paused': is_paused(), 'active': _build_active})

# ── Phase state ────────────────────────────────────────────────────
@app.route('/api/project/state', methods=['GET'])
def api_project_state():
    """Return current phase + label so HUD can update the phase indicator."""
    try:
        from brain.project_agent import load_state, get_phase_label
        state = load_state()
        return jsonify({
            'phase':       state.get('phase', 'idle'),
            'phase_label': get_phase_label(),
            'progress':    state.get('progress', 0),
            'request':     state.get('request', ''),
        })
    except Exception as e:
        log.error(f'/api/project/state 500: {e}', exc_info=True)
        return jsonify({
            'phase': 'idle', 'phase_label': 'READY',
            'progress': 0, 'request': '', 'error': str(e),
        }), 500

@app.route('/api/project/reset', methods=['POST'])
def api_project_reset():
    """Reset project state so user can start a new project."""
    from brain.project_agent import reset_state
    reset_state()
    socketio.emit('project_phase', {'phase': 'idle', 'label': 'READY'})
    return jsonify({'status': 'reset'})

# ── Approve phase ──────────────────────────────────────────────────
@app.route('/api/project/approve', methods=['POST'])
def api_project_approve():
    """User tapped Approve — advance the phase."""
    try:
        from brain.project_agent import (
            approve, run_architecture, run_layout,
            create_project_async, load_state, get_phase_label, PHASE_DEV
        )

        def _notify(text):
            try: send_project_notify(text)
            except Exception as _e: log.debug(f'approve _notify: {_e}')

        resp_text, state, next_action = approve(notify_fn=_notify)
        _notify(resp_text)

        try:
            socketio.emit('project_phase', {
                'phase': state.get('phase'),
                'label': get_phase_label(),
            })
        except Exception as _e:
            log.debug(f'approve phase emit: {_e}')

        if next_action == 'architecture':
            def _bg():
                try:
                    resp, st = run_architecture(notify_fn=_notify)
                    socketio.emit('project_phase', {'phase': st['phase'], 'label': get_phase_label()})
                    socketio.emit('project_plan', {'content': resp})
                except Exception as _e:
                    log.error(f'run_architecture bg: {_e}', exc_info=True)
                    _notify(f'❌ Architecture failed: {_e}')
            socketio.start_background_task(_bg)

        elif next_action == 'layout':
            def _bg():
                try:
                    resp, st = run_layout(notify_fn=_notify)
                    socketio.emit('project_phase', {'phase': st['phase'], 'label': get_phase_label()})
                    socketio.emit('project_plan', {'content': resp})
                except Exception as _e:
                    log.error(f'run_layout bg: {_e}', exc_info=True)
                    _notify(f'❌ Layout failed: {_e}')
            socketio.start_background_task(_bg)

        elif next_action == 'development':
            proj_request = state.get('request', '')

            def sfn(event, sdata):
                try:
                    if event == 'stream_start':
                        socketio.emit('code_stream_start', {
                            'filename': sdata.get('file', 'unknown'),
                            'file_num': sdata.get('num', 0),
                            'total':    sdata.get('total', 0),
                            'lang':     sdata.get('lang', 'text'),
                        })
                    elif event == 'stream_chunk':
                        socketio.emit('code_stream_chunk', {
                            'filename': sdata.get('file', 'unknown'),
                            'lines':    sdata.get('chunk', '').split('\n'),
                            'done':     False,
                        })
                    elif event == 'stream_end':
                        socketio.emit('code_stream_end', {
                            'filename': sdata.get('file', 'unknown'),
                            'lines':    sdata.get('lines', 0),
                        })
                except Exception as _e:
                    log.debug(f'approve sfn: {_e}')

            create_project_async(proj_request, notify_fn=_notify, stream_fn=sfn)

        return jsonify({'status': 'ok', 'next': next_action})

    except Exception as e:
        log.error(f'/api/project/approve 500: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


# ── Project chat (plan/debug/questions) ───────────────────────────
@app.route('/api/project/chat', methods=['POST'])
def api_project_chat():
    """
    Handle all PROJECT tab text input — dispatches to:
    - run_requirements() for a new project description
    - project_chat() for questions, change requests, approvals
    """
    try:
        data = request.get_json(silent=True) or {}
        msg  = data.get('message', '').strip()
        if not msg:
            return jsonify({'error': 'Empty message'}), 400

        from brain.project_agent import (
            run_requirements, project_chat, load_state,
            get_phase_label, PHASE_IDLE
        )
        state = load_state()
        phase = state.get('phase', PHASE_IDLE)

        def _emit_phase(ph):
            try:
                socketio.emit('project_phase', {
                    'phase': ph,
                    'label': get_phase_label(),
                })
            except Exception as _e:
                log.debug(f'_emit_phase: {_e}')

        def _emit_plan(content):
            try:
                socketio.emit('project_plan', {'content': content})
            except Exception as _e:
                log.debug(f'_emit_plan: {_e}')

        def _notify(text):
            """Safe notify — never raises, works from background thread."""
            try:
                send_project_notify(text)
            except Exception as _e:
                log.debug(f'_notify: {_e}')

        if phase == PHASE_IDLE:
            # ── New project: run Phase 1 requirements analysis ──────
            def _bg_req():
                try:
                    resp, st = run_requirements(msg, notify_fn=_notify)
                    _emit_phase(st.get('phase', phase))
                    _emit_plan(resp)
                except Exception as _e:
                    log.error(f'run_requirements failed: {_e}', exc_info=True)
                    _notify(f'❌ Requirements analysis failed: {_e}')
            socketio.start_background_task(_bg_req)
            return jsonify({'status': 'started',
                            'message': '💠 Analysing your project...'})

        else:
            # ── In-progress: chat / approve / change request ─────────
            def _bg_chat():
                try:
                    resp, next_action = project_chat(msg, notify_fn=_notify)
                    _emit_phase(load_state().get('phase', phase))
                    _emit_plan(resp)

                    if next_action == 'architecture':
                        from brain.project_agent import run_architecture
                        resp2, st2 = run_architecture(notify_fn=_notify)
                        _emit_phase(st2.get('phase', phase))
                        _emit_plan(resp2)

                    elif next_action == 'layout':
                        from brain.project_agent import run_layout
                        resp2, st2 = run_layout(notify_fn=_notify)
                        _emit_phase(st2.get('phase', phase))
                        _emit_plan(resp2)

                    elif next_action == 'development':
                        from brain.project_agent import create_project_async, load_state as _ls

                        def sfn(event, sdata):
                            try:
                                if event == 'stream_start':
                                    socketio.emit('code_stream_start', {
                                        'filename': sdata.get('file', 'unknown'),
                                        'file_num': sdata.get('num', 0),
                                        'total':    sdata.get('total', 0),
                                        'lang':     sdata.get('lang', 'text'),
                                    })
                                elif event == 'stream_chunk':
                                    socketio.emit('code_stream_chunk', {
                                        'filename': sdata.get('file', 'unknown'),
                                        'lines':    sdata.get('chunk', '').split('\n'),
                                        'done':     False,
                                    })
                                elif event == 'stream_end':
                                    socketio.emit('code_stream_end', {
                                        'filename': sdata.get('file', 'unknown'),
                                        'lines':    sdata.get('lines', 0),
                                    })
                            except Exception as _e:
                                log.debug(f'sfn: {_e}')

                        create_project_async(
                            _ls().get('request', ''),
                            notify_fn=_notify,
                            stream_fn=sfn,
                        )
                except Exception as _e:
                    log.error(f'project_chat bg failed: {_e}', exc_info=True)
                    _notify(f'❌ Error: {_e}')

            socketio.start_background_task(_bg_chat)
            return jsonify({'status': 'started', 'message': '⚙️ Processing...'})

    except Exception as e:
        log.error(f'/api/project/chat 500: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500



@app.route('/favicon.ico')
def favicon():
    return '', 204

def start_hud_server():
    threading.Thread(target=_monitor, daemon=True, name="Metrics").start()
    log.info(f"HUD → http://localhost:{HUD_PORT}")
    socketio.run(app, host=HUD_HOST, port=HUD_PORT, debug=False, allow_unsafe_werkzeug=True)
