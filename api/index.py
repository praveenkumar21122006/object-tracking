# Vercel lightweight wrapper - UI only (no torch/YOLO to stay under 500MB)
# Heavy inference (YOLOv8 + torch) runs via Docker/Render: see Dockerfile, Procfile, web_app.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Import full app if available (local/Docker), fallback to lightweight UI on Vercel
try:
    # Try to load heavy dependencies - will fail on Vercel due to size, fallback to light mode
    import importlib.util
    has_torch = importlib.util.find_spec("torch") is not None
    if has_torch:
        from web_app import app as full_app
        app = full_app
        print("[Vercel] Full app loaded with YOLO")
    else:
        raise ImportError("torch not available on Vercel")
except Exception as e:
    print(f"[Vercel] Lightweight mode: {e}")

    HTML_LIGHT = r"""
<!doctype html>
<html><head><meta charset="utf-8"><title>Object Tracking - Vercel</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>*{box-sizing:border-box;font-family:Inter,system-ui}body{margin:0;background:#0f1220;color:#e6e6f0}
header{padding:24px;background:linear-gradient(135deg,#1a1f3d,#0f1220);border-bottom:1px solid #22264a}
.container{max-width:900px;margin:0 auto;padding:24px}
.card{background:#181c36;border:1px solid #242850;border-radius:16px;padding:18px;margin-bottom:16px}
.badge{padding:4px 10px;border-radius:999px;background:#23284d;font-size:12px}
a{color:#7d8bff} code{background:#0a0c1a;padding:2px 6px;border-radius:6px}
</style></head><body>
<header><h1>🎯 Real-Time Object Tracking</h1><p>YOLOv8 + ByteTrack/SORT — Vercel (lightweight) + Docker (full inference)</p></header>
<div class="container">
<div class="card">
<h3>⚠️ Vercel Serverless Limit</h3>
<p>Vercel functions max 500MB — torch (2GB) + ultralytics (5.8GB bundle) exceeds limit.</p>
<p><span class="badge">Build error</span> <code>Total bundle size 5875MB > 500MB</code></p>
<p><strong>Use full inference via:</strong></p>
<ul>
<li>Docker: <code>docker build -t tracking . && docker run -p 5000:5000 tracking</code> → <code>http://localhost:5000</code> (see <code>Dockerfile:1</code>)</li>
<li>Render/Railway: Connect <a href="https://github.com/praveenkumar21122006/object-tracking">GitHub repo</a> → Start <code>gunicorn web_app:app</code> (<code>Procfile:1</code>)</li>
<li>Local: <code>python web_app.py</code> → <code>http://127.0.0.1:5000</code> (currently running)</li>
</ul>
</div>
<div class="card">
<h3>✅ What works on Vercel (this page)</h3>
<ul>
<li>UI shell + API docs + sample videos (without YOLO inference)</li>
<li>Direct link to full demo videos: <a href="https://github.com/praveenkumar21122006/object-tracking">GitHub README</a></li>
</ul>
<p>Full CLI: <code>python app.py --source 0 --tracker bytetrack</code> | <code>python demo.py</code></p>
<p>Browser (full): <code>python web_app.py</code> → Upload / Sample (<code>bus_pan.mp4</code>) → Run → Download</p>
</div>
<div class="card">
<h3>📦 Project Structure</h3>
<pre style="background:#0a0c1a;padding:12px;border-radius:10px;overflow:auto">app.py                    # CLI: capture→detect→track→visualize
web_app.py                # Flask + ByteTrack/SORT + MJPEG /stream
tracker/detector.py       # YOLODetector (torch fallback)
tracker/object_tracker.py # ByteTrack/SORT/Centroid
utils/visualization.py    # trails + HUD
vercel.json + api/index.py # Vercel lightweight wrapper (this file)
Dockerfile + Procfile      # Full deploy (Render/HF)</pre>
<p>GitHub: <a href="https://github.com/praveenkumar21122006/object-tracking">praveenkumar21122006/object-tracking</a> — push ready (see below)</p>
</div>
<div class="card" style="font-size:12px;opacity:0.7">
Vercel Project: <code>praveenkumar21122006s-projects/object-tracking</code> — last deploy failed due to torch size. This lightweight build fixes bundle size.
</div>
</div></body></html>
    """

    @app.route("/")
    def index():
        return render_template_string(HTML_LIGHT)

    @app.route("/api/samples")
    def samples():
        return jsonify({"mode":"lightweight","samples":[{"id":"sample","path":"/tmp/opencode/samples/bus_pan.mp4"}],
                        "note":"Full YOLO inference requires Docker/Render (torch exceeds Vercel 500MB). Use python web_app.py locally.",
                        "vercel_url":"https://object-tracking-pn812cqk0-praveenkumar21122006s-projects.vercel.app",
                        "github":"https://github.com/praveenkumar21122006/object-tracking"})

    @app.route("/api/track_sample", methods=["POST"])
    def track_sample_light():
        return jsonify({"ok":False,"error":"YOLO inference disabled on Vercel (5875MB > 500MB limit). Use Docker: docker build -t tracking . && docker run -p 5000:5000 tracking OR Render with Procfile, OR local: python web_app.py","mode":"lightweight"}), 501

    @app.route("/api/track", methods=["POST"])
    def track_light():
        return jsonify({"ok":False,"error":"Vercel lightweight mode - see / for deploy instructions"}), 501

# Vercel expects `app`
