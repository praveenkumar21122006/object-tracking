#!/usr/bin/env python3
"""
Browser-based Real-Time Object Tracking
- Upload video / use sample / RTSP URL
- Select tracker (bytetrack/sort/centroid), confidence
- Live MJPEG preview + downloadable tracked mp4
"""
import os, time, pathlib, threading, queue, base64
from flask import Flask, request, jsonify, send_from_directory, Response, render_template_string
from flask_cors import CORS
import cv2
import numpy as np

from tracker.detector import YOLODetector
from tracker.object_tracker import build_tracker
from utils.visualization import Visualizer

app = Flask(__name__)
CORS(app)
os.makedirs("output", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Globals for detector (lazy loaded)
detector = None
detector_lock = threading.Lock()

def get_detector(conf=0.35, model="yolov8n.pt"):
    global detector
    with detector_lock:
        if detector is None or abs(detector.conf - conf) > 1e-6 or detector.model.ckpt_path != model:
            print(f"[Web] Loading detector {model} conf={conf}")
            detector = YOLODetector(model_path=model, conf=conf, iou=0.45, classes=None, imgsz=640)
        else:
            detector.conf = conf
    return detector

HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Real-Time Object Tracking</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;font-family:Inter,system-ui,Segoe UI,Roboto,Helvetica,Arial}
body{margin:0;background:#0f1220;color:#e6e6f0}
header{padding:24px 28px;border-bottom:1px solid #22264a;background:linear-gradient(135deg,#1a1f3d,#0f1220)}
header h1{margin:0;font-size:22px}
header p{opacity:0.7;margin:6px 0 0}
.container{max-width:1200px;margin:0 auto;padding:24px;display:grid;grid-template-columns:360px 1fr;gap:24px}
.card{background:#181c36;border:1px solid #242850;border-radius:16px;padding:18px;box-shadow:0 8px 24px rgba(0,0,0,0.3)}
label{font-size:12px;letter-spacing:0.08em;text-transform:uppercase;opacity:0.8;display:block;margin:12px 0 6px}
input,select,button{width:100%;padding:10px 12px;border-radius:10px;border:1px solid #2a2f5a;background:#0f1220;color:#fff;outline:none}
input[type=file]{padding:6px}
button{background:#5b6cff;border:none;font-weight:600;cursor:pointer;margin-top:14px}
button:disabled{opacity:0.5;cursor:not-allowed}
.secondary{background:#23284d}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.pill{display:inline-block;padding:4px 10px;border-radius:999px;background:#23284d;font-size:12px;margin-right:6px;border:1px solid #2a2f5a}
.preview{width:100%;background:#0a0c1a;border-radius:12px;overflow:hidden;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;border:1px solid #242850}
.preview img,.preview video{width:100%;height:100%;object-fit:contain;background:#000}
.log{background:#0a0c1a;border-radius:10px;padding:12px;font-family:monospace;font-size:12px;max-height:180px;overflow:auto;border:1px solid #242850;white-space:pre-wrap}
.badge{padding:2px 8px;border-radius:999px;font-size:11px;border:1px solid #2a2f5a}
.ok{color:#3dd68c;border-color:#1a5a3a;background:#0f2a1f}
.warn{color:#ffcc66}
a{color:#7d8bff}
</style>
</head>
<body>
<header>
  <h1>🎯 Real-Time Object Tracking in Video Streams</h1>
  <p>YOLOv8 + ByteTrack / SORT / Centroid &middot; Upload video, RTSP or sample &rarr; tracked output in browser</p>
</header>

<div class="container">
  <div class="card">
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
      <span class="pill">YOLOv8n (6MB)</span><span class="pill">Kalman Filter</span><span class="pill">Web + API</span>
    </div>

    <label>Tracker</label>
    <select id="tracker">
      <option value="bytetrack">ByteTrack (best)</option>
      <option value="sort">SORT</option>
      <option value="centroid">Centroid</option>
    </select>

    <div class="grid2">
      <div><label>Confidence</label><input id="conf" type="number" min="0.1" max="0.9" step="0.05" value="0.35"></div>
      <div><label>Model</label>
        <select id="model"><option value="yolov8n.pt">yolov8n.pt</option><option value="yolov8s.pt">yolov8s.pt</option></select>
      </div>
    </div>

    <label>Video Source</label>
    <select id="sourceMode" onchange="onMode()">
      <option value="upload">Upload video (mp4/avi/mov)</option>
      <option value="sample">Sample: bus_pan.mp4 (4 objects)</option>
      <option value="demo_traffic">Sample: demo_traffic.mp4</option>
      <option value="url">RTSP / HTTP URL</option>
    </select>

    <div id="uploadBox"><label>Upload</label><input id="file" type="file" accept="video/*"></div>
    <div id="urlBox" style="display:none"><label>Stream URL</label><input id="url" placeholder="rtsp://user:pass@ip/stream or https://..."></div>

    <button id="runBtn" onclick="run()">▶ Run Tracking</button>
    <button class="secondary" onclick="live()">🔴 Live MJPEG Preview</button>
    <div id="status" class="log" style="margin-top:12px">Ready. Select source and click Run.</div>

    <div style="margin-top:12px;font-size:12px;opacity:0.7">
      API: <code>POST /api/track</code> &middot; <code>GET /stream</code> (MJPEG) &middot; <code>GET /api/samples</code>
    </div>
  </div>

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
      <strong>Preview</strong><span id="meta" class="badge">idle</span>
    </div>
    <div class="preview" id="previewBox"><span style="opacity:0.5">No output yet — run tracking</span></div>
    <div style="margin-top:12px;display:flex;gap:10px">
      <a id="dl" style="display:none" href="#" download><button class="secondary" style="width:auto;padding:8px 14px">⬇ Download tracked mp4</button></a>
      <span id="stats" style="font-size:13px;opacity:0.8"></span>
    </div>
    <div style="margin-top:14px">
      <strong style="font-size:13px">How it works</strong>
      <div style="font-size:12px;opacity:0.75;line-height:1.6;margin-top:6px">
        Capture → YOLOv8 detection (COCO 80 classes) → Kalman + Hungarian assignment (ByteTrack/SORT) → trails + FPS overlay → mp4 export.
        Server is <code>web_app.py:1</code>, detector <code>tracker/detector.py:16</code>, tracker <code>tracker/object_tracker.py:52</code>.
      </div>
    </div>
  </div>
</div>

<script>
function onMode(){
  const m=document.getElementById('sourceMode').value;
  document.getElementById('uploadBox').style.display = m==='upload'?'block':'none';
  document.getElementById('urlBox').style.display = m==='url'?'block':'none';
}
async function run(){
  const btn=document.getElementById('runBtn');
  const tracker=document.getElementById('tracker').value;
  const conf=document.getElementById('conf').value;
  const model=document.getElementById('model').value;
  const mode=document.getElementById('sourceMode').value;
  const log=document.getElementById('status');
  const meta=document.getElementById('meta');
  btn.disabled=true; meta.textContent='processing…'; log.textContent='Uploading / processing…\n';
  try{
    let res;
    if(mode==='upload'){
      const f=document.getElementById('file').files[0];
      if(!f){ alert('Choose a video file'); btn.disabled=false; return; }
      const fd=new FormData();
      fd.append('video',f); fd.append('tracker',tracker); fd.append('conf',conf); fd.append('model',model);
      res=await fetch('/api/track',{method:'POST',body:fd});
    } else if(mode==='url'){
      const url=document.getElementById('url').value;
      if(!url){ alert('Enter URL'); btn.disabled=false; return;}
      res=await fetch('/api/track_url',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url, tracker, conf, model})});
    } else {
      const map={sample:'/tmp/opencode/samples/bus_pan.mp4', demo_traffic:'/tmp/opencode/demo_traffic.mp4'};
      res=await fetch('/api/track_sample',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:map[mode], tracker, conf, model})});
    }
    const data=await res.json();
    log.textContent = JSON.stringify(data,null,2);
    if(data.ok){
      meta.textContent=`done • ${data.frames} frames • ${data.fps.toFixed(1)} fps • ${data.detections_avg} dets avg`;
      document.getElementById('previewBox').innerHTML=`<video controls autoplay muted src="${data.video_url}"></video>`;
      const dl=document.getElementById('dl'); dl.href=data.video_url; dl.style.display='inline-block';
      document.getElementById('stats').textContent=`${data.tracker} @ conf ${data.conf} • output: ${data.video_url}`;
    } else {
      meta.textContent='error';
    }
  }catch(e){
    log.textContent='Error: '+e;
    meta.textContent='error';
  }
  btn.disabled=false;
}
function live(){
  document.getElementById('previewBox').innerHTML='<img src=\"/stream?src=sample&tracker=bytetrack&conf=0.35\" style=\"width:100%\">';
  document.getElementById('meta').textContent='live MJPEG • /stream';
  document.getElementById('status').textContent='Live stream: GET /stream?src=sample (loops bus_pan.mp4 with real-time overlay). Browser MJPEG - press Run to generate mp4 instead.';
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/samples")
def samples():
    return jsonify({
        "samples": [
            {"id":"sample","path":"/tmp/opencode/samples/bus_pan.mp4","desc":"bus.jpg pan (4 objects, YOLO verified)"},
            {"id":"demo_traffic","path":"/tmp/opencode/demo_traffic.mp4","desc":"synthetic traffic 90 frames"},
            {"id":"run_output","path":"output/run_output.mp4","desc":"last tracked output"}
        ]
    })

def process_video(source, tracker_type, conf, model, output_path):
    det = get_detector(conf=conf, model=model)
    det.conf = conf
    trk_cfg = dict(tracker_type=tracker_type, track_thresh=conf, track_buffer=30, match_thresh=0.3, min_hits=3)
    if tracker_type=="centroid":
        trk_cfg["track_buffer"]=30
    tracker = build_tracker(trk_cfg)
    viz = Visualizer(draw_trails=True, trail_length=30)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 640
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps<1 or fps>60:
        fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w,h))
    frames=0
    dets_total=0
    t0=time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = det.detect(frame)
        tracks = tracker.update(dets)
        frame = viz.draw(frame, tracks, det, fps=0)
        out.write(frame)
        frames+=1
        dets_total+=len(dets)
        if frames%30==0:
            print(f"[Web] frame {frames} dets {len(dets)} tracks {len(tracks)}")
    cap.release()
    out.release()
    elapsed = time.time()-t0
    return frames, dets_total/max(1,frames), elapsed, fps

@app.route("/api/track", methods=["POST"])
def track_upload():
    tracker = request.form.get("tracker","bytetrack")
    conf = float(request.form.get("conf","0.35"))
    model = request.form.get("model","yolov8n.pt")
    f = request.files.get("video")
    if not f:
        return jsonify({"ok":False,"error":"no video file"}),400
    in_path = f"uploads/{int(time.time())}_{f.filename}"
    f.save(in_path)
    out_path = f"output/tracked_{int(time.time())}.mp4"
    try:
        frames, avg_dets, elapsed, fps = process_video(in_path, tracker, conf, model, out_path)
        return jsonify({"ok":True,"frames":frames,"detections_avg":round(avg_dets,2),"elapsed":round(elapsed,2),"fps":fps,"video_url":f"/video/{os.path.basename(out_path)}","tracker":tracker,"conf":conf})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),500

@app.route("/api/track_sample", methods=["POST"])
def track_sample():
    data = request.get_json(force=True)
    path = data.get("path")
    tracker = data.get("tracker","bytetrack")
    conf = float(data.get("conf",0.35))
    model = data.get("model","yolov8n.pt")
    if not path or not os.path.exists(path):
        return jsonify({"ok":False,"error":f"sample not found: {path}"}),400
    out_path = f"output/tracked_{int(time.time())}.mp4"
    try:
        frames, avg_dets, elapsed, fps = process_video(path, tracker, conf, model, out_path)
        return jsonify({"ok":True,"frames":frames,"detections_avg":round(avg_dets,2),"elapsed":round(elapsed,2),"fps":fps,"video_url":f"/video/{os.path.basename(out_path)}","tracker":tracker,"conf":conf,"source":path})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok":False,"error":str(e)}),500

@app.route("/api/track_url", methods=["POST"])
def track_url():
    data=request.get_json(force=True)
    url=data.get("url")
    tracker=data.get("tracker","bytetrack")
    conf=float(data.get("conf",0.35))
    model=data.get("model","yolov8n.pt")
    out_path=f"output/tracked_{int(time.time())}.mp4"
    try:
        frames, avg_dets, elapsed, fps = process_video(url, tracker, conf, model, out_path)
        return jsonify({"ok":True,"frames":frames,"detections_avg":round(avg_dets,2),"elapsed":round(elapsed,2),"fps":fps,"video_url":f"/video/{os.path.basename(out_path)}","tracker":tracker,"conf":conf})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),500

@app.route("/video/<path:filename>")
def video(filename):
    return send_from_directory("output", filename)

@app.route("/stream")
def stream():
    src = request.args.get("src","sample")
    tracker_type = request.args.get("tracker","bytetrack")
    conf = float(request.args.get("conf","0.35"))
    if src=="sample":
        src_path="/tmp/opencode/samples/bus_pan.mp4"
    else:
        src_path=src

    def gen():
        det = get_detector(conf=conf)
        det.conf=conf
        tracker = build_tracker(dict(tracker_type=tracker_type, track_thresh=conf, track_buffer=30, match_thresh=0.3, min_hits=3))
        viz = Visualizer(draw_trails=True)
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCannot open source\r\n'
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                continue
            dets = det.detect(frame)
            tracks = tracker.update(dets)
            frame = viz.draw(frame, tracks, det, fps=0)
            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY),80])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            time.sleep(0.08)  # ~12 fps
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    print("Browser app at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
