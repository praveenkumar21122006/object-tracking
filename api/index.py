# Vercel serverless wrapper for Flask app
# Note: Heavy deps (torch ~2GB, ultralytics, opencv) exceed Vercel's 250MB serverless limit.
# This wrapper serves the UI and API; for full YOLO inference use Docker/Render deploy (see Dockerfile/Procfile).
# Vercel will auto-install from requirements.txt; if limit exceeded, build will fail with "exceeds 250MB".
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from web_app import app  # Flask app at web_app.py:1

# Vercel expects `app` variable
# For local compatibility, no additional code needed
