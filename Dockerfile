FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# yolov8n.pt will be auto-downloaded on first run if not present
EXPOSE 5000

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "web_app:app"]
