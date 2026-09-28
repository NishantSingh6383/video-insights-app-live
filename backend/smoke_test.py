"""Quick smoke test for the API. Run: python smoke_test.py"""
import os

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
os.environ.setdefault('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1,testserver')
django.setup()

import numpy as np
import cv2
import tempfile
from pathlib import Path
from django.test import Client
from django.conf import settings

client = Client()
failures = []


def check(name, condition, detail=""):
    status_str = "PASS" if condition else "FAIL"
    print(f"[{status_str}] {name} {detail}")
    if not condition:
        failures.append(name)


# 1. Health check
r = client.get('/api/health/')
check("health", r.status_code == 200 and r.json()['status'] == 'healthy', str(r.json()))

# 2. Techniques
r = client.get('/api/techniques/')
check("techniques list", r.status_code == 200 and len(r.json()) == 5)

r = client.get('/api/techniques/motion/')
check("technique detail", r.status_code == 200 and r.json()['id'] == 'motion')

r = client.get('/api/techniques/bogus/')
check("technique 404", r.status_code == 404)

# 3. Path traversal attempts are rejected
r = client.get('/api/videos/..%2f..%2fconfig%2fsettings/analytics/')
check("traversal analytics 404", r.status_code == 404, f"status={r.status_code}")

r = client.delete('/api/videos/not-a-uuid/')
check("traversal delete 404", r.status_code == 404, f"status={r.status_code}")

# 4. Upload a tiny synthetic video, summarize it, get analytics, delete it
tmp = Path(tempfile.mkdtemp()) / "test.mp4"
writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), 24, (320, 240))
rng = np.random.default_rng(42)
for i in range(72):  # 3 seconds
    frame = np.full((240, 320, 3), (i * 3) % 255, dtype=np.uint8)
    # moving square for motion signal
    x = (i * 4) % 280
    frame[100:140, x:x + 40] = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    writer.write(frame)
writer.release()

with open(tmp, 'rb') as f:
    r = client.post('/api/videos/upload/', {'video': f})
check("upload", r.status_code == 201, str(r.json())[:200])
file_id = r.json().get('file_id')

if file_id:
    r = client.post(
        f'/api/videos/{file_id}/summarize/',
        data='{"technique": "combined", "output_type": "both", "summary_percent": 20}',
        content_type='application/json',
    )
    body = r.json()
    check("summarize", r.status_code == 200 and body.get('key_frames_selected', 0) > 0, str(body)[:200])
    if r.status_code == 200:
        out = settings.OUTPUT_ROOT / Path(body['output_path']).name
        check("summary file exists", out.exists() and out.stat().st_size > 0)

    r = client.get(f'/api/videos/{file_id}/analytics/?sample_rate=2&max_frames=50')
    check("analytics", r.status_code == 200 and len(r.json().get('frame_scores', [])) > 0)

    r = client.get(f'/api/videos/{file_id}/compare/?sample_rate=2&max_frames=40')
    check("compare", r.status_code == 200 and 'techniques' in r.json())

    # AI insights should return 503 without an API key (or 200 if one is set)
    r = client.post(f'/api/videos/{file_id}/insights/')
    check("insights gated", r.status_code in (200, 503), f"status={r.status_code}")

    r = client.delete(f'/api/videos/{file_id}/')
    check("delete", r.status_code == 200)

print()
if failures:
    print(f"FAILED: {failures}")
    raise SystemExit(1)
print("All smoke tests passed.")