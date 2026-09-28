# Video Insights

AI-powered video summarization application using multiple computer-vision techniques: Motion Analysis, Color Histograms, Event Detection, Object Detection, and a Combined Approach — with optional Claude-powered AI insights.

## Features

- **Motion-based Summarization**: Uses optical flow / frame differencing to detect movement patterns and selects representative frames per motion segment
- **Color/Histogram-based**: Extracts color histograms and selects visually diverse frames
- **Event-based**: Detects significant activity spikes using motion thresholds
- **Object Detection**: Identifies frames containing significant objects or moving entities
- **Combined Approach**: Integrates motion, quality, and event scoring with configurable weights
- **Output types**: Playable video summary (H.264), static storyboard grid, or both with downloadable frames
- **Analytics dashboard**: Motion heatmap, per-frame importance chart, algorithm statistics, and technique comparison
- **Scene detection**: Finds shot boundaries from colour-histogram jumps, using an adaptive (mean + 3σ) threshold, and marks them on the score chart
- **CSV export**: Download per-frame motion/colour/event/combined scores with timestamps and scene numbers
- **AI Insights (optional)**: Claude interprets your video's analytics and recommends the best technique and summary length
- **Dark mode**: Follows your system preference, toggleable from the header, persisted across sessions

## Tech Stack

- **Backend**: Django + Django REST Framework
- **Frontend**: Django Templates + Vanilla JavaScript + CSS (served by Django — no separate frontend build)
- **Video Processing**: OpenCV, NumPy, scikit-learn, SciPy, imageio-ffmpeg
- **AI (optional)**: Anthropic Claude API

## Project Structure

```
├── backend/
│   ├── config/          # Django settings & configuration
│   ├── api/             # REST API endpoints (+ optional AI insights)
│   ├── services/        # Video summarization algorithms
│   ├── templates/       # HTML templates
│   ├── static/          # CSS & JavaScript
│   └── smoke_test.py    # End-to-end API smoke tests
├── uploads/             # Uploaded video files
└── outputs/             # Generated summaries
```

## Quick Start

### Prerequisites

- Python 3.10+

### Setup & Run

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python manage.py runserver
```

Then open **http://127.0.0.1:8000** in your browser.

### Or use the batch file (Windows)

Double-click `start-backend.bat` (creates the venv and installs dependencies on first run).

### Enable AI Insights (optional)

Set an Anthropic API key before starting the server:

```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-...

# Linux/Mac
export ANTHROPIC_API_KEY=sk-ant-...
```

When configured, an "AI Insights" card appears on the Analytics page. Without a key, the rest of the app works normally.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health/` | Health check (includes `ai_insights_available`) |
| GET | `/api/techniques/` | List all summarization techniques |
| GET | `/api/techniques/{id}/` | Get technique details |
| POST | `/api/videos/upload/` | Upload a video file |
| POST | `/api/videos/{file_id}/summarize/` | Summarize uploaded video |
| GET | `/api/videos/{file_id}/analytics/` | Per-frame analytics, scene cuts, heatmap & stats |
| GET | `/api/videos/{file_id}/analytics/export/` | Download per-frame analytics as CSV |
| GET | `/api/videos/{file_id}/compare/` | Compare all techniques on the same video |
| POST | `/api/videos/{file_id}/insights/` | AI-written interpretation of the analytics (requires `ANTHROPIC_API_KEY`) |
| DELETE | `/api/videos/{file_id}/` | Delete video and summaries |

## Configuration Options

### Summarization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frame_sample_rate` | 2 | Extract every Nth frame |
| `max_frames` | 3000 | Maximum frames to process |
| `output_fps` | 24 | Output video framerate |
| `summary_percent` | 20 | Portion of the original video to keep (5–80) |
| `n_clusters` | 15 | Number of clusters for selection |
| `motion_threshold` | 5.0 | Motion threshold for events |

### Combined Technique Weights

| Weight | Default | Description |
|--------|---------|-------------|
| `motion_weight` | 0.4 | Weight for motion features |
| `color_weight` | 0.3 | Weight for color features |
| `event_weight` | 0.3 | Weight for event detection |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DJANGO_SECRET_KEY` | dev key | **Set in production** |
| `DJANGO_DEBUG` | `True` | Set to `False` in production |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1` | Comma-separated hostnames |
| `ANTHROPIC_API_KEY` | — | Enables the AI Insights feature |
| `MAX_UPLOAD_MB` | `500` | Maximum upload size in MB |
| `SUMMARY_MAX_FRAME_WIDTH` | `0` | Downscale frames to this width during extraction; `0` keeps native resolution |
| `SUMMARY_MAX_EXTRACT_FRAMES` | `500` | Cap on frames held in memory per summarization |
| `ANALYTICS_CACHE_TTL` | `3600` | Seconds to cache analytics/comparison results |

> **Memory note:** summarization loads every sampled frame into RAM at once. At native
> 1080p, ~500 frames is roughly 3GB. On a small host, set `SUMMARY_MAX_FRAME_WIDTH=640`
> and `SUMMARY_MAX_EXTRACT_FRAMES=180` to keep peak usage a few hundred MB.

## Testing

```bash
cd backend
python smoke_test.py
```

Runs an end-to-end check: uploads a synthetic video, summarizes it, fetches analytics, verifies path-traversal protection, and cleans up.

## Deployment

### Render (free tier)

The repo ships a [`render.yaml`](render.yaml) blueprint, so deploying is:

1. Push this repo to GitHub.
2. In Render, choose **New → Blueprint** and pick the repo.
3. Render reads `render.yaml`, provisions the web service, and generates `DJANGO_SECRET_KEY` automatically.
4. Optionally add `ANTHROPIC_API_KEY` in the dashboard to switch on AI Insights.

The blueprint pins Python 3.11, runs `collectstatic`, and starts gunicorn with a 600s
timeout (summarization is synchronous, so the default 30s worker timeout would kill
long encodes).

Free-tier caveats worth knowing:

- **Cold starts.** The instance sleeps after ~15 minutes idle; the next request takes ~1 minute.
- **Ephemeral disk.** `uploads/` and `outputs/` are wiped on every restart and deploy. Fine for
  a demo; attach a disk or use object storage to make summaries durable.
- **512MB RAM / 0.1 CPU.** Hence the conservative `MAX_UPLOAD_MB=40`,
  `SUMMARY_MAX_FRAME_WIDTH=640` and `SUMMARY_MAX_EXTRACT_FRAMES=180` defaults in the blueprint.
  Raise them on a paid instance.

Netlify is not an option here — it only hosts static assets and serverless functions, and this
app is a stateful Django server doing CPU-bound OpenCV work.

### Generic (gunicorn)

For production deployment with gunicorn:

```bash
export DJANGO_DEBUG=False
export DJANGO_SECRET_KEY=<random secret>
export DJANGO_ALLOWED_HOSTS=your-domain.com
gunicorn config.wsgi:application --bind 0.0.0.0:8000
```

Notes:
- With `DEBUG=False`, HTTPS security headers (HSTS, secure cookies) are enabled automatically. Set `DJANGO_SECURE_SSL_REDIRECT=False` if TLS terminates at a proxy that doesn't forward the scheme.
- Serve `/uploads/` and `/outputs/` via your web server (nginx/Apache) in production; Django only serves them when `DEBUG=True`.
- Summarization runs synchronously in the request. Celery + Redis settings are wired in `config/celery.py` for moving it to a background queue if you need concurrent processing.

## License

MIT