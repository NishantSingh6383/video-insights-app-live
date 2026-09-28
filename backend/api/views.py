import csv
import io
import logging
import uuid

import cv2
import numpy as np
from pathlib import Path
from django.conf import settings
from django.core.cache import cache
from django.http import HttpResponse
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response

from services import (
    MotionSummarizer,
    ColorSummarizer,
    EventSummarizer,
    ObjectDetectionSummarizer,
    CombinedSummarizer,
    AnalyticsEngine,
)
from services.base import SummaryConfig
from .serializers import SummarizeRequestSerializer, TechniqueInfoSerializer
from .ai_insights import generate_insights, insights_available

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = ('.mp4', '.mpg', '.mpeg', '.avi', '.mov', '.mkv', '.webm')

# Frames are downscaled to this width before analytics to bound memory usage.
ANALYTICS_FRAME_WIDTH = settings.ANALYTICS_FRAME_WIDTH


def _cache_key(kind: str, file_id: str, *params) -> str:
    """Cache key for a derivative of an upload.

    No explicit invalidation is needed: file_id is a server-generated UUID that
    is never reused, and deleting a video removes the file, so _find_video 404s
    before any cached entry could be served. Leftovers age out via TTL.
    """
    return 'vi:{}:{}:{}'.format(kind, file_id, ':'.join(str(p) for p in params))


def _validate_file_id(file_id: str) -> bool:
    """File IDs are server-generated UUIDs; anything else is rejected.

    This also guards against path traversal, since file_id is interpolated
    into filesystem paths and glob patterns.
    """
    try:
        uuid.UUID(file_id)
        return True
    except (ValueError, AttributeError):
        return False


def _find_video(file_id: str):
    """Locate an uploaded video by its file_id. Returns a Path or None."""
    if not _validate_file_id(file_id):
        return None
    for ext in ALLOWED_EXTENSIONS:
        candidate = settings.MEDIA_ROOT / f"{file_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _extract_frames_for_analytics(filepath: Path, sample_rate: int, max_frames: int):
    """Extract downscaled frames for analytics endpoints.

    Downscaling bounds memory: 500 full-HD frames would be ~3GB as a numpy
    array; at 480px wide it's ~100MB.
    """
    cap = cv2.VideoCapture(str(filepath))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Fill a single preallocated array rather than building a list and calling
    # np.array() on it. That copy doubled peak memory - 300 frames at 480px is
    # ~117MB, so the list plus its copy cost ~234MB on top of a ~140MB baseline,
    # enough to get the worker OOM-killed on a 512MB instance. The kill returns
    # an empty body, which the browser reports as a JSON parse error.
    buffer = None
    read_count = 0
    kept = 0

    while kept < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if read_count % sample_rate == 0:
            h, w = frame.shape[:2]
            if w > ANALYTICS_FRAME_WIDTH:
                scale = ANALYTICS_FRAME_WIDTH / w
                frame = cv2.resize(frame, (ANALYTICS_FRAME_WIDTH, max(1, int(h * scale))))
            if buffer is None:
                buffer = np.empty((max_frames, *frame.shape), dtype=np.uint8)
            buffer[kept] = frame
            kept += 1
        read_count += 1
    cap.release()

    if buffer is None:
        return np.empty((0, 0, 0, 3), dtype=np.uint8), fps

    # A slice is a view, so this does not copy.
    return buffer[:kept], fps


TECHNIQUES = {
    'motion': {
        'class': MotionSummarizer,
        'info': {
            'id': 'motion',
            'name': 'Motion-based',
            'description': 'Analyzes optical flow between frames to detect movement patterns. '
                          'Uses KMeans clustering to select representative frames from different motion intensities.',
            'parameters': [
                {'name': 'n_clusters', 'type': 'int', 'default': 10, 'min': 2, 'max': 50,
                 'description': 'Number of motion clusters to create'},
            ]
        }
    },
    'color': {
        'class': ColorSummarizer,
        'info': {
            'id': 'color',
            'name': 'Color/Histogram-based',
            'description': 'Extracts color histograms and clusters frames by visual similarity. '
                          'Selects visually diverse frames that represent the video\'s color palette.',
            'parameters': [
                {'name': 'n_clusters', 'type': 'int', 'default': 15, 'min': 2, 'max': 50,
                 'description': 'Number of color clusters to create'},
            ]
        }
    },
    'event': {
        'class': EventSummarizer,
        'info': {
            'id': 'event',
            'name': 'Event-based',
            'description': 'Detects significant activity spikes using motion thresholds. '
                          'Captures key moments when something interesting happens.',
            'parameters': [
                {'name': 'motion_threshold', 'type': 'float', 'default': 5.0, 'min': 0.5, 'max': 30,
                 'description': 'Motion magnitude threshold to detect events'},
            ]
        }
    },
    'object_detection': {
        'class': ObjectDetectionSummarizer,
        'info': {
            'id': 'object_detection',
            'name': 'Object Detection',
            'description': 'Identifies frames containing significant objects or moving entities. '
                          'Uses contour detection to find frames with notable visual elements.',
            'parameters': [
                {'name': 'min_objects', 'type': 'int', 'default': 1, 'min': 1, 'max': 10,
                 'description': 'Minimum objects required to include frame'},
            ]
        }
    },
    'combined': {
        'class': CombinedSummarizer,
        'info': {
            'id': 'combined',
            'name': 'Combined Approach',
            'description': 'Integrates motion, color, and event detection for comprehensive summarization. '
                          'Weights each technique to balance visual diversity with activity capture.',
            'parameters': [
                {'name': 'n_clusters', 'type': 'int', 'default': 15, 'min': 2, 'max': 50,
                 'description': 'Number of clusters for final selection'},
                {'name': 'motion_weight', 'type': 'float', 'default': 0.4, 'min': 0, 'max': 1,
                 'description': 'Weight for motion features'},
                {'name': 'color_weight', 'type': 'float', 'default': 0.3, 'min': 0, 'max': 1,
                 'description': 'Weight for color features'},
                {'name': 'event_weight', 'type': 'float', 'default': 0.3, 'min': 0, 'max': 1,
                 'description': 'Weight for event detection'},
            ]
        }
    },
}


@api_view(['GET'])
def health_check(request):
    """Health check endpoint."""
    return Response({
        'status': 'healthy',
        'service': 'Video Insights API',
    })


@api_view(['GET'])
def list_techniques(request):
    """List all available summarization techniques."""
    techniques = [tech['info'] for tech in TECHNIQUES.values()]
    serializer = TechniqueInfoSerializer(techniques, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def get_technique(request, technique_id):
    """Get details of a specific technique."""
    if technique_id not in TECHNIQUES:
        return Response(
            {'error': f'Unknown technique: {technique_id}'},
            status=status.HTTP_404_NOT_FOUND
        )
    serializer = TechniqueInfoSerializer(TECHNIQUES[technique_id]['info'])
    return Response(serializer.data)


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_video(request):
    """Upload a video file and get its information."""
    if 'video' not in request.FILES:
        return Response(
            {'error': 'No video file provided'},
            status=status.HTTP_400_BAD_REQUEST
        )

    video_file = request.FILES['video']

    ext = Path(video_file.name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return Response(
            {'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'},
            status=status.HTTP_400_BAD_REQUEST
        )

    if video_file.size > settings.MAX_UPLOAD_SIZE:
        return Response(
            {'error': f'File too large. Maximum size: {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB'},
            status=status.HTTP_400_BAD_REQUEST
        )

    file_id = str(uuid.uuid4())
    filename = f"{file_id}{ext}"
    filepath = settings.MEDIA_ROOT / filename

    with open(filepath, 'wb') as f:
        for chunk in video_file.chunks():
            f.write(chunk)

    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        filepath.unlink(missing_ok=True)
        return Response(
            {'error': 'Could not read video file'},
            status=status.HTTP_400_BAD_REQUEST
        )

    video_info = {
        'file_id': file_id,
        'filename': video_file.name,
        'size_bytes': video_file.size,
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    video_info['duration_seconds'] = video_info['total_frames'] / video_info['fps'] if video_info['fps'] > 0 else 0

    cap.release()

    logger.info("Uploaded video %s (%s, %d bytes)", file_id, video_file.name, video_file.size)
    return Response(video_info, status=status.HTTP_201_CREATED)


@api_view(['POST'])
def summarize_video(request, file_id):
    """Summarize an uploaded video using the specified technique."""
    filepath = _find_video(file_id)
    if not filepath:
        return Response(
            {'error': 'Video not found', 'code': 'video_unavailable'},
            status=status.HTTP_404_NOT_FOUND
        )

    serializer = SummarizeRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    technique_id = data['technique']
    output_type = data.get('output_type', 'dynamic')

    if technique_id not in TECHNIQUES:
        return Response(
            {'error': f'Unknown technique: {technique_id}'},
            status=status.HTTP_400_BAD_REQUEST
        )

    config = SummaryConfig(
        frame_sample_rate=data['frame_sample_rate'],
        max_frames=data['max_frames'],
        output_fps=data['output_fps'],
        output_type=output_type,
        max_frame_width=settings.SUMMARY_MAX_FRAME_WIDTH,
    )

    summarizer_class = TECHNIQUES[technique_id]['class']
    summarizer = summarizer_class(config=config)

    # Output paths
    output_filename = f"{file_id}_{technique_id}_summary.mp4"
    output_path = settings.OUTPUT_ROOT / output_filename

    storyboard_filename = f"{file_id}_{technique_id}_storyboard.jpg"
    storyboard_path = settings.OUTPUT_ROOT / storyboard_filename

    # Get video info to calculate n_clusters based on summary_percent
    cap = cv2.VideoCapture(str(filepath))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Calculate number of frames to select based on summary_percent
    summary_percent = data.get('summary_percent', 20)
    video_duration = total_frames / fps if fps > 0 else 30

    # Target output frames based on desired duration
    target_duration = video_duration * summary_percent / 100
    target_frames = max(int(target_duration * data['output_fps']), 30)

    # Extract 3x target for good selection pool, but cap for speed and memory
    frames_to_extract = min(target_frames * 3, settings.SUMMARY_MAX_EXTRACT_FRAMES, total_frames)
    optimal_sample_rate = max(1, total_frames // frames_to_extract)

    config.frame_sample_rate = optimal_sample_rate
    config.max_frames = frames_to_extract + 20

    logger.info(
        "Summarizing %s: %.1fs video, target %d frames, extracting ~%d (rate=%d)",
        file_id, video_duration, target_frames, frames_to_extract, optimal_sample_rate,
    )

    kwargs = {}
    kwargs['n_clusters'] = target_frames
    kwargs['min_frames'] = target_frames

    if 'n_clusters' in data and data['n_clusters']:
        kwargs['n_clusters'] = data['n_clusters']
    if 'motion_threshold' in data:
        kwargs['motion_threshold'] = data['motion_threshold']
    if 'motion_weight' in data:
        kwargs['motion_weight'] = data['motion_weight']
    if 'color_weight' in data:
        kwargs['color_weight'] = data['color_weight']
    if 'event_weight' in data:
        kwargs['event_weight'] = data['event_weight']

    try:
        result = summarizer.summarize(
            str(filepath),
            str(output_path),
            storyboard_path=str(storyboard_path) if output_type in ('static', 'both') else None,
            **kwargs
        )
    except Exception:
        logger.exception("Summarization failed for %s (technique=%s)", file_id, technique_id)
        return Response(
            {'error': 'Summarization failed. Check server logs for details.'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    compression_ratio = result.key_frames_selected / result.total_frames_processed if result.total_frames_processed > 0 else 0

    response_data = {
        'total_frames_processed': result.total_frames_processed,
        'key_frames_selected': result.key_frames_selected,
        'technique': result.technique,
        'output_type': output_type,
        'duration_seconds': round(result.duration_seconds, 2),
        'compression_ratio': round(compression_ratio, 4),
        'metadata': result.metadata,
    }

    # Add video URL if dynamic - use actual output path from result (may have different extension)
    if output_type in ('dynamic', 'both'):
        actual_output_path = Path(result.output_path)
        actual_filename = actual_output_path.name
        response_data['output_path'] = str(actual_output_path)
        response_data['output_url'] = f'/outputs/{actual_filename}'

    # Add storyboard URL if static
    if output_type in ('static', 'both'):
        response_data['storyboard_url'] = f'/outputs/{storyboard_filename}'
        response_data['frames'] = result.frames_data or []

    return Response(response_data)


@api_view(['DELETE'])
def delete_video(request, file_id):
    """Delete an uploaded video and its summaries."""
    if not _validate_file_id(file_id):
        return Response(
            {'error': 'Video not found', 'code': 'video_unavailable'},
            status=status.HTTP_404_NOT_FOUND
        )

    deleted = []

    for ext in ALLOWED_EXTENSIONS:
        filepath = settings.MEDIA_ROOT / f"{file_id}{ext}"
        if filepath.exists():
            filepath.unlink()
            deleted.append(filepath.name)

    for output_file in settings.OUTPUT_ROOT.glob(f"{file_id}_*"):
        output_file.unlink()
        deleted.append(output_file.name)

    if not deleted:
        return Response(
            {'error': 'Video not found', 'code': 'video_unavailable'},
            status=status.HTTP_404_NOT_FOUND
        )

    logger.info("Deleted video %s (%d files)", file_id, len(deleted))
    return Response({'deleted': deleted})


def _analytics_payload(file_id, filepath, sample_rate, max_frames):
    """Build (or fetch from cache) the analytics payload for a video.

    Shared by the JSON endpoint and the CSV export so the export never triggers
    a second, redundant optical-flow pass.

    Returns (payload, error_response). Exactly one is not None.
    """
    cache_key = _cache_key('analytics', file_id, sample_rate, max_frames)
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info("Analytics cache hit for %s", file_id)
        return cached, None

    frames, fps = _extract_frames_for_analytics(filepath, sample_rate, max_frames)

    if len(frames) == 0:
        return None, Response(
            {'error': 'Could not extract frames from video'},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Only every Nth frame was kept, so the effective frame rate of this
    # sequence is fps/sample_rate; passing raw fps made every reported
    # timestamp sample_rate times too small.
    engine = AnalyticsEngine()
    analytics = engine.compute_full_analytics(frames, fps / sample_rate if sample_rate else fps)

    payload = {
        'frame_scores': [
            {
                'index': fa.index,
                'motion': round(fa.motion_score, 4),
                'color': round(fa.color_diversity, 4),
                'event': round(fa.event_score, 4),
                'combined': round(fa.combined_score, 4),
                'timestamp': round(fa.timestamp, 2),
                'is_keyframe': fa.is_keyframe,
            }
            for fa in analytics.frame_analytics
        ],
        'motion_heatmap': analytics.motion_heatmap,
        'scenes': analytics.scenes,
        'summary_stats': analytics.summary_stats,
        'technique_scores': analytics.technique_scores,
    }

    cache.set(cache_key, payload)
    return payload, None


def _analytics_params(request):
    """Parse the shared sample_rate/max_frames query params."""
    sample_rate = max(1, int(request.GET.get('sample_rate', 10)))
    max_frames = min(max(1, int(request.GET.get('max_frames', 500))), settings.ANALYTICS_MAX_FRAMES)
    return sample_rate, max_frames


@api_view(['GET'])
def get_analytics(request, file_id):
    """Get comprehensive analytics for an uploaded video."""
    filepath = _find_video(file_id)
    if not filepath:
        return Response(
            {'error': 'Video not found', 'code': 'video_unavailable'},
            status=status.HTTP_404_NOT_FOUND
        )

    try:
        sample_rate, max_frames = _analytics_params(request)
    except ValueError:
        return Response(
            {'error': 'sample_rate and max_frames must be integers'},
            status=status.HTTP_400_BAD_REQUEST
        )

    payload, error = _analytics_payload(file_id, filepath, sample_rate, max_frames)
    return error or Response(payload)


@api_view(['GET'])
def export_analytics(request, file_id):
    """Download per-frame analytics as CSV, for use outside the dashboard."""
    filepath = _find_video(file_id)
    if not filepath:
        return Response(
            {'error': 'Video not found', 'code': 'video_unavailable'},
            status=status.HTTP_404_NOT_FOUND
        )

    try:
        sample_rate, max_frames = _analytics_params(request)
    except ValueError:
        return Response(
            {'error': 'sample_rate and max_frames must be integers'},
            status=status.HTTP_400_BAD_REQUEST
        )

    payload, error = _analytics_payload(file_id, filepath, sample_rate, max_frames)
    if error:
        return error

    scene_of = {}
    for number, scene in enumerate(payload['scenes'], start=1):
        for idx in range(scene['start_index'], scene['end_index'] + 1):
            scene_of[idx] = number

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        'frame_index', 'timestamp_seconds', 'scene',
        'motion', 'color_diversity', 'event', 'combined', 'is_keyframe',
    ])
    for row in payload['frame_scores']:
        writer.writerow([
            row['index'], row['timestamp'], scene_of.get(row['index'], ''),
            row['motion'], row['color'], row['event'], row['combined'],
            int(row['is_keyframe']),
        ])

    response = HttpResponse(buffer.getvalue(), content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{file_id}_analytics.csv"'
    return response


@api_view(['GET'])
def compare_techniques(request, file_id):
    """Compare all summarization techniques on the same video."""
    filepath = _find_video(file_id)
    if not filepath:
        return Response(
            {'error': 'Video not found', 'code': 'video_unavailable'},
            status=status.HTTP_404_NOT_FOUND
        )

    try:
        sample_rate = max(1, int(request.GET.get('sample_rate', 10)))
        max_frames = min(max(1, int(request.GET.get('max_frames', 300))), settings.ANALYTICS_MAX_FRAMES)
    except ValueError:
        return Response(
            {'error': 'sample_rate and max_frames must be integers'},
            status=status.HTTP_400_BAD_REQUEST
        )

    cache_key = _cache_key('compare', file_id, sample_rate, max_frames)
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info("Compare cache hit for %s", file_id)
        return Response(cached)

    frames, _ = _extract_frames_for_analytics(filepath, sample_rate, max_frames)

    if len(frames) == 0:
        return Response(
            {'error': 'Could not extract frames from video'},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Compare techniques
    engine = AnalyticsEngine()
    comparison = engine.compare_techniques(frames)

    response_data = {
        'total_frames': len(frames),
        'techniques': comparison,
    }

    cache.set(cache_key, response_data)
    return Response(response_data)


@api_view(['POST'])
def ai_insights(request, file_id):
    """Generate a narrative summary of the video's analytics with recommendations."""
    if not insights_available():
        return Response(
            {'error': 'Insights generation is not available.'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    filepath = _find_video(file_id)
    if not filepath:
        return Response(
            {'error': 'Video not found', 'code': 'video_unavailable'},
            status=status.HTTP_404_NOT_FOUND
        )

    frames, fps = _extract_frames_for_analytics(filepath, sample_rate=10, max_frames=300)

    if len(frames) == 0:
        return Response(
            {'error': 'Could not extract frames from video'},
            status=status.HTTP_400_BAD_REQUEST
        )

    engine = AnalyticsEngine()
    # sample_rate=10 above, so the sequence runs at a tenth of the native rate.
    analytics = engine.compute_full_analytics(frames, fps / 10 if fps else fps)

    # Summarization context (optional, sent by the frontend after a summary run)
    summary_context = request.data if isinstance(request.data, dict) else {}

    try:
        insights = generate_insights(analytics.summary_stats, summary_context)
    except Exception:
        logger.exception("AI insights generation failed for %s", file_id)
        return Response(
            {'error': 'AI insights generation failed. Check server logs for details.'},
            status=status.HTTP_502_BAD_GATEWAY
        )

    return Response({'insights': insights})