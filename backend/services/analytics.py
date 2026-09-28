import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import base64

# Dense optical flow dominates the cost of every analytics call and its runtime
# scales with pixel count. Motion scores are normalised and only used to rank
# frames relative to each other, so a 240px working width is ample; the heatmap
# is upscaled for display afterwards.
FLOW_WIDTH = 240

# Farneback parameters tuned for speed over precision, for the same reason:
# fewer pyramid levels and iterations barely move the normalised scores.
_FARNEBACK = {
    'pyr_scale': 0.5, 'levels': 2, 'winsize': 13,
    'iterations': 2, 'poly_n': 5, 'poly_sigma': 1.1, 'flags': 0,
}


@dataclass
class FrameAnalytics:
    """Analytics data for a single frame."""
    index: int
    motion_score: float = 0.0
    color_diversity: float = 0.0
    event_score: float = 0.0
    combined_score: float = 0.0
    is_keyframe: bool = False
    timestamp: float = 0.0


@dataclass
class VideoAnalytics:
    """Complete analytics for a video."""
    frame_analytics: list[FrameAnalytics] = field(default_factory=list)
    motion_heatmap: Optional[str] = None  # Base64 encoded image
    scenes: list[dict] = field(default_factory=list)
    keyframe_thumbnails: list[dict] = field(default_factory=list)
    technique_scores: dict = field(default_factory=dict)
    summary_stats: dict = field(default_factory=dict)


class AnalyticsEngine:
    """Engine for generating video analytics and visualizations."""

    def __init__(self):
        self.frame_data = []

    def compute_full_analytics(
        self,
        frames: np.ndarray,
        fps: float = 30.0,
        keyframe_indices: list[int] = None
    ) -> VideoAnalytics:
        """Compute comprehensive analytics for video frames."""

        keyframe_indices = keyframe_indices or []
        analytics = VideoAnalytics()

        # Compute all feature scores. The flow pass also yields the heatmap
        # accumulator, so optical flow runs exactly once per request.
        motion_scores, heatmap_accumulator = self._flow_pass(frames)
        histograms = self._color_histograms(frames)
        color_scores = self._compute_color_diversity(histograms)
        scenes = self._detect_scenes(histograms, fps)
        event_scores = self._compute_event_scores(motion_scores)

        # Normalize scores
        motion_norm = self._normalize(motion_scores)
        color_norm = self._normalize(color_scores)
        event_norm = self._normalize(event_scores)

        # Combined score
        combined = 0.4 * motion_norm + 0.3 * color_norm + 0.3 * event_norm

        # Build frame analytics
        for i in range(len(frames)):
            fa = FrameAnalytics(
                index=i,
                motion_score=float(motion_norm[i]) if i < len(motion_norm) else 0,
                color_diversity=float(color_norm[i]),
                event_score=float(event_norm[i]) if i < len(event_norm) else 0,
                combined_score=float(combined[i]) if i < len(combined) else 0,
                is_keyframe=i in keyframe_indices,
                timestamp=i / fps if fps > 0 else 0
            )
            analytics.frame_analytics.append(fa)

        # Generate motion heatmap
        analytics.motion_heatmap = self._render_motion_heatmap(frames, heatmap_accumulator)

        # Generate keyframe thumbnails with annotations
        analytics.keyframe_thumbnails = self._generate_keyframe_thumbnails(
            frames, keyframe_indices, motion_norm, color_norm
        )

        analytics.scenes = scenes

        # Summary statistics
        analytics.summary_stats = {
            'total_frames': len(frames),
            'keyframes_count': len(keyframe_indices),
            'scene_count': len(scenes),
            'avg_scene_duration': round(
                float(np.mean([s['duration'] for s in scenes])), 2
            ) if scenes else 0.0,
            'avg_motion': float(np.mean(motion_scores)) if len(motion_scores) > 0 else 0,
            'max_motion': float(np.max(motion_scores)) if len(motion_scores) > 0 else 0,
            'motion_variance': float(np.var(motion_scores)) if len(motion_scores) > 0 else 0,
            'color_diversity_avg': float(np.mean(color_scores)),
            'high_activity_frames': int(np.sum(motion_norm > 0.7)) if len(motion_norm) > 0 else 0,
            'low_activity_frames': int(np.sum(motion_norm < 0.3)) if len(motion_norm) > 0 else 0,
        }

        # Store technique-specific scores
        analytics.technique_scores = {
            'motion': motion_norm.tolist(),
            'color': color_norm.tolist(),
            'event': event_norm.tolist(),
            'combined': combined.tolist(),
        }

        return analytics

    def _flow_pass(self, frames: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Run dense optical flow once, deriving both outputs that need it.

        Per-frame motion scores and the aggregated motion heatmap are both just
        reductions over the same flow fields. Computing them in separate passes
        meant every analytics request paid for optical flow twice.

        Returns (motion_scores, heatmap_accumulator). The accumulator is at flow
        resolution and is None when there are too few frames to compare.
        """
        n = len(frames)
        if n < 2:
            return np.zeros(n), None

        def to_flow_gray(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            if w > FLOW_WIDTH:
                gray = cv2.resize(gray, (FLOW_WIDTH, max(1, round(h * FLOW_WIDTH / w))))
            return gray

        prev_gray = to_flow_gray(frames[0])
        accumulator = np.zeros(prev_gray.shape, dtype=np.float32)
        motion_scores = [0.0]  # First frame has no motion

        for frame in frames[1:]:
            curr_gray = to_flow_gray(frame)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **_FARNEBACK)
            magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
            accumulator += magnitude
            motion_scores.append(float(magnitude.mean()))
            prev_gray = curr_gray

        return np.array(motion_scores), accumulator

    def _color_histograms(self, frames: np.ndarray) -> list[np.ndarray]:
        """Normalised 8x8x8 BGR histograms, shared by diversity and scene detection."""
        histograms = []
        for frame in frames:
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)
        return histograms

    def _detect_scenes(self, histograms: list[np.ndarray], fps: float) -> list[dict]:
        """Detect shot boundaries from jumps between consecutive colour histograms.

        A hard cut shows up as a spike in the distance between adjacent frames'
        histograms. The threshold adapts to the clip (mean + 3 sigma) so busy and
        static footage both work, with a floor so that noise within a single
        static shot doesn't register as a cut.
        """
        if len(histograms) < 3:
            return []

        distances = np.array([
            float(np.linalg.norm(histograms[i] - histograms[i - 1]))
            for i in range(1, len(histograms))
        ])

        threshold = max(float(distances.mean() + 3 * distances.std()), 0.25)
        cuts = [int(i + 1) for i, d in enumerate(distances) if d >= threshold]

        boundaries = [0] + cuts + [len(histograms)]
        scenes = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= start:
                continue
            scenes.append({
                'start_index': start,
                'end_index': end - 1,
                'start_time': round(start / fps, 2) if fps > 0 else 0.0,
                'end_time': round((end - 1) / fps, 2) if fps > 0 else 0.0,
                'duration': round((end - start) / fps, 2) if fps > 0 else 0.0,
            })
        return scenes

    def _compute_color_diversity(self, histograms: list[np.ndarray]) -> np.ndarray:
        """Compute color diversity score for each frame."""
        diversity_scores = []

        # Compute diversity (distance from neighbors)
        for i, hist in enumerate(histograms):
            start = max(0, i - 5)
            end = min(len(histograms), i + 6)
            neighbors = histograms[start:end]

            if len(neighbors) > 1:
                distances = [np.linalg.norm(hist - n) for n in neighbors]
                diversity_scores.append(np.mean(distances))
            else:
                diversity_scores.append(0)

        return np.array(diversity_scores)

    def _compute_event_scores(self, motion_scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Compute event importance scores."""
        if len(motion_scores) == 0:
            return np.array([])

        normalized = self._normalize(motion_scores)
        event_scores = np.zeros_like(normalized)

        in_event = False
        event_start = 0

        for i, score in enumerate(normalized):
            if score > threshold:
                if not in_event:
                    in_event = True
                    event_start = i
                event_scores[i] = score * 1.5  # Boost event frames
            else:
                if in_event:
                    in_event = False
                    event_length = i - event_start
                    # Boost based on event duration
                    event_scores[event_start:i] *= (1 + np.log1p(event_length) * 0.2)

        return event_scores

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to 0-1 range."""
        if len(arr) == 0:
            return arr
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    def _render_motion_heatmap(self, frames: np.ndarray, accumulator: Optional[np.ndarray]) -> str:
        """Render the accumulated flow magnitudes from _flow_pass as a base64 JPEG."""
        if accumulator is None or len(frames) < 2:
            return ""

        h, w = frames[0].shape[:2]

        # Normalize and apply colormap
        heatmap = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = heatmap.astype(np.uint8)
        # Accumulated at flow resolution; bring it back to frame size to blend.
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Blend with first frame for context
        blended = cv2.addWeighted(frames[0], 0.4, heatmap_colored, 0.6, 0)

        # Resize for web display
        max_dim = 640
        scale = min(max_dim / w, max_dim / h)
        if scale < 1:
            new_size = (int(w * scale), int(h * scale))
            blended = cv2.resize(blended, new_size)

        # Encode to base64
        _, buffer = cv2.imencode('.jpg', blended, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def _generate_keyframe_thumbnails(
        self,
        frames: np.ndarray,
        keyframe_indices: list[int],
        motion_scores: np.ndarray,
        color_scores: np.ndarray
    ) -> list[dict]:
        """Generate annotated thumbnails for keyframes."""
        thumbnails = []

        for idx in keyframe_indices:
            if idx >= len(frames):
                continue

            frame = frames[idx].copy()
            h, w = frame.shape[:2]

            # Resize thumbnail
            thumb_height = 120
            scale = thumb_height / h
            thumb = cv2.resize(frame, (int(w * scale), thumb_height))

            # Encode to base64
            _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            thumb_b64 = base64.b64encode(buffer).decode('utf-8')

            thumbnails.append({
                'index': idx,
                'thumbnail': thumb_b64,
                'motion_score': float(motion_scores[idx]) if idx < len(motion_scores) else 0,
                'color_score': float(color_scores[idx]) if idx < len(color_scores) else 0,
            })

        return thumbnails

    def compare_techniques(
        self,
        frames: np.ndarray,
        techniques: list[str] = None
    ) -> dict:
        """Compare different summarization techniques on the same video."""
        from . import (
            MotionSummarizer, ColorSummarizer, EventSummarizer,
            ObjectDetectionSummarizer, CombinedSummarizer
        )
        from .base import SummaryConfig

        techniques = techniques or ['motion', 'color', 'event', 'object_detection', 'combined']

        config = SummaryConfig(frame_sample_rate=1, max_frames=len(frames))

        summarizers = {
            'motion': MotionSummarizer(config, n_clusters=10),
            'color': ColorSummarizer(config, n_clusters=10),
            'event': EventSummarizer(config),
            'object_detection': ObjectDetectionSummarizer(config),
            'combined': CombinedSummarizer(config, n_clusters=10),
        }

        results = {}

        for name in techniques:
            if name not in summarizers:
                continue

            summarizer = summarizers[name]
            try:
                key_frames = summarizer.select_key_frames(frames)

                # Summarizers record which indices they selected; fall back to
                # sequential indices if a technique didn't set them.
                keyframe_indices = getattr(summarizer, '_selected_indices', None)
                if not keyframe_indices or len(keyframe_indices) != len(key_frames):
                    keyframe_indices = list(range(len(key_frames)))
                # numpy ints aren't JSON-serializable
                keyframe_indices = [int(i) for i in keyframe_indices]

                results[name] = {
                    'keyframe_count': len(key_frames),
                    'keyframe_indices': sorted(set(keyframe_indices)),
                    'coverage': len(set(keyframe_indices)) / len(frames) if len(frames) > 0 else 0,
                }
            except Exception as e:
                results[name] = {
                    'error': str(e),
                    'keyframe_count': 0,
                    'keyframe_indices': [],
                }

        return results
