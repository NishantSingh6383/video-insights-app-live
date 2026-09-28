import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import base64


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

        # Compute all feature scores
        motion_scores = self._compute_motion_scores(frames)
        color_scores = self._compute_color_diversity(frames)
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
        analytics.motion_heatmap = self._generate_motion_heatmap(frames)

        # Generate keyframe thumbnails with annotations
        analytics.keyframe_thumbnails = self._generate_keyframe_thumbnails(
            frames, keyframe_indices, motion_norm, color_norm
        )

        # Summary statistics
        analytics.summary_stats = {
            'total_frames': len(frames),
            'keyframes_count': len(keyframe_indices),
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

    def _compute_motion_scores(self, frames: np.ndarray) -> np.ndarray:
        """Compute motion magnitude between consecutive frames."""
        if len(frames) < 2:
            return np.zeros(len(frames))

        motion_scores = [0.0]  # First frame has no motion
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_scores.append(np.mean(magnitude))
            prev_gray = curr_gray

        return np.array(motion_scores)

    def _compute_color_diversity(self, frames: np.ndarray) -> np.ndarray:
        """Compute color diversity score for each frame."""
        diversity_scores = []
        histograms = []

        # First pass: compute all histograms
        for frame in frames:
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)

        # Second pass: compute diversity (distance from neighbors)
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

    def _generate_motion_heatmap(self, frames: np.ndarray) -> str:
        """Generate an aggregated motion heatmap as base64 image."""
        if len(frames) < 2:
            return ""

        h, w = frames[0].shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            heatmap += magnitude
            prev_gray = curr_gray

        # Normalize and apply colormap
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = heatmap.astype(np.uint8)
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
