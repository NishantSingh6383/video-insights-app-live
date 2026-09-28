import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from .base import BaseSummarizer, SummaryConfig
from typing import Optional


class EventSummarizer(BaseSummarizer):
    """
    Advanced Event-based video summarization.

    Features:
    - Multi-scale temporal analysis
    - Shot boundary detection using histogram difference
    - Highlight detection using motion + audio proxy (brightness changes)
    - Anomaly detection for unusual frames
    - Event importance ranking with temporal context
    """

    technique_name = "event"

    def __init__(
        self,
        config: Optional[SummaryConfig] = None,
        motion_threshold: float = 5.0,
        min_event_frames: int = 3
    ):
        super().__init__(config)
        self.motion_threshold = motion_threshold
        self.min_event_frames = min_event_frames

    def compute_motion_energy(self, frames: np.ndarray) -> np.ndarray:
        """Compute motion energy using optical flow magnitude."""
        motion_energy = [0.0]
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_energy.append(np.mean(magnitude))
            prev_gray = curr_gray

        return np.array(motion_energy)

    def compute_histogram_difference(self, frames: np.ndarray) -> np.ndarray:
        """Compute histogram difference for shot boundary detection."""
        hist_diff = [0.0]

        prev_hist = self._compute_frame_histogram(frames[0])

        for frame in frames[1:]:
            curr_hist = self._compute_frame_histogram(frame)
            # Chi-square distance
            diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CHISQR)
            hist_diff.append(diff)
            prev_hist = curr_hist

        return np.array(hist_diff)

    def _compute_frame_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute normalized color histogram."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def compute_brightness_changes(self, frames: np.ndarray) -> np.ndarray:
        """Compute brightness changes (proxy for flash/highlight detection)."""
        brightness = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(np.mean(gray))

        brightness = np.array(brightness)

        # Compute rate of change
        changes = np.abs(np.diff(brightness, prepend=brightness[0]))
        return changes

    def detect_shot_boundaries(
        self,
        hist_diff: np.ndarray,
        motion_energy: np.ndarray
    ) -> list[int]:
        """Detect shot boundaries using histogram and motion cues."""
        # Normalize signals
        hist_norm = (hist_diff - hist_diff.min()) / (hist_diff.max() - hist_diff.min() + 1e-8)
        motion_norm = (motion_energy - motion_energy.min()) / (motion_energy.max() - motion_energy.min() + 1e-8)

        # Combined signal
        combined = 0.7 * hist_norm + 0.3 * motion_norm

        # Adaptive threshold
        threshold = np.mean(combined) + 2 * np.std(combined)

        # Find peaks
        peaks, properties = find_peaks(combined, height=threshold, distance=15, prominence=0.1)

        return peaks.tolist()

    def detect_highlights(
        self,
        motion_energy: np.ndarray,
        brightness_changes: np.ndarray
    ) -> list[tuple[int, int, float]]:
        """Detect highlight segments based on activity levels."""
        # Smooth signals
        motion_smooth = savgol_filter(motion_energy, min(15, len(motion_energy) // 2 * 2 + 1), 3)
        brightness_smooth = savgol_filter(brightness_changes, min(15, len(brightness_changes) // 2 * 2 + 1), 3)

        # Combined activity score
        motion_norm = (motion_smooth - motion_smooth.min()) / (motion_smooth.max() - motion_smooth.min() + 1e-8)
        bright_norm = (brightness_smooth - brightness_smooth.min()) / (brightness_smooth.max() - brightness_smooth.min() + 1e-8)

        activity = 0.7 * motion_norm + 0.3 * bright_norm

        # Adaptive threshold
        threshold = np.percentile(activity, 70)

        # Find highlight segments
        highlights = []
        in_highlight = False
        start = 0

        for i, score in enumerate(activity):
            if score > threshold:
                if not in_highlight:
                    in_highlight = True
                    start = i
            else:
                if in_highlight:
                    if i - start >= self.min_event_frames:
                        importance = np.mean(activity[start:i])
                        highlights.append((start, i - 1, importance))
                    in_highlight = False

        # Handle end case
        if in_highlight and len(activity) - start >= self.min_event_frames:
            importance = np.mean(activity[start:])
            highlights.append((start, len(activity) - 1, importance))

        return highlights

    def detect_anomalies(self, frames: np.ndarray, features: np.ndarray) -> list[int]:
        """Detect anomalous frames using statistical analysis."""
        # Use IQR-based anomaly detection on combined features
        feature_means = np.mean(features, axis=1) if features.ndim > 1 else features

        q1, q3 = np.percentile(feature_means, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        anomalies = np.where((feature_means < lower_bound) | (feature_means > upper_bound))[0]
        return anomalies.tolist()

    def rank_events(
        self,
        highlights: list[tuple[int, int, float]],
        shot_boundaries: list[int],
        anomalies: list[int],
        motion_energy: np.ndarray
    ) -> list[tuple[int, float]]:
        """Rank all events by importance."""
        events = []

        # Add highlight events
        for start, end, importance in highlights:
            mid = (start + end) // 2
            events.append((mid, importance * 1.5))  # Boost highlights

        # Add shot boundaries
        for boundary in shot_boundaries:
            events.append((boundary, 0.8))

        # Add anomalies
        for anomaly in anomalies:
            events.append((anomaly, motion_energy[anomaly] / np.max(motion_energy)))

        # Sort by importance
        events.sort(key=lambda x: x[1], reverse=True)

        return events

    def select_key_frames(self, frames: np.ndarray, **kwargs) -> np.ndarray:
        """Select key frames based on event detection."""
        threshold = kwargs.get('motion_threshold', self.motion_threshold)
        self.motion_threshold = threshold

        if len(frames) < 2:
            return frames

        self._report_progress(45, "Computing motion energy...")
        motion_energy = self.compute_motion_energy(frames)

        self._report_progress(55, "Analyzing histogram differences...")
        hist_diff = self.compute_histogram_difference(frames)

        self._report_progress(60, "Computing brightness changes...")
        brightness_changes = self.compute_brightness_changes(frames)

        self._report_progress(65, "Detecting shot boundaries...")
        shot_boundaries = self.detect_shot_boundaries(hist_diff, motion_energy)

        self._report_progress(70, "Detecting highlights...")
        highlights = self.detect_highlights(motion_energy, brightness_changes)

        self._report_progress(75, "Detecting anomalies...")
        anomalies = self.detect_anomalies(frames, motion_energy)

        self._report_progress(80, "Ranking events...")
        ranked_events = self.rank_events(highlights, shot_boundaries, anomalies, motion_energy)

        # Use target from kwargs
        target_frames = kwargs.get('n_clusters', max(10, len(frames) // 10))
        min_frames = kwargs.get('min_frames', target_frames)
        target_frames = max(target_frames, min_frames)

        n_select = min(len(ranked_events), target_frames)

        if n_select == 0:
            # Fallback: uniform sampling
            step = max(1, len(frames) // target_frames)
            indices = list(range(0, len(frames), step))[:target_frames]
        else:
            indices = sorted(set([e[0] for e in ranked_events[:n_select]]))

        # Fill if not enough
        if len(indices) < target_frames:
            step = len(frames) // target_frames
            for i in range(0, len(frames), step):
                if len(indices) >= target_frames:
                    break
                if i not in indices:
                    indices.append(i)
            indices = sorted(indices)[:target_frames]

        self._selected_indices = indices

        self._report_progress(85, f"Selected {len(indices)} event frames")
        return frames[indices]
