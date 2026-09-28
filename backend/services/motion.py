import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from .base import BaseSummarizer, SummaryConfig
from typing import Optional


class MotionSummarizer(BaseSummarizer):
    """
    Advanced Motion-based video summarization.

    Features:
    - Dense optical flow with temporal smoothing
    - Motion segmentation to identify regions of interest
    - Scene cut detection using motion discontinuity
    - Adaptive clustering based on motion patterns
    - Temporal attention weighting
    """

    technique_name = "motion"

    def __init__(self, config: Optional[SummaryConfig] = None, n_clusters: int = 10):
        super().__init__(config)
        self.n_clusters = n_clusters

    def compute_dense_optical_flow(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute dense optical flow with motion segmentation.
        Returns motion magnitudes and flow fields.
        """
        self._report_progress(45, "Computing dense optical flow...")

        motion_features = []
        flow_fields = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for i, frame in enumerate(frames[1:], 1):
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Farneback optical flow with optimized parameters
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5,
                levels=5,  # More pyramid levels for better accuracy
                winsize=21,  # Larger window for smoother flow
                iterations=5,
                poly_n=7,
                poly_sigma=1.5,
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )

            # Compute magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Motion segmentation - focus on significant motion regions
            motion_mask = magnitude > np.percentile(magnitude, 75)
            significant_motion = np.mean(magnitude[motion_mask]) if np.any(motion_mask) else 0

            # Compute motion statistics
            motion_stats = {
                'mean': np.mean(magnitude),
                'max': np.max(magnitude),
                'std': np.std(magnitude),
                'significant': significant_motion,
                'coverage': np.sum(motion_mask) / magnitude.size,  # How much of frame has motion
                'direction_variance': np.var(angle[motion_mask]) if np.any(motion_mask) else 0
            }

            # Combined motion score with multiple factors
            motion_score = (
                0.3 * motion_stats['mean'] +
                0.2 * motion_stats['max'] +
                0.2 * motion_stats['significant'] +
                0.15 * motion_stats['coverage'] * 10 +
                0.15 * motion_stats['direction_variance']
            )

            motion_features.append(motion_score)
            flow_fields.append(flow)
            prev_gray = curr_gray

            if i % 30 == 0:
                progress = 45 + int(20 * i / len(frames))
                self._report_progress(progress, f"Analyzing motion frame {i}/{len(frames)}...")

        return np.array(motion_features), flow_fields

    def detect_scene_cuts(self, motion_features: np.ndarray) -> list[int]:
        """Detect scene cuts using motion discontinuity."""
        if len(motion_features) < 3:
            return []

        # Smooth the motion signal
        smoothed = gaussian_filter1d(motion_features, sigma=2)

        # Compute second derivative to find sudden changes
        diff = np.abs(np.diff(smoothed))

        # Find peaks in the difference signal (scene cuts)
        threshold = np.mean(diff) + 2 * np.std(diff)
        peaks, _ = find_peaks(diff, height=threshold, distance=10)

        return peaks.tolist()

    def compute_temporal_attention(self, motion_features: np.ndarray) -> np.ndarray:
        """
        Compute temporal attention weights.
        Frames that are different from their neighbors get higher attention.
        """
        n = len(motion_features)
        attention = np.ones(n)

        for i in range(n):
            # Look at local neighborhood
            start = max(0, i - 5)
            end = min(n, i + 6)
            local = motion_features[start:end]

            if len(local) > 1:
                # Higher attention for frames different from neighbors
                local_mean = np.mean(local)
                local_std = np.std(local) + 1e-8
                attention[i] = 1 + abs(motion_features[i] - local_mean) / local_std

        return attention / np.max(attention)

    def adaptive_clustering(
        self,
        frames: np.ndarray,
        motion_features: np.ndarray,
        attention_weights: np.ndarray
    ) -> tuple[np.ndarray, list[int]]:
        """
        Adaptive clustering using motion features and attention.
        Uses DBSCAN for automatic cluster discovery, falls back to KMeans.
        """
        self._report_progress(70, "Performing adaptive clustering...")

        # Combine motion and attention into feature matrix
        weighted_features = motion_features * attention_weights

        # Normalize features
        features = np.column_stack([
            (motion_features - motion_features.min()) / (motion_features.max() - motion_features.min() + 1e-8),
            (weighted_features - weighted_features.min()) / (weighted_features.max() - weighted_features.min() + 1e-8),
            attention_weights
        ])

        # Try DBSCAN first for automatic cluster discovery
        try:
            dbscan = DBSCAN(eps=0.3, min_samples=3)
            labels = dbscan.fit_predict(features)

            # If DBSCAN finds reasonable clusters, use them
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if 3 <= n_clusters <= self.n_clusters * 2:
                return self._select_from_clusters(frames, labels, weighted_features)
        except Exception:
            pass

        # Fall back to KMeans
        self._report_progress(75, "Using KMeans clustering...")
        n_clusters = min(self.n_clusters, len(frames) - 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        return self._select_from_clusters(frames, labels, weighted_features)

    def _select_from_clusters(
        self,
        frames: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray
    ) -> tuple[np.ndarray, list[int]]:
        """Select best frame from each cluster based on scores."""
        key_frames = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # Skip noise in DBSCAN
                continue

            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) > 0:
                # Select frame with highest score in cluster
                best_idx = cluster_indices[np.argmax(scores[cluster_indices])]
                key_frames.append((best_idx + 1, frames[best_idx + 1]))  # +1 because motion starts from frame 1

        # Sort by frame index and extract frames
        key_frames.sort(key=lambda x: x[0])
        indices = [f[0] for f in key_frames]
        return np.array([f[1] for f in key_frames]), indices

    def select_key_frames(self, frames: np.ndarray, **kwargs) -> np.ndarray:
        """Select key frames using optimized motion analysis."""
        target_frames = kwargs.get('n_clusters', self.n_clusters)
        min_frames = kwargs.get('min_frames', target_frames)
        target_frames = max(target_frames, min_frames)

        if len(frames) < 2:
            self._selected_indices = list(range(len(frames)))
            return frames

        if len(frames) <= target_frames:
            self._selected_indices = list(range(len(frames)))
            return frames

        self._report_progress(45, "Computing motion scores...")

        # Fast motion detection using frame differencing on small frames
        motion_scores = np.zeros(len(frames))
        small_size = (160, 90)
        prev_gray = cv2.cvtColor(cv2.resize(frames[0], small_size), cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(cv2.resize(frames[i], small_size), cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, curr_gray)
            motion_scores[i] = np.mean(diff)
            prev_gray = curr_gray

        self._report_progress(65, "Detecting scene changes...")

        # Detect scene cuts (high motion spikes)
        threshold = np.mean(motion_scores) + 2 * np.std(motion_scores)
        scene_cuts = np.where(motion_scores > threshold)[0].tolist()

        self._report_progress(75, "Selecting best frames...")

        # Select frames: divide into segments, pick highest motion from each
        step = len(frames) / target_frames
        selected_indices = []

        for i in range(target_frames):
            start = int(i * step)
            end = min(int((i + 1) * step), len(frames))
            if start >= end:
                start = end - 1

            segment_scores = motion_scores[start:end]
            best_in_segment = start + np.argmax(segment_scores)
            selected_indices.append(best_in_segment)

        # Include scene cuts
        for cut_idx in scene_cuts[:10]:  # Max 10 scene cuts
            if cut_idx not in selected_indices and len(selected_indices) < len(frames):
                selected_indices.append(cut_idx)

        selected_indices = sorted(set(selected_indices))[:target_frames]
        self._selected_indices = selected_indices

        self._report_progress(85, f"Selected {len(selected_indices)} key frames")
        return frames[selected_indices]
