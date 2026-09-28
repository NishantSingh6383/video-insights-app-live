import logging
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .base import BaseSummarizer, SummaryConfig

logger = logging.getLogger(__name__)


class CombinedSummarizer(BaseSummarizer):
    """
    Advanced Combined video summarization using multi-modal fusion.

    Features:
    - Adaptive weight learning based on video characteristics
    - Temporal consistency enforcement
    - Shot-aware summarization
    - Submodular optimization for diverse selection
    - Quality-aware frame selection
    - Deep feature approximation using CNN-like processing
    """

    technique_name = "combined"

    def __init__(
        self,
        config: Optional[SummaryConfig] = None,
        n_clusters: int = 15,
        motion_weight: float = 0.4,
        color_weight: float = 0.3,
        event_weight: float = 0.3,
        motion_threshold: float = 5.0,
    ):
        super().__init__(config)
        self.n_clusters = n_clusters
        self.motion_weight = motion_weight
        self.color_weight = color_weight
        self.event_weight = event_weight
        self.motion_threshold = motion_threshold

    def compute_motion_features(self, frames: np.ndarray) -> np.ndarray:
        """Compute motion features - optimized for speed."""
        motion_scores = np.zeros(len(frames))

        # Resize frames for faster optical flow
        small_size = (320, 240)
        prev_gray = cv2.cvtColor(cv2.resize(frames[0], small_size), cv2.COLOR_BGR2GRAY)

        for i, frame in enumerate(frames[1:], 1):
            small_frame = cv2.resize(frame, small_size)
            curr_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # Faster optical flow with smaller parameters
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_scores[i] = np.mean(magnitude) + 0.5 * np.max(magnitude)

            prev_gray = curr_gray

        return motion_scores

    def compute_visual_features(self, frames: np.ndarray) -> np.ndarray:
        """Compute visual features - simplified for speed."""
        features = []

        for frame in frames:
            # Resize for faster processing
            small = cv2.resize(frame, (160, 120))

            frame_features = []

            # Color histogram in HSV (most perceptually relevant)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            for ch in range(3):
                hist = cv2.calcHist([hsv], [ch], None, [16], [0, 256])
                frame_features.extend(hist.flatten())

            # Simple gradient magnitude
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(sobelx**2 + sobely**2)

            frame_features.extend([
                np.mean(gradient_mag),
                np.std(gradient_mag),
            ])

            features.append(np.array(frame_features))

        return np.array(features)

    def compute_quality_scores(self, frames: np.ndarray) -> np.ndarray:
        """Compute frame quality scores (blur, contrast, noise)."""
        quality_scores = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Contrast using standard deviation
            contrast = np.std(gray)

            # Brightness balance
            brightness = np.mean(gray)
            brightness_score = 1 - abs(brightness - 128) / 128

            # Combined quality score
            quality = (
                0.4 * min(laplacian_var / 500, 1) +  # Sharpness
                0.3 * min(contrast / 64, 1) +         # Contrast
                0.3 * brightness_score                 # Brightness balance
            )

            quality_scores.append(quality)

        return np.array(quality_scores)

    def detect_shots(self, frames: np.ndarray, motion_features: np.ndarray) -> list[tuple[int, int]]:
        """Detect shot boundaries and segment video."""
        # Histogram-based shot detection
        hist_diff = [0.0]
        prev_hist = self._compute_histogram(frames[0])

        for frame in frames[1:]:
            curr_hist = self._compute_histogram(frame)
            diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CHISQR)
            hist_diff.append(diff)
            prev_hist = curr_hist

        hist_diff = np.array(hist_diff)

        # Combine with motion for shot detection
        combined = 0.7 * (hist_diff / (hist_diff.max() + 1e-8)) + \
                   0.3 * (motion_features / (motion_features.max() + 1e-8))

        # Find shot boundaries
        threshold = np.mean(combined) + 2 * np.std(combined)
        boundaries = [0]

        for i in range(1, len(combined)):
            if combined[i] > threshold:
                boundaries.append(i)

        boundaries.append(len(frames) - 1)

        # Create shot segments
        shots = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]
        return shots

    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute color histogram for shot detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def compute_event_scores(self, motion_features: np.ndarray) -> np.ndarray:
        """Compute event importance scores."""
        # Smooth motion signal
        smoothed = gaussian_filter1d(motion_features, sigma=2)

        # Find peaks (events)
        peaks, properties = find_peaks(smoothed, prominence=np.std(smoothed) * 0.5)

        event_scores = np.zeros_like(motion_features)

        for peak in peaks:
            # Create event region around peak
            start = max(0, peak - 5)
            end = min(len(motion_features), peak + 6)
            event_scores[start:end] = smoothed[start:end]

        return event_scores

    def compute_temporal_consistency(self, features: np.ndarray) -> np.ndarray:
        """Compute temporal consistency weights to favor stable frames."""
        consistency = np.ones(len(features))

        for i in range(1, len(features) - 1):
            # Compare with neighbors
            prev_dist = np.linalg.norm(features[i] - features[i-1])
            next_dist = np.linalg.norm(features[i] - features[i+1])

            # Higher consistency if similar to neighbors (stable frame)
            consistency[i] = 1.0 / (1.0 + (prev_dist + next_dist) / 2)

        return consistency

    def adaptive_weight_learning(
        self,
        frames: np.ndarray,
        motion_features: np.ndarray,
        visual_features: np.ndarray
    ) -> tuple[float, float, float]:
        """Learn adaptive weights based on video characteristics."""
        # Motion intensity
        motion_intensity = np.std(motion_features)

        # Visual diversity
        visual_diversity = np.mean([
            np.linalg.norm(visual_features[i] - visual_features[j])
            for i in range(min(20, len(visual_features)))
            for j in range(i+1, min(20, len(visual_features)))
        ]) if len(visual_features) > 1 else 0

        # Normalize
        total = motion_intensity + visual_diversity + 1

        motion_w = 0.3 + 0.2 * (motion_intensity / total)
        color_w = 0.3 + 0.2 * (visual_diversity / total)
        event_w = 1.0 - motion_w - color_w

        return motion_w, color_w, event_w

    def submodular_selection(
        self,
        frames: np.ndarray,
        scores: np.ndarray,
        visual_features: np.ndarray,
        shots: list[tuple[int, int]],
        n_select: int
    ) -> list[int]:
        """Submodular optimization for diverse, representative selection."""
        selected = []
        selected_set = set()

        # Ensure each shot has at least one frame
        for start, end in shots:
            if end > start:
                shot_scores = scores[start:end+1]
                best_in_shot = start + np.argmax(shot_scores)
                if best_in_shot not in selected_set:
                    selected.append(best_in_shot)
                    selected_set.add(best_in_shot)

        # Fill remaining slots with diverse frames
        remaining_slots = n_select - len(selected)

        if remaining_slots > 0:
            # For efficiency with large n_select, use a hybrid approach
            remaining_indices = [i for i in range(len(frames)) if i not in selected_set]

            if remaining_slots > 100:
                # For large selections, prioritize by score with temporal spacing
                scored_remaining = [(i, scores[i]) for i in remaining_indices]
                scored_remaining.sort(key=lambda x: x[1], reverse=True)

                min_spacing = max(1, len(frames) // n_select)

                for idx, score in scored_remaining:
                    if len(selected) >= n_select:
                        break
                    # Check temporal spacing
                    if all(abs(idx - s) >= min_spacing for s in selected):
                        selected.append(idx)
                        selected_set.add(idx)

                # If still need more, add without spacing constraint
                for idx, score in scored_remaining:
                    if len(selected) >= n_select:
                        break
                    if idx not in selected_set:
                        selected.append(idx)
                        selected_set.add(idx)
            else:
                # Original submodular selection for smaller selections
                for _ in range(remaining_slots):
                    if not remaining_indices:
                        break

                    best_gain = -np.inf
                    best_idx = None

                    for idx in remaining_indices:
                        score_gain = scores[idx]

                        if selected:
                            min_dist = min(
                                np.linalg.norm(visual_features[idx] - visual_features[s])
                                for s in selected[-20:]  # Only compare to recent selections for speed
                            )
                            diversity_gain = min_dist
                        else:
                            diversity_gain = 0

                        gain = 0.6 * score_gain + 0.4 * diversity_gain

                        if gain > best_gain:
                            best_gain = gain
                            best_idx = idx

                    if best_idx is not None:
                        selected.append(best_idx)
                        selected_set.add(best_idx)
                        remaining_indices.remove(best_idx)

        return sorted(selected)

    def select_key_frames(self, frames: np.ndarray, **kwargs) -> np.ndarray:
        """Select key frames using optimized multi-modal analysis."""
        n_clusters = kwargs.get('n_clusters', self.n_clusters)
        min_frames = kwargs.get('min_frames', max(30, n_clusters))
        target_frames = max(n_clusters, min_frames)

        logger.debug("select_key_frames: %d frames, target=%d", len(frames), target_frames)

        if len(frames) <= target_frames:
            self._selected_indices = list(range(len(frames)))
            return frames

        # OPTIMIZED: Analyze every Nth frame, interpolate scores for others
        analyze_step = max(1, len(frames) // 100)  # Analyze ~100 frames max
        analyze_indices = list(range(0, len(frames), analyze_step))

        self._report_progress(42, f"Analyzing {len(analyze_indices)} key points...")

        # 1. Motion detection (fast - small frames)
        motion_scores = np.zeros(len(frames))
        small_size = (160, 90)
        prev_small = cv2.cvtColor(cv2.resize(frames[0], small_size), cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frames)):
            curr_small = cv2.cvtColor(cv2.resize(frames[i], small_size), cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_small, curr_small)
            motion_scores[i] = np.mean(diff)
            prev_small = curr_small

        self._report_progress(55, "Computing quality scores...")

        # 2. Quality scores (sharpness) - only for analyzed frames, interpolate
        quality_sparse = {}
        for idx in analyze_indices:
            gray = cv2.cvtColor(cv2.resize(frames[idx], (320, 180)), cv2.COLOR_BGR2GRAY)
            quality_sparse[idx] = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Interpolate quality scores
        quality_scores = np.zeros(len(frames))
        indices = sorted(quality_sparse.keys())
        values = [quality_sparse[i] for i in indices]
        quality_scores = np.interp(range(len(frames)), indices, values)

        self._report_progress(70, "Scoring frames...")

        # 3. Combined score: motion + quality
        motion_norm = (motion_scores - motion_scores.min()) / (motion_scores.max() - motion_scores.min() + 1e-8)
        quality_norm = (quality_scores - quality_scores.min()) / (quality_scores.max() - quality_scores.min() + 1e-8)

        # High motion = interesting, high quality = sharp
        combined_scores = 0.6 * motion_norm + 0.4 * quality_norm

        self._report_progress(80, "Selecting best frames...")

        # 4. Select: divide into segments, pick best from each
        step = len(frames) / target_frames
        selected_indices = []

        for i in range(target_frames):
            start = int(i * step)
            end = min(int((i + 1) * step), len(frames))
            if start >= end:
                start = end - 1

            # Pick frame with highest combined score in segment
            segment_scores = combined_scores[start:end]
            best_in_segment = start + np.argmax(segment_scores)
            selected_indices.append(best_in_segment)

        self._selected_indices = selected_indices
        key_frames = frames[selected_indices]

        self._report_progress(90, f"Selected {len(key_frames)} key frames")
        return key_frames

    def select_key_frames_advanced(self, frames: np.ndarray, **kwargs) -> np.ndarray:
        """Select key frames using advanced multi-modal fusion (slower but more accurate)."""
        n_clusters = kwargs.get('n_clusters', self.n_clusters)
        min_frames = kwargs.get('min_frames', max(30, n_clusters))

        # Ensure we select the requested number of frames
        n_clusters = max(n_clusters, min_frames)

        self._report_progress(42, "Computing motion features...")
        motion_features = self.compute_motion_features(frames)

        self._report_progress(50, "Computing visual features...")
        visual_features = self.compute_visual_features(frames)

        self._report_progress(58, "Computing quality scores...")
        quality_scores = self.compute_quality_scores(frames)

        self._report_progress(63, "Detecting shots...")
        shots = self.detect_shots(frames, motion_features)

        self._report_progress(68, "Computing event scores...")
        event_scores = self.compute_event_scores(motion_features)

        self._report_progress(72, "Learning adaptive weights...")
        motion_w, color_w, event_w = self.adaptive_weight_learning(
            frames, motion_features, visual_features
        )

        # Normalize all features
        def normalize(arr):
            min_val, max_val = arr.min(), arr.max()
            if max_val - min_val < 1e-8:
                return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)

        motion_norm = normalize(motion_features)
        event_norm = normalize(event_scores)
        quality_norm = normalize(quality_scores)

        # Visual diversity (local contrast)
        visual_diversity = np.zeros(len(frames))
        for i in range(len(frames)):
            neighbors = visual_features[max(0, i-3):min(len(frames), i+4)]
            if len(neighbors) > 1:
                visual_diversity[i] = np.mean([
                    np.linalg.norm(visual_features[i] - n) for n in neighbors
                ])
        color_norm = normalize(visual_diversity)

        self._report_progress(78, "Computing combined scores...")

        # Combined score with quality weighting
        combined_scores = (
            motion_w * motion_norm +
            color_w * color_norm +
            event_w * event_norm
        ) * (0.7 + 0.3 * quality_norm)  # Quality boost

        # Temporal consistency
        consistency = self.compute_temporal_consistency(visual_features)
        combined_scores *= (0.8 + 0.2 * consistency)

        self._report_progress(82, "Performing submodular selection...")

        # Select frames
        selected_indices = self.submodular_selection(
            frames, combined_scores, visual_features, shots, n_clusters
        )

        logger.debug("submodular_selection returned %d frames", len(selected_indices))

        # Ensure minimum frames - if submodular selection returned too few,
        # fill with uniformly sampled frames
        if len(selected_indices) < min_frames:
            logger.debug("Filling frames: %d -> %d", len(selected_indices), min_frames)
            selected_set = set(selected_indices)
            step = max(1, len(frames) // min_frames)

            for i in range(0, len(frames), step):
                if len(selected_indices) >= min_frames:
                    break
                if i not in selected_set:
                    selected_indices.append(i)
                    selected_set.add(i)

            # If still not enough, just add sequential frames
            if len(selected_indices) < min_frames:
                for i in range(len(frames)):
                    if len(selected_indices) >= min_frames:
                        break
                    if i not in selected_set:
                        selected_indices.append(i)
                        selected_set.add(i)

            selected_indices = sorted(selected_indices)
            logger.debug("After filling: %d frames", len(selected_indices))

        # Store selected indices for timestamp calculation
        self._selected_indices = selected_indices

        key_frames = frames[selected_indices]

        self._report_progress(88, f"Selected {len(key_frames)} key frames")
        return key_frames
