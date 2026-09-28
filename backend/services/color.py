import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim
from .base import BaseSummarizer, SummaryConfig
from typing import Optional


class ColorSummarizer(BaseSummarizer):
    """
    Advanced Color-based video summarization.

    Features:
    - Multi-space color analysis (RGB, HSV, LAB)
    - SSIM-based similarity for perceptual quality
    - Submodular optimization for diverse selection
    - Dominant color extraction using K-means
    - Texture analysis using Gabor filters
    """

    technique_name = "color"

    def __init__(self, config: Optional[SummaryConfig] = None, n_clusters: int = 15):
        super().__init__(config)
        self.n_clusters = n_clusters

    def extract_multi_space_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Extract histograms from multiple color spaces."""
        features = []

        # RGB histogram (8 bins per channel)
        for i in range(3):
            hist = cv2.calcHist([frame], [i], None, [16], [0, 256])
            features.extend(hist.flatten())

        # HSV histogram (better for perceptual similarity)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])  # Hue
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])   # Saturation
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])   # Value
        features.extend(h_hist.flatten())
        features.extend(s_hist.flatten())
        features.extend(v_hist.flatten())

        # LAB histogram (perceptually uniform)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        for i in range(3):
            hist = cv2.calcHist([lab], [i], None, [8], [0, 256])
            features.extend(hist.flatten())

        return np.array(features, dtype=np.float32)

    def extract_dominant_colors(self, frame: np.ndarray, k: int = 5) -> np.ndarray:
        """Extract dominant colors using K-means clustering."""
        # Reshape and convert to float
        pixels = frame.reshape(-1, 3).astype(np.float32)

        # Sample pixels for efficiency
        n_samples = min(10000, len(pixels))
        indices = np.random.choice(len(pixels), n_samples, replace=False)
        samples = pixels[indices]

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

        # Weight by cluster size
        unique, counts = np.unique(labels, return_counts=True)
        weights = counts / counts.sum()

        # Return weighted color features
        return np.concatenate([centers.flatten(), weights])

    def extract_texture_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract texture features using Gabor filters."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        features = []
        # Multiple orientations and frequencies
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            for frequency in [0.1, 0.3]:
                kernel = cv2.getGaborKernel(
                    (21, 21), 5.0, theta, 10.0/frequency, 0.5, 0, cv2.CV_32F
                )
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                features.extend([np.mean(filtered), np.std(filtered)])

        return np.array(features)

    def compute_ssim_matrix(self, frames: np.ndarray) -> np.ndarray:
        """Compute SSIM similarity matrix between frames."""
        self._report_progress(55, "Computing SSIM similarity matrix...")

        n = len(frames)
        # Resize frames for faster SSIM computation
        resized = []
        for frame in frames:
            small = cv2.resize(frame, (160, 120))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            resized.append(gray)

        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            similarity_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                score = ssim(resized[i], resized[j])
                similarity_matrix[i, j] = score
                similarity_matrix[j, i] = score

            if i % 20 == 0:
                progress = 55 + int(15 * i / n)
                self._report_progress(progress, f"Computing similarity {i}/{n}...")

        return similarity_matrix

    def submodular_selection(
        self,
        features: np.ndarray,
        similarity_matrix: np.ndarray,
        n_select: int
    ) -> list[int]:
        """
        Submodular optimization for diverse frame selection.
        Maximizes coverage while minimizing redundancy.
        """
        self._report_progress(75, "Performing submodular optimization...")

        n = len(features)
        selected = []
        remaining = set(range(n))

        # Greedy selection with diminishing returns
        for _ in range(min(n_select, n)):
            best_gain = -np.inf
            best_idx = None

            for idx in remaining:
                # Compute marginal gain
                if not selected:
                    gain = np.sum(similarity_matrix[idx])  # Coverage
                else:
                    # Coverage gain minus redundancy
                    coverage = np.sum(similarity_matrix[idx])
                    redundancy = np.max([similarity_matrix[idx, s] for s in selected])
                    gain = coverage - 2.0 * redundancy  # Penalize similarity to selected

                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return sorted(selected)

    def extract_comprehensive_features(self, frames: np.ndarray) -> np.ndarray:
        """Extract comprehensive visual features from all frames."""
        self._report_progress(45, "Extracting visual features...")

        features = []
        for i, frame in enumerate(frames):
            # Multi-space histogram
            hist_features = self.extract_multi_space_histogram(frame)

            # Dominant colors
            color_features = self.extract_dominant_colors(frame)

            # Texture features
            texture_features = self.extract_texture_features(frame)

            # Combine all features
            combined = np.concatenate([
                hist_features / (np.linalg.norm(hist_features) + 1e-8),
                color_features / (np.linalg.norm(color_features) + 1e-8),
                texture_features / (np.linalg.norm(texture_features) + 1e-8)
            ])
            features.append(combined)

            if i % 30 == 0:
                progress = 45 + int(10 * i / len(frames))
                self._report_progress(progress, f"Processing frame {i}/{len(frames)}...")

        return np.array(features)

    def select_key_frames(self, frames: np.ndarray, **kwargs) -> np.ndarray:
        """Select visually diverse key frames - optimized."""
        target_frames = kwargs.get('n_clusters', self.n_clusters)
        min_frames = kwargs.get('min_frames', target_frames)
        target_frames = max(target_frames, min_frames)

        if len(frames) <= target_frames:
            self._selected_indices = list(range(len(frames)))
            return frames

        self._report_progress(45, "Extracting color features...")

        # Fast color histogram extraction on small frames
        histograms = []
        for frame in frames:
            small = cv2.resize(frame, (80, 60))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-8)
            histograms.append(hist)

        histograms = np.array(histograms)

        self._report_progress(60, "Computing color diversity...")

        # Compute diversity score (difference from neighbors)
        diversity_scores = np.zeros(len(frames))
        for i in range(len(frames)):
            neighbors = []
            for j in range(max(0, i-5), min(len(frames), i+6)):
                if i != j:
                    neighbors.append(np.sum(np.abs(histograms[i] - histograms[j])))
            diversity_scores[i] = np.mean(neighbors) if neighbors else 0

        self._report_progress(75, "Selecting diverse frames...")

        # Select frames: pick most diverse from each segment
        step = len(frames) / target_frames
        selected_indices = []

        for i in range(target_frames):
            start = int(i * step)
            end = min(int((i + 1) * step), len(frames))
            if start >= end:
                start = end - 1

            segment_scores = diversity_scores[start:end]
            best_in_segment = start + np.argmax(segment_scores)
            selected_indices.append(best_in_segment)

        self._selected_indices = selected_indices

        self._report_progress(85, f"Selected {len(selected_indices)} diverse frames")
        return frames[selected_indices]
