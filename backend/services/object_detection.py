import cv2
import numpy as np
from .base import BaseSummarizer, SummaryConfig
from typing import Optional
from pathlib import Path


class ObjectDetectionSummarizer(BaseSummarizer):
    """
    Advanced Object Detection-based video summarization.

    Features:
    - Background subtraction using MOG2/KNN
    - Contour analysis with size and shape filtering
    - Object tracking across frames
    - Saliency detection for visual attention
    - Frame importance based on object presence and movement
    """

    technique_name = "object_detection"

    def __init__(
        self,
        config: Optional[SummaryConfig] = None,
        confidence_threshold: float = 0.5,
        min_objects: int = 1,
    ):
        super().__init__(config)
        self.confidence_threshold = confidence_threshold
        self.min_objects = min_objects

        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

    def compute_saliency_map(self, frame: np.ndarray) -> np.ndarray:
        """Compute visual saliency map using spectral residual approach."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Spectral residual saliency
        dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
        magnitude, phase = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])

        log_magnitude = np.log(magnitude + 1e-8)
        spectral_residual = log_magnitude - cv2.GaussianBlur(log_magnitude, (3, 3), 0)

        # Reconstruct
        real = np.exp(spectral_residual) * np.cos(phase)
        imag = np.exp(spectral_residual) * np.sin(phase)
        dft_modified = np.stack([real, imag], axis=-1)

        saliency = cv2.idft(dft_modified)
        saliency = cv2.magnitude(saliency[:, :, 0], saliency[:, :, 1])
        saliency = cv2.GaussianBlur(saliency, (9, 9), 2.5)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        return saliency

    def detect_moving_objects(self, frame: np.ndarray) -> list[dict]:
        """Detect moving objects using background subtraction and contour analysis."""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows (gray pixels)
        fg_mask[fg_mask == 127] = 0

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * 0.002  # Minimum 0.2% of frame
        max_area = frame_area * 0.5    # Maximum 50% of frame

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / (h + 1e-8)

                # Filter by aspect ratio (avoid very elongated shapes)
                if 0.2 < aspect_ratio < 5:
                    # Compute additional features
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / (hull_area + 1e-8)

                    objects.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'centroid': (x + w // 2, y + h // 2),
                        'aspect_ratio': aspect_ratio,
                        'solidity': solidity
                    })

        return objects

    def detect_static_objects(self, frame: np.ndarray, saliency: np.ndarray) -> list[dict]:
        """Detect salient static objects using edge detection and saliency."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Combine with saliency
        saliency_uint8 = (saliency * 255).astype(np.uint8)
        combined = cv2.bitwise_and(edges, saliency_uint8)

        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined = cv2.dilate(combined, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * 0.005

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Compute saliency score for this region
                region_saliency = np.mean(saliency[y:y+h, x:x+w])

                if region_saliency > 0.3:  # Only keep salient regions
                    objects.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'saliency': region_saliency,
                        'type': 'static'
                    })

        return objects

    def track_objects(self, prev_objects: list[dict], curr_objects: list[dict]) -> int:
        """Simple centroid-based object tracking to count consistent objects."""
        if not prev_objects or not curr_objects:
            return len(curr_objects)

        matched = 0
        max_distance = 100  # Maximum centroid distance for matching

        for curr_obj in curr_objects:
            if 'centroid' not in curr_obj:
                continue

            for prev_obj in prev_objects:
                if 'centroid' not in prev_obj:
                    continue

                dist = np.sqrt(
                    (curr_obj['centroid'][0] - prev_obj['centroid'][0]) ** 2 +
                    (curr_obj['centroid'][1] - prev_obj['centroid'][1]) ** 2
                )

                if dist < max_distance:
                    matched += 1
                    break

        return matched

    def compute_frame_score(
        self,
        moving_objects: list[dict],
        static_objects: list[dict],
        saliency: np.ndarray,
        tracked_count: int
    ) -> float:
        """Compute overall frame importance score."""
        # Moving object score
        moving_score = len(moving_objects) * 0.3

        # Add bonus for larger objects
        if moving_objects:
            areas = [obj['area'] for obj in moving_objects]
            moving_score += np.log1p(np.mean(areas)) * 0.1

        # Static salient object score
        static_score = len(static_objects) * 0.2
        if static_objects:
            saliencies = [obj.get('saliency', 0) for obj in static_objects]
            static_score += np.mean(saliencies) * 0.2

        # Overall saliency
        saliency_score = np.mean(saliency) * 0.2

        # Tracking consistency bonus
        tracking_score = tracked_count * 0.1

        return moving_score + static_score + saliency_score + tracking_score

    def analyze_frames(self, frames: np.ndarray) -> list[tuple[int, float, dict]]:
        """Analyze all frames and compute importance scores."""
        self._report_progress(45, "Analyzing frames for objects...")

        # Reset background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

        frame_scores = []
        prev_objects = []

        for i, frame in enumerate(frames):
            # Compute saliency
            saliency = self.compute_saliency_map(frame)

            # Detect moving objects
            moving_objects = self.detect_moving_objects(frame)

            # Detect static salient objects
            static_objects = self.detect_static_objects(frame, saliency)

            # Track objects
            tracked_count = self.track_objects(prev_objects, moving_objects)

            # Compute frame score
            score = self.compute_frame_score(
                moving_objects, static_objects, saliency, tracked_count
            )

            frame_scores.append((i, score, {
                'moving_count': len(moving_objects),
                'static_count': len(static_objects),
                'tracked': tracked_count,
                'saliency_mean': float(np.mean(saliency))
            }))

            prev_objects = moving_objects

            if i % 20 == 0:
                progress = 45 + int(35 * i / len(frames))
                self._report_progress(progress, f"Analyzing frame {i}/{len(frames)}...")

        return frame_scores

    def select_key_frames(self, frames: np.ndarray, **kwargs) -> np.ndarray:
        """Select frames with significant object presence."""
        min_objects = kwargs.get('min_objects', self.min_objects)

        frame_scores = self.analyze_frames(frames)

        self._report_progress(82, "Selecting key frames...")

        # Sort by score
        sorted_scores = sorted(frame_scores, key=lambda x: x[1], reverse=True)

        # Select top frames with diversity
        n_select = max(10, len(frames) // 10)
        selected_indices = []
        min_distance = max(3, len(frames) // n_select // 2)

        for idx, score, _ in sorted_scores:
            # Check minimum distance from already selected
            if all(abs(idx - s) >= min_distance for s in selected_indices):
                selected_indices.append(idx)
                if len(selected_indices) >= n_select:
                    break

        # Sort by frame order
        selected_indices.sort()

        # Ensure we meet target
        target_frames = kwargs.get('n_clusters', n_select)
        min_frames = kwargs.get('min_frames', target_frames)
        target_frames = max(target_frames, min_frames)

        # Fill if not enough
        if len(selected_indices) < target_frames:
            step = len(frames) // target_frames
            selected_set = set(selected_indices)
            for i in range(0, len(frames), step):
                if len(selected_indices) >= target_frames:
                    break
                if i not in selected_set:
                    selected_indices.append(i)
                    selected_set.add(i)
            selected_indices.sort()

        selected_indices = selected_indices[:target_frames]
        self._selected_indices = selected_indices

        key_frames = frames[selected_indices]
        self._report_progress(88, f"Selected {len(key_frames)} frames")

        return key_frames
