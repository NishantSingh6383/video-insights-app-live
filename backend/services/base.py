import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SummaryConfig:
    """Configuration for video summarization."""
    frame_sample_rate: int = 10
    max_frames: int = 1000
    output_fps: int = 24
    output_codec: str = 'mp4v'
    output_type: str = 'dynamic'  # 'dynamic', 'static', or 'both'
    storyboard_cols: int = 4  # Columns in storyboard grid
    thumbnail_width: int = 320  # Width of each thumbnail
    # Downscale frames to this width during extraction (0 = keep native size).
    # Extraction holds every sampled frame in memory, so this caps peak RSS on
    # small hosts: 520 frames of 1080p is ~3.2GB, the same frames at 640px is ~350MB.
    max_frame_width: int = 0


@dataclass
class SummaryResult:
    """Result of video summarization."""
    output_path: str
    total_frames_processed: int
    key_frames_selected: int
    technique: str
    duration_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)
    # Static summary outputs
    storyboard_path: Optional[str] = None
    storyboard_url: Optional[str] = None
    frames_data: Optional[list] = None  # List of {index, timestamp, thumbnail_base64}


class BaseSummarizer(ABC):
    """Base class for all video summarization techniques."""

    technique_name: str = "base"

    def __init__(self, config: Optional[SummaryConfig] = None):
        self.config = config or SummaryConfig()
        self._progress_callback: Optional[Callable[[int, str], None]] = None

    def set_progress_callback(self, callback: Callable[[int, str], None]):
        """Set callback for progress updates. callback(percentage, message)"""
        self._progress_callback = callback

    def _report_progress(self, percentage: int, message: str):
        if self._progress_callback:
            self._progress_callback(percentage, message)

    def extract_frames(self, video_path: str) -> tuple[np.ndarray, dict]:
        """Extract frames from video at specified sample rate."""
        self._report_progress(0, "Opening video file...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_info = {
            'total_frames': total_frame_count,
            'fps': fps,
            'width': width,
            'height': height,
            'duration': total_frame_count / fps if fps > 0 else 0,
        }

        frames = []
        frame_count = 0

        self._report_progress(5, "Extracting frames...")

        while len(frames) < self.config.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.config.frame_sample_rate == 0:
                max_width = self.config.max_frame_width
                if max_width and frame.shape[1] > max_width:
                    scale = max_width / frame.shape[1]
                    frame = cv2.resize(frame, (max_width, max(1, int(frame.shape[0] * scale))))
                frames.append(frame)

                progress = min(40, 5 + int(35 * len(frames) / min(self.config.max_frames, total_frame_count // self.config.frame_sample_rate + 1)))
                if len(frames) % 50 == 0:
                    self._report_progress(progress, f"Extracted {len(frames)} frames...")

            frame_count += 1

        cap.release()

        if len(frames) == 0:
            raise ValueError("No frames could be extracted from video")

        self._report_progress(40, f"Extracted {len(frames)} frames total")
        return np.array(frames), video_info

    def create_video(self, frames: np.ndarray, output_path: str) -> str:
        """Write frames to output video file."""
        self._report_progress(90, f"Creating output video with {len(frames)} frames...")

        if len(frames) == 0:
            raise ValueError("No frames to write")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fps = float(self.config.output_fps)
        logger.debug("create_video: %d frames at %.1f fps (%.2fs)", len(frames), fps, len(frames) / fps)

        # Try imageio with ffmpeg first (best browser compatibility)
        try:
            output_path = self._create_video_imageio(frames, output_path, fps)
            self._report_progress(100, f"Video created with {len(frames)} frames (H.264)")
            return output_path
        except Exception:
            logger.warning("imageio/H.264 encoding failed, falling back to OpenCV mp4v", exc_info=True)
            self._report_progress(92, "imageio failed, trying OpenCV...")

        # OpenCV fallback
        height, width = frames[0].shape[:2]

        # Try mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

        if not writer.isOpened():
            raise ValueError("Could not initialize video writer")

        # Write frames
        frames_written = 0
        for frame in frames:
            if frame is not None and frame.size > 0:
                if len(frame.shape) == 3:
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    writer.write(frame.astype(np.uint8))
                    frames_written += 1

        writer.release()
        logger.debug("OpenCV fallback wrote %d frames", frames_written)

        if frames_written == 0:
            raise ValueError("No frames were written to video")

        self._report_progress(100, f"Video created with {frames_written} frames (mp4v)")
        return output_path

    def _create_video_imageio(self, frames: np.ndarray, output_path: str, fps: float) -> str:
        """Create video using imageio with ffmpeg backend (H.264 codec)."""
        import imageio.v2 as imageio

        # Convert BGR to RGB for imageio
        rgb_frames = []
        for frame in frames:
            if frame is not None and frame.size > 0:
                if len(frame.shape) == 3:
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frames.append(frame.astype(np.uint8))

        if not rgb_frames:
            raise ValueError("No valid frames to write")

        # Use imageio-ffmpeg to write H.264 video
        imageio.mimwrite(
            output_path,
            rgb_frames,
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',
            output_params=['-crf', '23']
        )

        # Verify file was created
        file_size = Path(output_path).stat().st_size
        logger.debug("imageio wrote %s (%d bytes)", output_path, file_size)

        return output_path

    def create_storyboard(self, frames: np.ndarray, output_path: str) -> str:
        """Create a storyboard grid image from key frames."""
        self._report_progress(90, "Creating storyboard...")

        if len(frames) == 0:
            raise ValueError("No frames for storyboard")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Calculate thumbnail size maintaining aspect ratio
        orig_h, orig_w = frames[0].shape[:2]
        thumb_w = self.config.thumbnail_width
        thumb_h = int(orig_h * thumb_w / orig_w)

        # Calculate grid dimensions
        cols = min(self.config.storyboard_cols, len(frames))
        rows = (len(frames) + cols - 1) // cols

        # Create storyboard canvas
        padding = 4
        storyboard_w = cols * thumb_w + (cols + 1) * padding
        storyboard_h = rows * thumb_h + (rows + 1) * padding
        storyboard = np.ones((storyboard_h, storyboard_w, 3), dtype=np.uint8) * 30  # Dark gray bg

        # Place thumbnails
        for i, frame in enumerate(frames):
            row = i // cols
            col = i % cols

            # Resize frame
            thumb = cv2.resize(frame, (thumb_w, thumb_h))

            # Calculate position
            x = col * thumb_w + (col + 1) * padding
            y = row * thumb_h + (row + 1) * padding

            # Place on storyboard
            storyboard[y:y+thumb_h, x:x+thumb_w] = thumb

        cv2.imwrite(output_path, storyboard)
        self._report_progress(95, "Storyboard created")

        return output_path

    def create_frame_thumbnails(
        self,
        frames: np.ndarray,
        frame_indices: list[int],
        video_fps: float
    ) -> list[dict]:
        """Create base64-encoded thumbnails for each key frame."""
        self._report_progress(92, "Creating frame thumbnails...")

        thumbnails = []
        thumb_w = self.config.thumbnail_width
        orig_h, orig_w = frames[0].shape[:2]
        thumb_h = int(orig_h * thumb_w / orig_w)

        for i, frame in enumerate(frames):
            # Resize frame
            thumb = cv2.resize(frame, (thumb_w, thumb_h))

            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            thumb_base64 = base64.b64encode(buffer).decode('utf-8')

            # Calculate timestamp based on sampled frame index
            # frame_indices are indices in the sampled frames array
            # Multiply by sample_rate to get original video frame number
            sampled_idx = frame_indices[i] if i < len(frame_indices) else i
            original_frame_idx = sampled_idx * self.config.frame_sample_rate
            timestamp = original_frame_idx / video_fps if video_fps > 0 else 0

            thumbnails.append({
                'index': i,
                'frame_index': sampled_idx,
                'timestamp': round(timestamp, 2),
                'thumbnail': thumb_base64,
            })

        return thumbnails

    @abstractmethod
    def select_key_frames(self, frames: np.ndarray, **kwargs) -> np.ndarray:
        """Select key frames from extracted frames. Must be implemented by subclasses."""
        pass

    def summarize(
        self,
        video_path: str,
        output_path: str,
        storyboard_path: Optional[str] = None,
        **kwargs
    ) -> SummaryResult:
        """Main method to summarize a video."""
        import time
        start_time = time.time()

        frames, video_info = self.extract_frames(video_path)

        # Store selected indices for timestamp calculation
        self._selected_indices = []
        key_frames = self.select_key_frames(frames, **kwargs)

        # Get selected indices (set by select_key_frames or default to sequential)
        key_frame_indices = getattr(self, '_selected_indices', None)
        if not key_frame_indices or len(key_frame_indices) != len(key_frames):
            key_frame_indices = list(range(len(key_frames)))

        result = SummaryResult(
            output_path=output_path,
            total_frames_processed=len(frames),
            key_frames_selected=len(key_frames),
            technique=self.technique_name,
            metadata=video_info,
        )

        output_type = self.config.output_type

        # Create dynamic video
        if output_type in ('dynamic', 'both'):
            self.create_video(key_frames, output_path)

        # Create static outputs
        if output_type in ('static', 'both'):
            if storyboard_path:
                self.create_storyboard(key_frames, storyboard_path)
                result.storyboard_path = storyboard_path

            # Create thumbnails for individual frame access
            result.frames_data = self.create_frame_thumbnails(
                key_frames,
                key_frame_indices,
                video_info.get('fps', 24)
            )

        duration = time.time() - start_time
        result.duration_seconds = duration

        return result
