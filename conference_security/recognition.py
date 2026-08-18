import base64
import math
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


def cosine_distance(first: list[float], second: list[float]) -> float:
    if len(first) != len(second) or not first:
        return math.inf
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return math.inf if denominator == 0 else float(1 - np.dot(a, b) / denominator)


def iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def decode_frame(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The submitted camera frame is not a supported image")
    return image


def image_quality(face: np.ndarray) -> float:
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class InsightFaceEngine:
    """Loads one server-side identity model and never falls back to pixel matching."""

    def __init__(self) -> None:
        self._app = None
        self._attempted = False
        self._error: str | None = None
        self._lock = threading.RLock()

    def _load(self) -> None:
        with self._lock:
            if self._attempted:
                return
            self._attempted = True
            try:
                import onnxruntime
                from insightface.app import FaceAnalysis

                providers = onnxruntime.get_available_providers()
                selected_providers = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider") if provider in providers]
                if not selected_providers:
                    raise RuntimeError("No compatible ONNX Runtime execution provider is available")
                # Gatewatch needs only detection and recognition. Loading landmarks and age/gender
                # models makes each CPU frame substantially slower without improving identity matching.
                app = FaceAnalysis(name="buffalo_l", providers=selected_providers, allowed_modules=["detection", "recognition"])
                app.prepare(ctx_id=0 if "CUDAExecutionProvider" in selected_providers else -1, det_size=(512, 512))
                self._app = app
            except Exception as error:
                self._error = f"{type(error).__name__}: {error}"

    def status(self) -> str:
        if self._app is not None:
            return "insightface"
        if self._attempted:
            return "unavailable"
        return "not-initialized"

    def detect(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], list[float], float, np.ndarray]] | None:
        self._load()
        if self._app is None:
            return None
        try:
            result: list[tuple[tuple[int, int, int, int], list[float], float, np.ndarray]] = []
            for face in self._app.get(frame):
                left, top, right, bottom = (int(value) for value in face.bbox)
                left, top = max(0, left), max(0, top)
                right, bottom = min(frame.shape[1], right), min(frame.shape[0], bottom)
                crop = frame[top:bottom, left:right]
                if crop.size == 0:
                    continue
                embedding = np.asarray(face.normed_embedding, dtype=np.float32).flatten()
                if embedding.size == 0:
                    continue
                result.append(((left, top, right - left, bottom - top), embedding.tolist(), image_quality(crop), crop.copy()))
            return result
        except Exception as error:
            self._error = f"{type(error).__name__}: {error}"
            self._app = None
            return None


@dataclass
class Track:
    track_id: str
    box: tuple[int, int, int, int]
    embedding: list[float]
    quality: float
    crop: np.ndarray
    last_seen: float

    def snapshot(self) -> "Track":
        """Return an independent, in-memory copy for an explicit operator action."""
        return Track(
            track_id=self.track_id,
            box=self.box,
            embedding=self.embedding.copy(),
            quality=self.quality,
            crop=self.crop.copy(),
            last_seen=time.monotonic(),
        )


@dataclass
class EnrollmentCapture:
    capture_id: str
    station_id: str
    operator_id: str
    track: Track
    expires_at: float


class EnrollmentCaptureStore:
    """Bounded in-memory captures created only after an operator selects a face."""

    def __init__(self, expires_after: float = 300.0, max_captures: int = 100) -> None:
        self.expires_after = expires_after
        self.max_captures = max_captures
        self._captures: dict[str, EnrollmentCapture] = {}
        self._lock = threading.RLock()

    def _prune(self, now: float) -> None:
        self._captures = {
            capture_id: capture
            for capture_id, capture in self._captures.items()
            if capture.expires_at > now
        }
        while len(self._captures) >= self.max_captures:
            oldest_id = min(self._captures, key=lambda capture_id: self._captures[capture_id].expires_at)
            self._captures.pop(oldest_id)

    def create(self, station_id: str, operator_id: str, track: Track) -> EnrollmentCapture:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            capture = EnrollmentCapture(
                capture_id=str(uuid.uuid4()),
                station_id=station_id,
                operator_id=operator_id,
                track=track.snapshot(),
                expires_at=now + self.expires_after,
            )
            self._captures[capture.capture_id] = capture
            return capture

    def get(self, capture_id: str, station_id: str, operator_id: str) -> Track | None:
        with self._lock:
            self._prune(time.monotonic())
            capture = self._captures.get(capture_id)
            if capture is None or capture.station_id != station_id or capture.operator_id != operator_id:
                return None
            return capture.track

    def consume(self, capture_id: str) -> None:
        with self._lock:
            self._captures.pop(capture_id, None)


class MultiFaceTracker:
    """Per-station tracker using InsightFace embeddings; Haar remains only for non-identifying face boxes."""

    def __init__(self, expires_after: float = 2.0) -> None:
        cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade))
        self.expires_after = expires_after
        self.tracks: dict[str, Track] = {}

    def _detections(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], list[float], float, np.ndarray]]:
        identity_detections = identity_engine.detect(frame)
        if identity_detections is not None:
            return identity_detections

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result: list[tuple[tuple[int, int, int, int], list[float], float, np.ndarray]] = []
        for x, y, width, height in self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)):
            box = (int(x), int(y), int(width), int(height))
            face = frame[y:y + height, x:x + width]
            if face.size:
                result.append((box, [], image_quality(face), face.copy()))
        return result

    def process(self, frame: np.ndarray) -> list[Track]:
        now = time.monotonic()
        matched: set[str] = set()
        result: list[Track] = []
        for box, embedding, quality, face in self._detections(frame):
            best_id, best_score = None, 0.0
            for track_id, track in self.tracks.items():
                score = iou(box, track.box)
                if track_id not in matched and score > best_score:
                    best_id, best_score = track_id, score
            if best_id and best_score >= 0.25:
                track = self.tracks[best_id]
                track.box, track.embedding, track.quality, track.crop, track.last_seen = box, embedding, quality, face.copy(), now
            else:
                track = Track(str(uuid.uuid4()), box, embedding, quality, face.copy(), now)
                self.tracks[track.track_id] = track
            matched.add(track.track_id)
            result.append(track)
        self.tracks = {key: value for key, value in self.tracks.items() if now - value.last_seen <= self.expires_after}
        return result

    def get(self, track_id: str) -> Track | None:
        track = self.tracks.get(track_id)
        if track and time.monotonic() - track.last_seen <= self.expires_after:
            return track
        return None


class TrackerRegistry:
    def __init__(self) -> None:
        self._trackers: dict[str, MultiFaceTracker] = {}

    def for_station(self, station_id: str) -> MultiFaceTracker:
        return self._trackers.setdefault(station_id, MultiFaceTracker())


identity_engine = InsightFaceEngine()
trackers = TrackerRegistry()
enrollment_captures = EnrollmentCaptureStore()