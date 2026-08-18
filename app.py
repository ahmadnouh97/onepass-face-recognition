"""Capture faces from a camera and identify faces seen earlier in the session."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import cv2
except ImportError as error:
    cv2 = None
    CV2_ERROR = error
else:
    CV2_ERROR = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None

PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_ROOT = PROJECT_ROOT / "db"
IMAGES_PATH = DATABASE_ROOT / "images"
FACES_PATH = DATABASE_ROOT / "faces"
DATA_PATH = DATABASE_ROOT / "data"
UNIQUE_FACES_PATH = DATABASE_ROOT / "unique_faces"


def ensure_dependencies() -> None:
    if CV2_ERROR is not None:
        raise RuntimeError("OpenCV is not installed. Run `uv sync` and retry.") from CV2_ERROR


def create_storage() -> None:
    for directory in (IMAGES_PATH, FACES_PATH, DATA_PATH, UNIQUE_FACES_PATH):
        directory.mkdir(parents=True, exist_ok=True)


def create_face_detector(confidence: float) -> tuple[str, Any]:
    """Prefer MediaPipe; use OpenCV's bundled Haar detector when TensorFlow is unavailable."""
    ensure_dependencies()
    if mp is not None:
        return "mediapipe", mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=confidence
        )
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Could not load OpenCV face detector from {cascade_path}")
    print("MediaPipe is unavailable; using OpenCV's local Haar face detector.")
    return "haar", cascade
def parse_camera_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def initialize_camera(source: int | str):
    ensure_dependencies()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera source {source!r}. Check --source or DROIDCAM_URL.")
    return cap


def detect_faces(frame: Any, detector: tuple[str, Any], scale: float = 1.0) -> list[tuple[int, int, int, int]]:
    """Return clamped (x, y, width, height) boxes in the original frame."""
    if not 0 < scale <= 1:
        raise ValueError("scale must be greater than 0 and at most 1")
    detector_name, detector_instance = detector
    frame_height, frame_width = frame.shape[:2]
    if detector_name == "mediapipe":
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if scale != 1:
            rgb_frame = cv2.resize(rgb_frame, None, fx=scale, fy=scale)
        results = detector_instance.process(rgb_frame)
        boxes: list[tuple[int, int, int, int]] = []
        for detection in results.detections or []:
            bbox = detection.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * frame_width))
            y1 = max(0, int(bbox.ymin * frame_height))
            x2 = min(frame_width, int((bbox.xmin + bbox.width) * frame_width))
            y2 = min(frame_height, int((bbox.ymin + bbox.height) * frame_height))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale != 1:
        gray_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale)
    detected = detector_instance.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    inverse_scale = 1 / scale
    boxes = []
    for x, y, width, height in detected:
        x1 = max(0, int(x * inverse_scale))
        y1 = max(0, int(y * inverse_scale))
        x2 = min(frame_width, int((x + width) * inverse_scale))
        y2 = min(frame_height, int((y + height) * inverse_scale))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes
def draw_faces(frame: Any, faces: list[tuple[int, int, int, int]]) -> None:
    for x, y, width, height in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)


def generate_identifier() -> str:
    return str(uuid.uuid4())


def capture_photo(frame: Any, faces: list[tuple[int, int, int, int]]) -> tuple[list[Path], Path, str] | None:
    if not faces:
        print("No faces detected; nothing was captured.")
        return None
    frame_identifier = generate_identifier()
    frame_path = IMAGES_PATH / f"{frame_identifier}.jpg"
    if not cv2.imwrite(str(frame_path), frame):
        raise RuntimeError(f"Could not save captured frame to {frame_path}")
    face_paths: list[Path] = []
    for index, (x, y, width, height) in enumerate(faces):
        face_crop = frame[y : y + height, x : x + width]
        if face_crop.size == 0:
            continue
        face_path = FACES_PATH / f"{frame_identifier}_face_{index:02d}.jpg"
        if not cv2.imwrite(str(face_path), face_crop):
            raise RuntimeError(f"Could not save face crop to {face_path}")
        print(f"Face captured: {face_path}")
        face_paths.append(face_path)
    if not face_paths:
        frame_path.unlink(missing_ok=True)
        print("No valid face crops were produced.")
        return None
    print("\a", end="", flush=True)
    return face_paths, frame_path, frame_identifier


def _opencv_embedding(face_path: Path) -> list[float]:
    """A dependency-free local fallback for matching repeated, similarly framed faces."""
    image = cv2.imread(str(face_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read face crop {face_path}")
    image = cv2.equalizeHist(cv2.resize(image, (64, 64)))
    return (image.astype("float32") / 255.0).flatten().tolist()


def get_faces_data(face_paths: list[Path], frame_path: Path) -> dict[str, dict[str, Any]]:
    faces_data: dict[str, dict[str, Any]] = {}
    for face_path in face_paths:
        if DeepFace is not None:
            result = DeepFace.represent(img_path=str(face_path), model_name="Facenet512", enforce_detection=False)
            representation = result[0] if isinstance(result, list) else result
            if not isinstance(representation, dict) or not representation.get("embedding"):
                raise RuntimeError(f"No embedding was generated for {face_path}")
            representation["embedding_backend"] = "deepface"
        else:
            representation = {
                "embedding": _opencv_embedding(face_path),
                "embedding_backend": "opencv",
            }
        representation["frame_path"] = str(frame_path)
        representation["face_path"] = str(face_path)
        faces_data[str(face_path)] = representation
    return faces_data
def save_face_data(data_path: Path, faces_data: dict[str, dict[str, Any]]) -> None:
    with data_path.open("w", encoding="utf-8") as handle:
        json.dump(faces_data, handle, ensure_ascii=False, indent=2)


def get_familiar_faces_data() -> dict[str, dict[str, Any]]:
    familiar_faces: dict[str, dict[str, Any]] = {}
    for data_file in DATA_PATH.glob("*_data.json"):
        try:
            with data_file.open(encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("expected an object")
            familiar_faces.update({path: value for path, value in data.items() if isinstance(value, dict) and value.get("embedding")})
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"Skipping unreadable face data {data_file.name}: {error}")
    return familiar_faces


def cosine_distance(first: list[float], second: list[float]) -> float:
    if len(first) != len(second) or not first:
        return math.inf
    try:
        dot_product = sum(float(a) * float(b) for a, b in zip(first, second))
        first_norm = math.sqrt(sum(float(value) ** 2 for value in first))
        second_norm = math.sqrt(sum(float(value) ** 2 for value in second))
    except (TypeError, ValueError):
        return math.inf
    if first_norm == 0 or second_norm == 0:
        return math.inf
    return 1 - dot_product / (first_norm * second_norm)


def find_similar_face(new_face_data: dict[str, Any], familiar_faces: dict[str, dict[str, Any]], threshold: float) -> tuple[str | None, dict[str, Any] | None]:
    new_embedding = new_face_data.get("embedding")
    if not isinstance(new_embedding, list):
        return None, None
    closest_path: str | None = None
    closest_data: dict[str, Any] | None = None
    closest_distance = math.inf
    for known_path, known_data in familiar_faces.items():
        if known_data.get("embedding_backend") != new_face_data.get("embedding_backend"):
            continue
        known_embedding = known_data.get("embedding")
        if not isinstance(known_embedding, list):
            continue
        distance = cosine_distance(new_embedding, known_embedding)
        if distance < closest_distance:
            closest_path, closest_data, closest_distance = known_path, known_data, distance
    if closest_distance < threshold:
        print(f"Similar face found: {closest_path} (distance: {closest_distance:.4f})")
        return closest_path, closest_data
    return None, None


def show_familiar_face(face_data: dict[str, Any], window_id: int) -> None:
    face_path = face_data.get("face_path")
    if face_path:
        image = cv2.imread(face_path)
        if image is not None:
            cv2.imshow(f"Familiar Face {window_id}", image)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.environ.get("DROIDCAM_URL") or "0", help="Camera index or stream URL; defaults to DROIDCAM_URL or 0.")
    parser.add_argument("--scale", type=float, default=1.0, help="Detection scale in (0, 1].")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum MediaPipe detection confidence.")
    parser.add_argument("--threshold", type=float, default=0.4, help="Maximum cosine distance for a match.")
    return parser


def run(args: argparse.Namespace) -> None:
    if not 0 < args.scale <= 1:
        raise ValueError("--scale must be greater than 0 and at most 1")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1")
    if args.threshold < 0:
        raise ValueError("--threshold must be non-negative")
    load_dotenv()
    create_storage()
    detector = create_face_detector(args.confidence)
    cap = initialize_camera(parse_camera_source(args.source))
    familiar_faces = get_familiar_faces_data()
    print("Press Space to capture faces, or q to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab a frame; stopping.")
                break
            faces = detect_faces(frame, detector, args.scale)
            draw_faces(frame, faces)
            cv2.imshow("Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != ord(" "):
                continue
            captured = capture_photo(frame, faces)
            if captured is None:
                continue
            face_paths, frame_path, frame_identifier = captured
            faces_data = get_faces_data(face_paths, frame_path)
            save_face_data(DATA_PATH / f"{frame_identifier}_data.json", faces_data)
            match_counter = 0
            for face_path, face_data in faces_data.items():
                known_path, known_data = find_similar_face(face_data, familiar_faces, args.threshold)
                if known_path and known_data:
                    print("This face is familiar.")
                    match_counter += 1
                    show_familiar_face(known_data, match_counter)
                else:
                    unique_path = UNIQUE_FACES_PATH / Path(face_path).name
                    shutil.copy2(face_path, unique_path)
                    print(f"New face detected; saved to {unique_path}")
                    familiar_faces[str(unique_path)] = face_data
    finally:
        cap.release()
        if hasattr(detector[1], "close"):
            detector[1].close()
        cv2.destroyAllWindows()


def main() -> int:
    args = build_parser().parse_args()
    try:
        run(args)
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
