import os
import cv2
import json
import uuid
import winsound  # For sound on Windows
from dotenv import load_dotenv
from deepface import DeepFace
import mediapipe as mp


# Load environment variables
load_dotenv()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

DROIDCAM_URL = os.environ.get("DROIDCAM_URL")

# Parameters for optimization
SCALE_FACTOR = 1    # Downscale factor for detection
FRAME_SKIP = 1      # Process face detection every N frames

IMAGES_PATH = os.path.join(os.path.dirname(__file__), "db", "images")
FACES_PATH = os.path.join(os.path.dirname(__file__), "db", "faces")
DATA_PATH = os.path.join(os.path.dirname(__file__), "db", "data")

os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(FACES_PATH, exist_ok=True)


def initialize_camera():
    """Initialize video capture."""
    # cap = cv2.VideoCapture(DROIDCAM_URL)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        exit()
    return cap


def detect_faces(frame, scale=1):
    """Detect faces in the downscaled frame and return bounding box coordinates."""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if scale < 1:
            rgb_frame = cv2.resize(rgb_frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

        results = face_detection.process(rgb_frame)

        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                faces.append((x, y, w, h))

        return faces
    except Exception as e:
        print("Face detection error:", e)
        return []


def draw_faces(frame, faces):
    """Draw bounding boxes around detected faces."""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def capture_photo(frame, faces, photo_count):
    """Capture and save each detected face as a separate image."""
    if not faces:
        print("No faces detected.")
        return
    
    # frame_identifier = f"captured_photo_{photo_count}.jpg"
    frame_identifier = generate_identifier()
    frame_name = f"{frame_identifier}.jpg"
    frame_path = os.path.join(IMAGES_PATH, frame_name)
    cv2.imwrite(frame_path, frame)

    faces_paths = []
    for i, (x, y, w, h) in enumerate(faces):
        face_crop = frame[y:y + h, x:x + w]
        face_identifier = f"{frame_identifier}_face_0{i}"
        face_name = f"{face_identifier}.jpg"
        face_path = os.path.join(FACES_PATH, face_name)

        cv2.imwrite(face_path, face_crop)
        print(f"Face captured and saved as '{face_path}'")
        faces_paths.append(face_path)
    
    # Play sound feedback (Windows only)
    winsound.Beep(1000, 500)
    return faces_paths, frame_path, frame_identifier


def get_faces_data(faces_paths, frame_path):
    faces_data = dict()
    for face_file in faces_paths:
        results = DeepFace.represent(face_file, model_name="Facenet", enforce_detection=False)
        result = results[0]
        result["frame_path"] = frame_path
        result["face_path"] = face_file
        faces_data[face_file] = result        

    return faces_data


def save_face_data(data_path, faces_data):
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(faces_data, f, ensure_ascii=False, indent=4)


def generate_identifier():
    return str(uuid.uuid4())


def main():
    cap = initialize_camera()
    frame_count = 0
    photo_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1

        # Perform face detection every FRAME_SKIP frames
        faces = detect_faces(frame, scale=SCALE_FACTOR) if frame_count % FRAME_SKIP == 0 else []
        draw_faces(frame, faces)

        cv2.imshow("DroidCam Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Capture photo on spacebar press
            photo_count += 1
            faces_paths, frame_path, frame_identifier = capture_photo(frame, faces, photo_count)
            frame_faces_data = get_faces_data(faces_paths, frame_path)
            frame_faces_data_path = os.path.join(DATA_PATH, f"{frame_identifier}_data.json")
            save_face_data(frame_faces_data_path, frame_faces_data)

        elif key == ord('q'):  # Quit on 'q' key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()