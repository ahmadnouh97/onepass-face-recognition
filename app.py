import os
import cv2
import json
import winsound  # For sound on Windows
from dotenv import load_dotenv
from deepface import DeepFace

# Load environment variables
load_dotenv()
DROIDCAM_URL = os.environ.get("DROIDCAM_URL")

# Parameters for optimization
SCALE_FACTOR = 0.5  # Downscale factor for detection
FRAME_SKIP = 1      # Process face detection every N frames


def initialize_camera():
    """Initialize video capture."""
    cap = cv2.VideoCapture(DROIDCAM_URL)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        exit()
    return cap


def detect_faces(frame):
    """Detect faces in the downscaled frame and return bounding box coordinates."""
    small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    try:
        faces = DeepFace.extract_faces(small_frame, detector_backend='opencv', enforce_detection=False)
        return [
            (
                int(face["facial_area"]["x"] / SCALE_FACTOR),
                int(face["facial_area"]["y"] / SCALE_FACTOR),
                int(face["facial_area"]["w"] / SCALE_FACTOR),
                int(face["facial_area"]["h"] / SCALE_FACTOR),
            )
            for face in faces
        ]
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
    
    frame_path = f"captured_photo_{photo_count}.jpg"
    cv2.imwrite(frame_path, frame)

    faces_paths = []
    for i, (x, y, w, h) in enumerate(faces):
        face_crop = frame[y:y + h, x:x + w]
        frame_path = f"captured_face_{photo_count}_{i}.jpg"
        cv2.imwrite(frame_path, face_crop)
        print(f"Face captured and saved as '{frame_path}'")
        faces_paths.append(frame_path)
    
    # Play sound feedback (Windows only)
    winsound.Beep(1000, 500)
    return faces_paths, frame_path


def get_faces_data(faces_paths, frame_path):
    faces_data = []
    for face_file in faces_paths:
        results = DeepFace.represent(face_file, model_name="Facenet", enforce_detection=False)
        result = results[0]
        result["face_path"] = face_file
        result["frame_path"] = frame_path
        faces_data.append(result)
    return faces_data


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
        faces = detect_faces(frame) if frame_count % FRAME_SKIP == 0 else []
        draw_faces(frame, faces)

        cv2.imshow("DroidCam Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Capture photo on spacebar press
            photo_count += 1
            faces_paths, frame_path = capture_photo(frame, faces, photo_count)
            faces_data = get_faces_data(faces_paths, frame_path)

            with open(f"faces_data_{photo_count}.json", "w", encoding="utf-8") as f:
                json.dump(faces_data, f, ensure_ascii=False, indent=4)

        elif key == ord('q'):  # Quit on 'q' key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()