import os
import cv2
import json
# import uuid
from datetime import datetime
import winsound
import shutil
from dotenv import load_dotenv
from deepface import DeepFace
import mediapipe as mp
import numpy as np
from scipy import spatial
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QWidget, QLabel, QScrollArea, QGroupBox,
                            QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QPainter, QPen, QColor, QFont
from PyQt5.QtCore import QRect
from PyQt5.QtCore import QThread, pyqtSignal


# Load environment variables
load_dotenv()

CAMERA_INDEX = 0 # 0 for built-in camera, 1 for usb camera , >=2 for droidcam

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
DROIDCAM_URL = os.getenv("DROIDCAM_URL")

def get_timestamp_id():
    """Generate a unique ID based on current timestamp with milliseconds"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Format: YYYYMMDD_HHMMSS_FFF

class LoadFacesThread(QThread):
    faces_loaded = pyqtSignal(dict)
    
    def __init__(self, data_path, faces_path):
        super().__init__()
        self.data_path = data_path
        self.faces_path = faces_path
    
    def run(self):
        """Load face data in background thread."""
        familiar_faces = {}
        if not os.path.exists(self.data_path):
            self.faces_loaded.emit(familiar_faces)
            return
        
        for file in os.listdir(self.data_path):
            if file.endswith("_data.json"):
                try:
                    face_path = os.path.join(self.faces_path, file.replace("_data.json", ".jpg"))
                    with open(os.path.join(self.data_path, file), "r", encoding="utf-8") as f:
                        face_data = json.load(f)
                        familiar_faces[face_path] = face_data
                except Exception as e:
                    print(f"Error loading face data: {str(e)}")
        
        self.faces_loaded.emit(familiar_faces)

class FaceLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.border_color = None
        self.match_id = None
        self.original_pixmap = None
        self.distance = None
        
    def set_face_image(self, face_path):
        self.original_pixmap = QPixmap(face_path)
        if not self.original_pixmap.isNull():
            self.original_pixmap = self.original_pixmap.scaled(
                150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.update()
        
    def set_match(self, match_id, color, distance):
        self.match_id = match_id
        self.border_color = color
        self.distance = distance
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw the original image centered
        if self.original_pixmap:
            x = (self.width() - self.original_pixmap.width()) // 2
            y = (self.height() - self.original_pixmap.height()) // 2
            painter.drawPixmap(x, y, self.original_pixmap)
            
            # Draw highlight if matched
            if self.border_color and self.match_id is not None:
                # Border
                pen = QPen(self.border_color)
                pen.setWidth(4)
                painter.setPen(pen)
                painter.drawRect(0, 0, self.width()-1, self.height()-1)
                
                # Match info box
                painter.setPen(Qt.white)
                painter.setBrush(QColor(0, 0, 0, 180))  # Semi-transparent black
                info_rect = QRect(0, self.height()-25, self.width(), 25)
                painter.drawRect(info_rect)
                
                # Match text
                painter.setFont(QFont("Arial", 8))
                painter.drawText(
                    info_rect, 
                    Qt.AlignCenter, 
                    f"Match #{self.match_id} ({self.distance:.3f})"
                )
        
        painter.end()

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize paths
        self.initialize_paths()
        
        # UI Components
        self.init_ui()
        
        # Camera setup
        self.cap = None
        self.frame_count = 0
        self.familiar_faces = self.get_familiar_faces_data()
        
        # Start camera
        self.initialize_camera(index=CAMERA_INDEX)
        
        # Timer for video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS
        
        # Track currently displayed familiar faces
        self.current_matches = {}
        self.next_window_id = 1

        self.db_window = None
        self.face_widgets = {}  # Track face widgets by their paths
        self.match_colors = {}  # Store match colors

        # Load familiar faces in a background thread
        self.familiar_faces = {}
        self.load_faces_thread = LoadFacesThread(self.UNIQUE_FACES_DATA_PATH, self.UNIQUE_FACES_PATH)
        self.load_faces_thread.faces_loaded.connect(self.on_faces_loaded)
        self.load_faces_thread.start()

    def on_faces_loaded(self, faces_data):
        """Callback when faces are loaded in background."""
        self.familiar_faces = faces_data
        self.status_label.setText(f"Loaded {len(faces_data)} known faces")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for main window only."""
        if event.key() == Qt.Key_Space:
            self.capture_faces()
        elif event.key() == Qt.Key_V:
            self.view_database()
        elif event.key() == Qt.Key_C:  # Add 'C' key for clearing faces
            self.clear_faces_ui()
        # elif event.key() == Qt.Key_A:
        #     self.add_selected_face_to_db()
        else:
            super().keyPressEvent(event)

    def handle_db_window_keys(self, event):
        """Handle key events for the database window only."""
        if event.key() == Qt.Key_Escape:
            self.db_window.close()
            self.db_window = None
        else:
            # Forward other keys to default handler
            QMainWindow.keyPressEvent(self.db_window, event)

    def initialize_paths(self):
        """Initialize all required directories."""
        self.IMAGES_PATH = os.path.join(os.path.dirname(__file__), "db", "images")
        self.FACES_PATH = os.path.join(os.path.dirname(__file__), "db", "faces")
        self.UNIQUE_FACES_PATH = os.path.join(os.path.dirname(__file__), "db", "unique_faces")
        self.UNIQUE_FACES_DATA_PATH = os.path.join(os.path.dirname(__file__), "db", "unique_faces_data")  # New

        os.makedirs(self.IMAGES_PATH, exist_ok=True)
        os.makedirs(self.FACES_PATH, exist_ok=True)
        os.makedirs(self.UNIQUE_FACES_PATH, exist_ok=True)
        os.makedirs(self.UNIQUE_FACES_DATA_PATH, exist_ok=True)  # New

    # def set_high_quality_mode(self, enable=True):
    #     """Enable high quality mode with tradeoffs."""
    #     if enable:
    #         # Higher resolution but may reduce FPS
    #         self.timer.setInterval(50)  # ~20 FPS instead of 30
    #         self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    #     else:
    #         # Balanced mode
    #         self.timer.setInterval(30)  # ~30 FPS
    #         self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

    def init_ui(self):
        """Initialize the main UI components."""
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left panel - Camera feed and controls
        left_panel = QVBoxLayout()
        
        # Camera feed
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(800, 600)  # Increased minimum size
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_panel.addWidget(self.camera_label)
        
        # Controls
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()

        # self.quality_btn = QPushButton("Toggle High Quality")
        # self.quality_btn.setCheckable(True)
        # self.quality_btn.toggled.connect(self.set_high_quality_mode)
        # control_layout.addWidget(self.quality_btn)
        self.clear_faces_btn = QPushButton("Reset")
        self.clear_faces_btn.clicked.connect(self.clear_faces_ui)
        self.clear_faces_btn.setFocusPolicy(Qt.NoFocus)  # Prevent stealing keyboard focus
        control_layout.addWidget(self.clear_faces_btn)

        self.capture_btn = QPushButton("Capture Faces (Space)")
        self.capture_btn.clicked.connect(self.capture_faces)
        control_layout.addWidget(self.capture_btn)
        
        # self.add_face_btn = QPushButton("Add Face Manually (A)")
        # self.add_face_btn.clicked.connect(self.add_selected_face_to_db)
        # control_layout.addWidget(self.add_face_btn)
        
        self.view_database_btn = QPushButton("View Database (V)")
        self.view_database_btn.clicked.connect(self.view_database)
        control_layout.addWidget(self.view_database_btn)
        
        control_group.setLayout(control_layout)
        left_panel.addWidget(control_group)
        
        # Right panel - Detected faces and matches
        right_panel = QVBoxLayout()
        
        # Detected faces
        self.detected_faces_group = QGroupBox("Detected Faces")
        self.detected_faces_layout = QHBoxLayout()
        self.detected_faces_group.setLayout(self.detected_faces_layout)
        right_panel.addWidget(self.detected_faces_group)
        
        # Matches
        self.matches_group = QGroupBox("Recognized Faces")
        self.matches_layout = QHBoxLayout()
        self.matches_group.setLayout(self.matches_layout)
        right_panel.addWidget(self.matches_group)
        
        # Status
        self.status_label = QLabel("Ready")
        right_panel.addWidget(self.status_label)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 60)
        main_layout.addLayout(right_panel, 40)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Set styles
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QPushButton {
                padding: 5px;
                font-weight: bold;
                min-width: 80px;
            }
            QLabel {
                font-size: 14px;
            }
            /* Specific style for Add to DB buttons */
            QPushButton[text="Add to DB"] {
                margin-top: 5px;
            }
            QPushButton[text="Reset"] {
                background-color: #607D8B;  /* Blue-gray */
                color: white;
                border: 1px solid #455A64;
            }
            QPushButton[text="Reset"]:hover {
                background-color: #546E7A;  /* Slightly darker blue-gray */
            }
            QPushButton[text="Reset"]:pressed {
                background-color: #455A64;  /* Even darker for pressed state */
            }
        """)
        # Prevent buttons from stealing keyboard focus
        btns = [
            self.capture_btn,
            self.view_database_btn
            # self.add_face_btn
        ]
        for btn in btns:
            btn.setFocusPolicy(Qt.NoFocus)
        
        # Ensure main window gets keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

    def clear_faces_ui(self):
        """Clear all detected and recognized faces from the UI."""
        # Clear detected faces
        for i in reversed(range(self.detected_faces_layout.count())): 
            widget = self.detected_faces_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        # Clear recognized faces
        for i in reversed(range(self.matches_layout.count())): 
            widget = self.matches_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        # Clear current matches tracking
        if hasattr(self, 'face_labels'):
            self.face_labels = {}
        self.status_label.setText("UI cleared")

    def initialize_camera(self, index=0):
        """Initialize video capture with high resolution."""
        if self.cap is not None:
            self.cap.release()
        
        if index <= 1:
            self.cap = cv2.VideoCapture(index)
        else:
            self.cap = cv2.VideoCapture(DROIDCAM_URL)
        
        # Set to highest possible resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Alternatively, let OpenCV choose the maximum supported resolution
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video stream")
            self.close()
        else:
            # Print actual resolution being used
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.status_label.setText(f"Camera initialized at {width}x{height}")

    def update_frame(self):
        """Update the camera feed with high quality."""
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("Failed to grab frame")
            return
            
        self.frame_count += 1
        
        # Perform face detection every FRAME_SKIP frames
        if self.frame_count % 1 == 0:
            self.faces_boxes = self.detect_faces(frame)
            self.draw_faces(frame, self.faces_boxes)
        
        # Convert to QImage without quality loss
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale the pixmap to fit the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.width(), 
            self.camera_label.height(), 
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation  # Use smooth scaling
        ))

    def detect_faces(self, frame, scale=1):
        """Detect faces in the frame."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            faces_boxes = []
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    w = int(bboxC.width * w)
                    h = int(bboxC.height * h)
                    faces_boxes.append((x, y, w, h))

            return faces_boxes
        except Exception as e:
            self.status_label.setText(f"Detection error: {str(e)}")
            return []

    def draw_faces(self, frame, faces):
        """Draw bounding boxes around detected faces."""
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def preprocess(self, img_path):
        # Check if file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Load as grayscale (directly)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image file or format")
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)
        
        # Convert back to 3-channel (if needed for DeepFace)
        return cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)

    def capture_faces(self):
        """Capture and process detected faces."""
        self.capture_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        QTimer.singleShot(200, lambda: self.capture_btn.setStyleSheet(""))

        if not hasattr(self, 'faces_boxes') or not self.faces_boxes:
            self.status_label.setText("No faces detected to capture")
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("Failed to grab frame for capture")
            return
            
        frame_identifier = get_timestamp_id()
        frame_name = f"{frame_identifier}.jpg"
        frame_path = os.path.join(self.IMAGES_PATH, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_processed = self.preprocess(frame_path)

        faces_paths = []
        for i, (x, y, w, h) in enumerate(self.faces_boxes):
            face_crop = frame_processed[y:y + h, x:x + w]
            face_identifier = f"{frame_identifier}_face_0{i}"
            face_name = f"{face_identifier}.jpg"
            face_path = os.path.join(self.FACES_PATH, face_name)
            cv2.imwrite(face_path, face_crop)
            faces_paths.append(face_path)
        
        # Update UI with captured faces
        self.show_detected_faces(faces_paths)
        
        # Play sound feedback
        winsound.Beep(1000, 500)
        
        # Process face data - save to unique_faces_data instead of data
        frame_faces_data = self.get_faces_data(faces_paths, frame_path)
        for face_file, face_data in frame_faces_data.items():
            # Save each face's data individually
            unique_filename = os.path.basename(face_file)
            data_path = os.path.join(self.UNIQUE_FACES_DATA_PATH, f"{os.path.splitext(unique_filename)[0]}_data.json")
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(face_data, f, ensure_ascii=False, indent=4)

        # Check for matches
        self.check_for_matches(frame_faces_data)

    def show_detected_faces(self, faces_paths):
        """Display the detected faces in the UI with optimized loading."""
        # Clear previous faces
        self.clear_detected_faces_ui()
        
        # Load all images first
        pixmaps = []
        for face_path in faces_paths:
            pixmap = QPixmap(face_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pixmaps.append(pixmap)
        
        # Then create UI elements
        for i, (face_path, pixmap) in enumerate(zip(faces_paths, pixmaps)):
            # Create container widget for face + button
            container = QWidget()
            container_layout = QVBoxLayout()
            container.setLayout(container_layout)
            
            # Create a custom label
            face_label = FaceLabel()
            face_label.setAlignment(Qt.AlignCenter)
            face_label.setFixedSize(150, 150)
            face_label.set_face_image(face_path)
            
            # Add button
            add_btn = QPushButton("Add to DB")
            add_btn.setFixedHeight(30)
            add_btn.setFocusPolicy(Qt.NoFocus)
            add_btn.clicked.connect(lambda _, path=face_path: self.add_selected_face_to_db(path))
            
            container_layout.addWidget(face_label)
            container_layout.addWidget(add_btn)
            self.detected_faces_layout.addWidget(container)
            
            # Store reference
            if not hasattr(self, 'face_labels'):
                self.face_labels = {}
            self.face_labels[face_path] = face_label

    def clear_detected_faces_ui(self):
        """Clear just the detected faces UI."""
        for i in reversed(range(self.detected_faces_layout.count())): 
            widget = self.detected_faces_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

    def add_selected_face_to_db(self, face_path):
        """Add a specific detected face to the database."""
        try:
            # First check for exact image duplicates (byte comparison)
            with open(face_path, 'rb') as f:
                new_face_bytes = f.read()
                
            # Compare against all existing faces in unique_faces
            for existing_face in os.listdir(self.UNIQUE_FACES_PATH):
                existing_path = os.path.join(self.UNIQUE_FACES_PATH, existing_face)
                with open(existing_path, 'rb') as f:
                    if f.read() == new_face_bytes:
                        QMessageBox.information(self, "Info", "This image already exists in the database")
                        return

            # If no exact duplicate found, proceed with adding
            face_name = f"manual_{get_timestamp_id()}.jpg"
            unique_face_path = os.path.join(self.UNIQUE_FACES_PATH, face_name)
            
            # Copy the face image to unique faces directory
            shutil.copy2(face_path, unique_face_path)
            
            # Create a dummy frame path
            frame_name = f"manual_{get_timestamp_id()}.jpg"
            frame_path = os.path.join(self.IMAGES_PATH, frame_name)
            cv2.imwrite(frame_path, cv2.imread(face_path))
            
            # Generate face data
            face_data = self.get_faces_data([unique_face_path], frame_path)
            
            if not face_data:
                raise ValueError("Could not process the selected face image")
                
            # Save face data
            data_filename = f"{os.path.splitext(face_name)[0]}_data.json"
            data_path = os.path.join(self.UNIQUE_FACES_DATA_PATH, data_filename)
            
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(face_data[unique_face_path], f, ensure_ascii=False, indent=4)
            
            # Add to in-memory database
            self.familiar_faces[unique_face_path] = face_data[unique_face_path]
            
            QMessageBox.information(self, "Success", "Face added to database successfully")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to add face: {str(e)}")

    def check_for_matches(self, frame_faces_data):
        """Check if any faces match known faces."""
        self.clear_matches_ui()
        self.match_colors = {}
        match_counter = 0
        
        for face_path, face_data in frame_faces_data.items():
            known_face_path, known_face_data, distance = self.find_similar_face(face_data, self.familiar_faces)
            
            if known_face_path:
                match_counter += 1
                hue = (match_counter * 60) % 360
                match_color = QColor.fromHsv(hue, 255, 255)
                self.match_colors[match_counter] = match_color
                
                self.highlight_detected_face(face_path, match_counter, match_color, distance)
                self.show_match(known_face_data, match_counter, match_color, distance)
                self.status_label.setText(f"Match found! ({match_counter} faces recognized)")
                # Remove the new face's data file since it's a duplicate
                face_name = os.path.splitext(os.path.basename(face_path))[0]
                data_path = os.path.join(self.UNIQUE_FACES_DATA_PATH, f"{face_name}_data.json")
                if os.path.exists(data_path):
                    try:
                        os.remove(data_path)
                    except Exception as e:
                        self.status_label.setText(f"Couldn't remove duplicate data: {str(e)}")
                
                # Also remove from unique_faces folder if it was copied there
                unique_face_path = os.path.join(self.UNIQUE_FACES_PATH, os.path.basename(face_path))
                if os.path.exists(unique_face_path):
                    try:
                        os.remove(unique_face_path)
                    except Exception as e:
                        self.status_label.setText(f"Couldn't remove duplicate face: {str(e)}")
                        
            else:
                # Add to unique faces (unchanged from before)
                unique_face_filename = os.path.basename(face_path)
                unique_face_path = os.path.join(self.UNIQUE_FACES_PATH, unique_face_filename)
                shutil.copy2(face_path, unique_face_path)
                
                # Add to in-memory database
                face_data["unique_face_path"] = unique_face_path
                self.familiar_faces[unique_face_path] = face_data
                self.status_label.setText("New faces added to database")

    def clear_matches_ui(self):
        """Clear just the matches UI."""
        for i in reversed(range(self.matches_layout.count())): 
            widget = self.matches_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

    def show_match(self, face_data, match_id, match_color, distance):
        """Display a matched face in the UI with colored border."""
        match_label = FaceLabel()
        match_label.setAlignment(Qt.AlignCenter)
        match_label.setFixedSize(150, 150)
        match_label.set_face_image(face_data["face_path"])
        match_label.set_match(match_id, match_color, distance)
        self.matches_layout.addWidget(match_label)

    def get_faces_data(self, faces_paths, frame_path):
        faces_data = dict()
        for face_file in faces_paths:
            try:
                results = DeepFace.represent(face_file, model_name="Facenet512", enforce_detection=False)
                result = results[0]
                result["frame_path"] = frame_path
                result["face_path"] = face_file
                faces_data[face_file] = result        
            except Exception as e:
                self.status_label.setText(f"Error processing face: {str(e)}")
        return faces_data

    def save_face_data(self, data_path, faces_data):
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(faces_data, f, ensure_ascii=False, indent=4)

    def get_familiar_faces_data(self):
        """Load all familiar faces data from unique_faces_data directory."""
        familiar_faces = {}
        if not os.path.exists(self.UNIQUE_FACES_DATA_PATH):
            return familiar_faces
        
        for file in os.listdir(self.UNIQUE_FACES_DATA_PATH):
            if file.endswith("_data.json"):
                try:
                    face_path = os.path.join(self.UNIQUE_FACES_PATH, file.replace("_data.json", ".jpg"))
                    with open(os.path.join(self.UNIQUE_FACES_DATA_PATH, file), "r", encoding="utf-8") as f:
                        face_data = json.load(f)
                        familiar_faces[face_path] = face_data
                except Exception as e:
                    self.status_label.setText(f"Error loading face data: {str(e)}")
        return familiar_faces

    def find_similar_face(self, new_face_data, familiar_faces, threshold=0.25):
        """Optimized face matching with numpy arrays."""
        if not familiar_faces:
            return None, None, float('inf')
        
        # Get the embedding of the new face
        new_embedding = np.array(new_face_data["embedding"])
        
        # Precompute all known embeddings
        known_paths = list(familiar_faces.keys())
        known_embeddings = np.array([familiar_faces[p]["embedding"] for p in known_paths])
        
        # Batch compute cosine distances
        distances = spatial.distance.cdist([new_embedding], known_embeddings, 'cosine')[0]
        
        # Find best match
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        if min_distance < threshold:
            best_path = known_paths[min_idx]
            best_data = familiar_faces[best_path]
            return best_path, best_data, min_distance
        
        return None, None, float('inf')

    def view_database(self):
        """Show only unique faces from the unique_faces directory."""
        # Close existing window if open
        if self.db_window is not None:
            self.db_window.close()
        
        # Create new window
        self.db_window = QMainWindow(self)
        self.db_window.setWindowTitle("Unique Faces Database")
        self.db_window.setGeometry(200, 200, 800, 600)
        
        self.db_window.keyPressEvent = self.handle_db_window_keys
        
        scroll = QScrollArea()

        scroll = QScrollArea()
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Get all unique face files
        unique_faces = [f for f in os.listdir(self.UNIQUE_FACES_PATH) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not unique_faces:
            label = QLabel("No unique faces in database")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
        else:
            # Sort faces by filename
            unique_faces.sort()
            
            for face_file in unique_faces:
                group = QGroupBox(face_file)
                hbox = QHBoxLayout()
                
                # Face image
                face_path = os.path.join(self.UNIQUE_FACES_PATH, face_file)
                face_label = QLabel()
                face_label.setFixedSize(150, 150)
                pixmap = QPixmap(face_path)
                
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    face_label.setPixmap(pixmap)
                
                # Face info (try to get additional data from familiar_faces)
                info_label = QLabel()
                face_data = None
                
                # Find matching data in familiar_faces
                for known_path, data in self.familiar_faces.items():
                    if os.path.basename(known_path) == face_file:
                        face_data = data
                        break
                
                info_text = f"<b>File:</b> {face_file}<br>"
                if face_data:
                    info_text += f"<b>First seen:</b> {os.path.basename(face_data.get('frame_path', 'unknown'))}<br>"
                    info_text += f"<b>Embedding:</b> {len(face_data.get('embedding', []))} dimensions"
                
                info_label.setText(info_text)
                
                # Delete button
                delete_btn = QPushButton("Delete")
                delete_btn.setFixedSize(80, 30)
                delete_btn.clicked.connect(lambda _, p=face_path: self.delete_face(p))
                
                hbox.addWidget(face_label)
                hbox.addWidget(info_label)
                hbox.addWidget(delete_btn)
                group.setLayout(hbox)
                layout.addWidget(group)
        
        widget.setLayout(layout)
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)

        self.db_window.setCentralWidget(scroll)
        self.db_window.show()
        self.db_window.raise_()

    def delete_face(self, face_path):
        """Delete a face from the unique faces database."""
        reply = QMessageBox.question(
            self, 'Delete Face', 
            f"Are you sure you want to delete {os.path.basename(face_path)}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                # Remove face image
                os.remove(face_path)
                
                # Remove face data
                face_name = os.path.splitext(os.path.basename(face_path))[0]
                data_path = os.path.join(self.UNIQUE_FACES_DATA_PATH, f"{face_name}_data.json")
                if os.path.exists(data_path):
                    os.remove(data_path)
                
                # Remove from in-memory database
                if face_path in self.familiar_faces:
                    del self.familiar_faces[face_path]
                
                QMessageBox.information(self, 'Success', 'Face deleted successfully')
                
                # Refresh the database view
                if self.db_window is not None:
                    self.db_window.close()
                self.view_database()
                
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Could not delete face: {str(e)}')

    def highlight_detected_face(self, face_path, match_id, match_color, distance):
        """Highlight a detected face that was matched in the UI."""
        if hasattr(self, 'face_labels') and face_path in self.face_labels:
            self.face_labels[face_path].set_match(match_id, match_color, distance)

    def add_border_to_pixmap(self, pixmap, color, match_id):
        """Add a colored border with match ID to a pixmap."""
        bordered = QPixmap(pixmap.width() + 20, pixmap.height() + 20)
        bordered.fill(Qt.transparent)
        
        painter = QPainter(bordered)
        try:
            # Draw colored border
            pen = QPen(color, 5)
            painter.setPen(pen)
            painter.drawRect(0, 0, bordered.width() - 1, bordered.height() - 1)
            
            # Draw match number in corner
            painter.setPen(Qt.black)
            painter.setBrush(color)
            painter.setFont(QFont("Arial", 12, QFont.Bold))
            painter.drawText(5, 15, str(match_id))
            
            # Draw original image centered
            painter.drawPixmap(10, 10, pixmap)
        finally:
            painter.end()
        
        return bordered

    def closeEvent(self, event):
        """Clean up when closing the application."""
        if self.cap is not None:
            self.cap.release()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = FaceRecognitionApp()
    window.show()
    app.exec_()