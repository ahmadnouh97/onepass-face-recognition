import os
import cv2
import json
import uuid
import winsound
import shutil
from dotenv import load_dotenv
from deepface import DeepFace
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QWidget, QLabel, QScrollArea, QGroupBox,
                            QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QSizePolicy

# Load environment variables
load_dotenv()

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

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
        self.initialize_camera()
        
        # Timer for video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS
        
        # Track currently displayed familiar faces
        self.current_matches = {}
        self.next_window_id = 1

        self.db_window = None

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for main window only."""
        if event.key() == Qt.Key_Space:
            self.capture_faces()
        elif event.key() == Qt.Key_V:
            self.view_database()
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
        self.DATA_PATH = os.path.join(os.path.dirname(__file__), "db", "data")
        self.UNIQUE_FACES_PATH = os.path.join(os.path.dirname(__file__), "db", "unique_faces")

        os.makedirs(self.IMAGES_PATH, exist_ok=True)
        os.makedirs(self.FACES_PATH, exist_ok=True)
        os.makedirs(self.DATA_PATH, exist_ok=True)
        os.makedirs(self.UNIQUE_FACES_PATH, exist_ok=True)

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
        
        self.capture_btn = QPushButton("Capture Faces (Space)")
        self.capture_btn.clicked.connect(self.capture_faces)
        control_layout.addWidget(self.capture_btn)
        
        # self.add_face_btn = QPushButton("Add Face to Database")
        # self.add_face_btn.clicked.connect(self.add_face_to_database)
        # control_layout.addWidget(self.add_face_btn)
        
        self.view_database_btn = QPushButton("View Database")
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
                padding: 8px;
                font-weight: bold;
            }
            QLabel {
                font-size: 14px;
            }
        """)
        # Prevent buttons from stealing keyboard focus
        for btn in [self.capture_btn, self.view_database_btn]:
            btn.setFocusPolicy(Qt.NoFocus)
        
        # Ensure main window gets keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

    def initialize_camera(self, index=0):
        """Initialize video capture with high resolution."""
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(index)
        
        # Set to highest possible resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Try max width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Try max height
        
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
            
        frame_identifier = str(uuid.uuid4())
        frame_name = f"{frame_identifier}.jpg"
        frame_path = os.path.join(self.IMAGES_PATH, frame_name)
        cv2.imwrite(frame_path, frame)

        faces_paths = []
        for i, (x, y, w, h) in enumerate(self.faces_boxes):
            face_crop = frame[y:y + h, x:x + w]
            face_identifier = f"{frame_identifier}_face_0{i}"
            face_name = f"{face_identifier}.jpg"
            face_path = os.path.join(self.FACES_PATH, face_name)

            cv2.imwrite(face_path, face_crop)
            faces_paths.append(face_path)
        
        # Update UI with captured faces
        self.show_detected_faces(faces_paths)
        
        # Play sound feedback
        winsound.Beep(1000, 500)
        
        # Process face data
        frame_faces_data = self.get_faces_data(faces_paths, frame_path)
        frame_faces_data_path = os.path.join(self.DATA_PATH, f"{frame_identifier}_data.json")
        self.save_face_data(frame_faces_data_path, frame_faces_data)

        # Check for matches
        self.check_for_matches(frame_faces_data)

    def show_detected_faces(self, faces_paths):
        """Display the detected faces in the UI."""
        # Clear previous faces
        for i in reversed(range(self.detected_faces_layout.count())): 
            self.detected_faces_layout.itemAt(i).widget().setParent(None)
            
        # Add new faces
        for face_path in faces_paths:
            face_label = QLabel()
            face_label.setAlignment(Qt.AlignCenter)
            face_label.setFixedSize(150, 150)
            
            pixmap = QPixmap(face_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                face_label.setPixmap(pixmap)
                self.detected_faces_layout.addWidget(face_label)

    def check_for_matches(self, frame_faces_data):
        """Check if any faces match known faces."""
        # Clear previous matches
        for i in reversed(range(self.matches_layout.count())): 
            self.matches_layout.itemAt(i).widget().setParent(None)
            
        match_counter = 0
        for face_path, face_data in frame_faces_data.items():
            known_face_path, known_face_data = self.find_similar_face(face_data, self.familiar_faces)
            
            if known_face_path:
                match_counter += 1
                self.show_match(known_face_data, match_counter)
                self.status_label.setText(f"Match found! ({match_counter} faces recognized)")
            else:
                # Add to unique faces
                unique_face_filename = os.path.basename(face_path)
                unique_face_path = os.path.join(self.UNIQUE_FACES_PATH, unique_face_filename)
                shutil.copy2(face_path, unique_face_path)
                
                # Add to in-memory database
                face_data["unique_face_path"] = unique_face_path
                self.familiar_faces[unique_face_path] = face_data
                self.status_label.setText("New faces added to database")

    def show_match(self, face_data, match_id):
        """Display a matched face in the UI."""
        match_label = QLabel()
        match_label.setAlignment(Qt.AlignCenter)
        match_label.setFixedSize(150, 150)
        
        pixmap = QPixmap(face_data["face_path"])
        if not pixmap.isNull():
            pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            match_label.setPixmap(pixmap)
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
        """Load all familiar faces data from the database."""
        familiar_faces = {}
        if not os.path.exists(self.DATA_PATH):
            return familiar_faces
        
        for file in os.listdir(self.DATA_PATH):
            if file.endswith("_data.json"):
                try:
                    with open(os.path.join(self.DATA_PATH, file), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for face_path, face_data in data.items():
                            familiar_faces[face_path] = face_data
                except Exception as e:
                    self.status_label.setText(f"Error loading face data: {str(e)}")
        return familiar_faces

    def find_similar_face(self, new_face_data, familiar_faces, threshold=0.6):
        """Check if the new face matches any familiar face."""
        for known_face_path, known_face_data in familiar_faces.items():
            try:
                distance = DeepFace.verify(
                    img1_path=new_face_data["face_path"],
                    img2_path=known_face_path,
                    model_name="Facenet512",
                    distance_metric="cosine",
                    enforce_detection=False
                )["distance"]
                
                if distance < threshold:
                    return known_face_path, known_face_data
            except Exception as e:
                self.status_label.setText(f"Comparison error: {str(e)}")
        return None, None

    def add_face_to_database(self):
        """Manual method to add a face to the database."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Face Image", "", "Images (*.png *.jpg *.jpeg)", options=options)
            
        if file_path:
            try:
                # Copy to faces directory
                face_name = f"manual_{str(uuid.uuid4())}.jpg"
                face_path = os.path.join(self.FACES_PATH, face_name)
                shutil.copy2(file_path, face_path)
                
                # Create face data
                frame_name = f"manual_{str(uuid.uuid4())}.jpg"
                frame_path = os.path.join(self.IMAGES_PATH, frame_name)
                cv2.imwrite(frame_path, cv2.imread(file_path))
                
                face_data = self.get_faces_data([face_path], frame_path)
                data_path = os.path.join(self.DATA_PATH, f"manual_{str(uuid.uuid4())}_data.json")
                self.save_face_data(data_path, face_data)
                
                # Add to in-memory database
                self.familiar_faces.update(face_data)
                self.status_label.setText("Face added to database successfully")
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to add face: {str(e)}")

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
                # Remove from filesystem
                os.remove(face_path)
                
                # Remove from familiar_faces dictionary
                for path in list(self.familiar_faces.keys()):
                    if os.path.basename(path) == os.path.basename(face_path):
                        del self.familiar_faces[path]
                
                # Remove corresponding data file
                data_files = [f for f in os.listdir(self.DATA_PATH) 
                            if f.endswith('_data.json')]
                
                for data_file in data_files:
                    data_path = os.path.join(self.DATA_PATH, data_file)
                    with open(data_path, 'r') as f:
                        data = json.load(f)
                    
                    # Check if this face is referenced in the data file
                    updated_data = {k:v for k,v in data.items() 
                                if os.path.basename(k) != os.path.basename(face_path)}
                    
                    # Save back if we removed something
                    if len(updated_data) < len(data):
                        if updated_data:
                            with open(data_path, 'w') as f:
                                json.dump(updated_data, f, indent=4)
                        else:
                            os.remove(data_path)
                
                QMessageBox.information(self, 'Success', 'Face deleted successfully')
                
                # Refresh the existing window instead of creating new one
                if self.db_window is not None:
                    self.db_window.close()
                self.view_database()  # This will now create just one window
                
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Could not delete face: {str(e)}')

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