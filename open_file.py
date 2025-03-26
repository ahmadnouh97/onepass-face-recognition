import os
import subprocess



subprocess.run(["start", os.path.join(os.path.dirname(__file__), "captured_photo_1.jpg")], shell=True)
subprocess.run(["start", os.path.join(os.path.dirname(__file__), "captured_face_1_0.jpg")], shell=True)