import os
import cv2
import winsound  # For sound on Windows
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Replace with your DroidCam IP and port if using Wi-Fi
DROIDCAM_URL = os.environ.get("DROIDCAM_URL")  # Update with your IP

# Initialize video capture
cap = cv2.VideoCapture(DROIDCAM_URL)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

photo_count = 0  # Counter for unique photo filenames

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Display the live feed
    cv2.imshow("DroidCam Feed", frame)

    # Wait for key press: spacebar captures the photo
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # Spacebar to capture the photo
        photo_count += 1  # Increment counter for each photo captured
        photo_filename = f"captured_photo_{photo_count}.jpg"  # Unique filename for each photo
        
        # Save the photo
        cv2.imwrite(photo_filename, frame)
        print(f"Photo captured and saved as '{photo_filename}'")

        # Display "Photo Captured!" on the screen for 2 seconds
        cv2.putText(frame, "Photo Captured!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("DroidCam Feed", frame)
        cv2.waitKey(2000)  # Display the message for 2 seconds

        # Play a sound for feedback (only works on Windows)
        winsound.Beep(1000, 500)  # Frequency, duration in milliseconds

    elif key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
