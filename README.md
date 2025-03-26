# Requirements
- Install Docker.

# Build and Run
- Build the Docker image::
    ```
    docker build -t one-pass-face-recognition .
    ```
- Run the Docker container::
    ```
    docker run --rm -e DROIDCAM_URL=http://192.168.1.107:4747/video one-pass-face-recognition
   ```