# OnePass Face Recognition

A local webcam/DroidCam app that detects faces, saves each capture, and recognises faces seen earlier. Images and embeddings remain in `db/`; the application does not send them to a remote service.

The default installation works with OpenCV only, including on machines where TensorFlow cannot run. If TensorFlow and MediaPipe are available, the app automatically uses the more accurate MediaPipe detector and DeepFace FaceNet512 embeddings.

## Requirements

- Python 3.11 or 3.12
- A working webcam, or DroidCam configured as a video stream
- [uv](https://docs.astral.sh/uv/) (recommended)

## Install

From the project root, create the environment and install the default, portable setup:

```powershell
uv sync
```

For the enhanced DeepFace/MediaPipe recognition mode, install the optional ML dependencies:

```powershell
uv sync --extra ml
```

If you use DroidCam, copy `.env_example` to `.env` and set its stream URL:

```dotenv
DROIDCAM_URL=http://<phone-ip>:4747/video
```

Leave the variable blank to use the default local camera.

## Run

```powershell
uv run python app.py
```

When the video window opens, press `Space` to save and identify detected faces, or `q` to close the app.

Select another camera or give a stream URL directly:

```powershell
uv run python app.py --source 1
uv run python app.py --source "http://<phone-ip>:4747/video"
```

Tuning example:

```powershell
uv run python app.py --scale 0.75 --confidence 0.6 --threshold 0.4
```

`--threshold` is the maximum cosine distance for treating two embeddings as the same person; lower values are stricter. The fallback OpenCV matcher is most reliable with similar framing and lighting; use `--extra ml` for the strongest recognition results.

## Stored data

The application creates ignored directories under `db/`:

- `images/` — complete captured frames
- `faces/` — individual face crops
- `data/` — saved embeddings and capture metadata
- `unique_faces/` — face crops first identified as new

Delete these folders if you want to remove locally stored recognition data.

## Troubleshooting

- Run `uv run python app.py --help` to see all options.
- If the camera cannot be opened, check `--source`, close other apps using the camera, and confirm DroidCam is running when applicable.
- If the enhanced ML mode cannot initialize TensorFlow, install the current Microsoft Visual C++ Redistributable and make sure the CPU supports TensorFlow. The default OpenCV mode remains available.
- If an old `.venv` points to a missing Python installation, delete only this project’s `.venv` folder and rerun `uv sync`.
