import os
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

os.environ["GATEWATCH_DATABASE_URL"] = "sqlite:///./db/test_gatewatch.db"
os.environ["GATEWATCH_UPLOAD_DIR"] = "db/test-gatewatch-media"
os.environ["GATEWATCH_DEMO_ADMIN_PASSWORD"] = "test-password"

from conference_security.api import app
from conference_security.recognition import EnrollmentCaptureStore, Track, iou, trackers
from conference_security.db import Base, engine
from conference_security.config import settings



@pytest.fixture(autouse=True)
def reset_database():
    Base.metadata.drop_all(bind=engine)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    yield
    Base.metadata.drop_all(bind=engine)
def auth(client: TestClient) -> dict[str, str]:
    response = client.post("/api/auth/token", json={"username": "admin", "password": "test-password"})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['token']}", "Content-Type": "application/json"}


def fresh_track(station_id: str, track_id: str = "track-1") -> Track:
    track = Track(track_id, (10, 10, 80, 80), [0.1] * 32, 80.0, np.full((80, 80, 3), 120, dtype=np.uint8), 999999.0)
    trackers.for_station(station_id).tracks[track_id] = track
    return track


def test_login_and_station_bootstrap():
    with TestClient(app) as client:
        headers = auth(client)
        stations = client.get("/api/stations", headers=headers).json()
        assert stations and stations[0]["name"] == "Main entrance"
        assert client.get("/api/events", headers=headers).json()[0]["active"] is True
        created = client.post("/api/users", headers=headers, json={"username": "gate-two", "password": "correct-horse-battery", "role": "operator"})
        assert created.status_code == 200


def test_enrollment_duplicate_review_and_confirmed_entry():
    with TestClient(app) as client:
        headers = auth(client)
        station_id = client.get("/api/stations", headers=headers).json()[0]["id"]
        fresh_track(station_id)
        response = client.post("/api/enrollments", headers=headers, json={"station_id": station_id, "track_id": "track-1", "display_name": "Ada Lovelace"})
        assert response.status_code == 200
        person = response.json()["person"]
        assert person["sample_count"] == 1
        fresh_track(station_id, "track-2")
        duplicate = client.post("/api/enrollments", headers=headers, json={"station_id": station_id, "track_id": "track-2", "display_name": "Different Name"})
        assert duplicate.json()["status"] == "duplicate_review"
        entry = client.post("/api/entries", headers=headers, json={"station_id": station_id, "track_id": "track-1", "person_id": person["id"], "decision": "confirmed", "confidence": 0.91})
        assert entry.status_code == 200
        assert entry.json()["person_name"] == "Ada Lovelace"
        assert client.get("/api/entries", headers=headers).json()[0]["decision"] == "confirmed"


def test_iou_supports_separate_tracks():
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0
    assert iou((0, 0, 10, 10), (5, 5, 10, 10)) > 0.1

def test_enrollment_uses_frozen_capture_after_track_expires():
    with TestClient(app) as client:
        headers = auth(client)
        station_id = client.get("/api/stations", headers=headers).json()[0]["id"]
        fresh_track(station_id)

        captured = client.post("/api/enrollment-captures", headers=headers, json={"station_id": station_id, "track_id": "track-1"})
        assert captured.status_code == 200
        capture_id = captured.json()["capture_id"]

        trackers.for_station(station_id).tracks.clear()
        enrollment = client.post("/api/enrollments", headers=headers, json={
            "station_id": station_id,
            "capture_id": capture_id,
            "display_name": "Frozen Capture",
        })
        assert enrollment.status_code == 200
        assert enrollment.json()["status"] == "enrolled"


def test_enrollment_capture_store_expires():
    store = EnrollmentCaptureStore(expires_after=0)
    track = Track("track-1", (0, 0, 80, 80), [0.1] * 32, 80.0, np.full((80, 80, 3), 120, dtype=np.uint8), 0.0)
    capture = store.create("station-1", "operator-1", track)
    assert store.get(capture.capture_id, "station-1", "operator-1") is None

def test_enrollment_rejects_tracks_without_a_production_embedding():
    with TestClient(app) as client:
        headers = auth(client)
        station_id = client.get("/api/stations", headers=headers).json()[0]["id"]
        track = fresh_track(station_id)
        track.embedding = []

        response = client.post("/api/enrollment-captures", headers=headers, json={"station_id": station_id, "track_id": track.track_id})

        assert response.status_code == 503
        assert "InsightFace ONNX recognition engine" in response.json()["detail"]
def test_insightface_loads_only_detection_and_recognition_modules(monkeypatch):
    from conference_security import recognition

    captured: dict[str, object] = {}

    class FakeFaceAnalysis:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def prepare(self, **kwargs):
            captured["prepare"] = kwargs

    fake_runtime = types.ModuleType("onnxruntime")
    fake_runtime.get_available_providers = lambda: ["CPUExecutionProvider"]
    fake_app = types.ModuleType("insightface.app")
    fake_app.FaceAnalysis = FakeFaceAnalysis
    fake_insightface = types.ModuleType("insightface")
    fake_insightface.app = fake_app
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_runtime)
    monkeypatch.setitem(sys.modules, "insightface", fake_insightface)
    monkeypatch.setitem(sys.modules, "insightface.app", fake_app)

    engine = recognition.InsightFaceEngine()
    engine._load()

    assert captured["init"] == {
        "name": "buffalo_l",
        "providers": ["CPUExecutionProvider"],
        "allowed_modules": ["detection", "recognition"],
    }
    assert captured["prepare"] == {"ctx_id": -1, "det_size": (512, 512)}
    assert engine.status() == "insightface"