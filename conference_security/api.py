from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import shutil
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import desc, select
from sqlalchemy.orm import Session
from .auth import create_token, current_user, decode_token, hash_password, require_role, verify_password
from .config import settings
from .db import SessionLocal, get_session, initialize_database
from .models import AuditLog, ConferenceEvent, Decision, EntryEvent, Person, Role, Station, User
from .recognition import decode_frame, enrollment_captures, identity_engine, trackers
from .schemas import EnrollmentCaptureOut, EnrollmentCaptureRequest, EntryDecisionRequest, EntryEventOut, EnrollmentRequest, EventCreate, PersonOut, StationCreate, TokenRequest, TokenResponse, TrackResult, UserCreate
from .services import active_event, audit, enrollment, last_entry, nearest_people, person_out, record_entry


def bootstrap() -> None:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    initialize_database()
    with SessionLocal() as session:
        if session.scalar(select(User).where(User.username == settings.demo_admin_username)) is None:
            session.add(User(username=settings.demo_admin_username, password_hash=hash_password(settings.demo_admin_password), role=Role.ADMINISTRATOR.value))
        if session.scalar(select(ConferenceEvent).where(ConferenceEvent.active.is_(True))) is None:
            session.add(ConferenceEvent(name="Default conference", active=True))
        if session.scalar(select(Station).where(Station.name == "Main entrance")) is None:
            session.add(Station(name="Main entrance"))
        session.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    bootstrap()
    yield


app = FastAPI(title="Gatewatch Conference Security", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["https://localhost:5173", "http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "recognition": identity_engine.status()}


@app.post("/api/auth/token", response_model=TokenResponse)
def login(request: TokenRequest, session: Session = Depends(get_session)) -> TokenResponse:
    user = session.scalar(select(User).where(User.username == request.username))
    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return TokenResponse(token=create_token(user), role=user.role, username=user.username)


@app.get("/api/me")
def me(user: User = Depends(current_user)) -> dict:
    return {"id": user.id, "username": user.username, "role": user.role}


@app.post("/api/users")
def create_user(request: UserCreate, session: Session = Depends(get_session), admin: User = Depends(require_role(Role.ADMINISTRATOR))) -> dict:
    if request.role not in {Role.ADMINISTRATOR.value, Role.OPERATOR.value}:
        raise HTTPException(status_code=422, detail="Role must be administrator or operator")
    if session.scalar(select(User).where(User.username == request.username)):
        raise HTTPException(status_code=409, detail="That operator ID is already in use")
    user = User(username=request.username, password_hash=hash_password(request.password), role=request.role)
    session.add(user)
    session.flush()
    audit(session, admin, "user.created", "user", user.id, {"username": user.username, "role": user.role})
    session.commit()
    return {"id": user.id, "username": user.username, "role": user.role}

@app.get("/api/events")
def events(session: Session = Depends(get_session), _: User = Depends(current_user)) -> list[dict]:
    return [{"id": event.id, "name": event.name, "active": event.active} for event in session.scalars(select(ConferenceEvent).order_by(desc(ConferenceEvent.created_at))).all()]


@app.post("/api/events")
def create_event(request: EventCreate, session: Session = Depends(get_session), user: User = Depends(require_role(Role.ADMINISTRATOR))) -> dict:
    with session.begin():
        session.query(ConferenceEvent).filter(ConferenceEvent.active.is_(True)).update({ConferenceEvent.active: False})
        event = ConferenceEvent(name=request.name, active=True)
        session.add(event)
        session.flush()
        audit(session, user, "event.created", "conference_event", event.id, {"name": event.name})
    return {"id": event.id, "name": event.name, "active": event.active}


@app.get("/api/stations")
def stations(session: Session = Depends(get_session), _: User = Depends(current_user)) -> list[dict]:
    return [{"id": station.id, "name": station.name, "enabled": station.enabled} for station in session.scalars(select(Station).order_by(Station.name)).all()]


@app.post("/api/stations")
def create_station(request: StationCreate, session: Session = Depends(get_session), user: User = Depends(require_role(Role.ADMINISTRATOR))) -> dict:
    station = Station(name=request.name)
    session.add(station)
    session.flush()
    audit(session, user, "station.created", "station", station.id, {"name": station.name})
    session.commit()
    return {"id": station.id, "name": station.name, "enabled": station.enabled}


@app.get("/api/people", response_model=list[PersonOut])
def people(query: str = "", session: Session = Depends(get_session), _: User = Depends(current_user)) -> list[dict]:
    statement = select(Person).order_by(Person.display_name)
    if query:
        statement = statement.where(Person.display_name.ilike(f"%{query}%"))
    return [person_out(person) for person in session.scalars(statement).unique().all()]


@app.delete("/api/people/{person_id}")
def delete_person(person_id: str, session: Session = Depends(get_session), user: User = Depends(require_role(Role.ADMINISTRATOR))) -> dict:
    person = session.get(Person, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person was not found")
    directory = settings.upload_dir / person.id
    audit(session, user, "person.deleted", "person", person.id, {"display_name": person.display_name})
    session.delete(person)
    session.commit()
    shutil.rmtree(directory, ignore_errors=True)
    return {"deleted": person_id}


@app.post("/api/enrollment-captures", response_model=EnrollmentCaptureOut)
def capture_enrollment(request: EnrollmentCaptureRequest, session: Session = Depends(get_session), user: User = Depends(require_role(Role.ADMINISTRATOR, Role.OPERATOR))) -> EnrollmentCaptureOut:
    station = session.get(Station, request.station_id)
    if station is None or not station.enabled:
        raise HTTPException(status_code=404, detail="Station is not available")
    track = trackers.for_station(station.id).get(request.track_id)
    if track is None:
        raise HTTPException(status_code=409, detail="This face is no longer in view. Select it again.")
    if not track.embedding:
        raise HTTPException(status_code=503, detail="Automatic enrollment requires the InsightFace ONNX recognition engine. Install the production-recognition extra and restart the API.")
    capture = enrollment_captures.create(station.id, user.id, track)
    return EnrollmentCaptureOut(capture_id=capture.capture_id, expires_in_seconds=int(enrollment_captures.expires_after))


@app.post("/api/enrollments")
def enroll(request: EnrollmentRequest, session: Session = Depends(get_session), user: User = Depends(require_role(Role.ADMINISTRATOR, Role.OPERATOR))) -> dict:
    station = session.get(Station, request.station_id)
    if station is None or not station.enabled:
        raise HTTPException(status_code=404, detail="Station is not available")
    track = enrollment_captures.get(request.capture_id, station.id, user.id) if request.capture_id else trackers.for_station(station.id).get(request.track_id or "")
    if track is None:
        raise HTTPException(status_code=409, detail="This saved face capture has expired. Select the face again.")
    if track.quality < 30:
        raise HTTPException(status_code=422, detail="Face image is too blurry to enroll. Ask the person to face the camera.")
    result = enrollment(session, user, station, track, request.display_name, request.credential_id, request.allow_distinct)
    if request.capture_id and result["status"] == "enrolled":
        enrollment_captures.consume(request.capture_id)
    return result

@app.post("/api/entries", response_model=EntryEventOut)
def decide_entry(request: EntryDecisionRequest, session: Session = Depends(get_session), user: User = Depends(require_role(Role.ADMINISTRATOR, Role.OPERATOR))) -> EntryEventOut:
    station = session.get(Station, request.station_id)
    if station is None or not station.enabled:
        raise HTTPException(status_code=404, detail="Station is not available")
    entry = record_entry(session, user, station, request.person_id, request.track_id, request.decision, request.confidence, request.reason)
    person = session.get(Person, entry.person_id) if entry.person_id else None
    return EntryEventOut(id=entry.id, person_id=entry.person_id, person_name=person.display_name if person else None, station_id=entry.station_id, decision=entry.decision, confidence=entry.confidence, created_at=entry.created_at, operator=user.username)


@app.get("/api/entries", response_model=list[EntryEventOut])
def entries(limit: int = Query(default=50, le=200), session: Session = Depends(get_session), _: User = Depends(current_user)) -> list[EntryEventOut]:
    records = session.scalars(select(EntryEvent).order_by(desc(EntryEvent.created_at)).limit(limit)).all()
    users = {user.id: user.username for user in session.scalars(select(User)).all()}
    people_by_id = {person.id: person.display_name for person in session.scalars(select(Person)).all()}
    return [EntryEventOut(id=record.id, person_id=record.person_id, person_name=people_by_id.get(record.person_id), station_id=record.station_id, decision=record.decision, confidence=record.confidence, created_at=record.created_at, operator=users.get(record.operator_id, "unknown")) for record in records]


@app.get("/api/audit")
def audit_log(limit: int = Query(default=100, le=500), session: Session = Depends(get_session), _: User = Depends(require_role(Role.ADMINISTRATOR))) -> list[dict]:
    return [{"id": item.id, "action": item.action, "target_type": item.target_type, "target_id": item.target_id, "detail": item.detail, "created_at": item.created_at} for item in session.scalars(select(AuditLog).order_by(desc(AuditLog.created_at)).limit(limit)).all()]


@app.websocket("/api/live/{station_id}")
async def live_gate(websocket: WebSocket, station_id: str, token: str = Query()):
    try:
        payload = decode_token(token)
    except HTTPException:
        await websocket.close(code=4401)
        return
    with SessionLocal() as session:
        user = session.get(User, payload.get("sub"))
        station = session.get(Station, station_id)
        if user is None or station is None or not station.enabled:
            await websocket.close(code=4403)
            return
    await websocket.accept()
    try:
        while True:
            frame = decode_frame(await websocket.receive_bytes())
            tracker = trackers.for_station(station_id)
            tracks = tracker.process(frame)
            with SessionLocal() as session:
                event = active_event(session)
                response: list[dict] = []
                for track in tracks:
                    candidates = nearest_people(session, track.embedding, limit=2) if track.embedding else []
                    status, candidate, confidence = "unknown", None, None
                    if track.quality < 30:
                        status = "low_quality"
                    elif not track.embedding:
                        status = "recognition_unavailable"
                    elif candidates and candidates[0][1] < settings.recognition_threshold:
                        candidate, distance = candidates[0]
                        confidence = round(max(0.0, 1 - distance), 3)
                        status = "recognized"
                        if len(candidates) > 1 and candidates[1][1] - distance < 0.04:
                            status = "ambiguous"
                    prior = last_entry(session, event.id, candidate.id) if candidate else None
                    if prior and status == "recognized":
                        status = "reentry"
                    x, y, width, height = track.box
                    response.append({"track_id": track.track_id, "box": [x, y, width, height], "status": status, "confidence": confidence, "person_id": candidate.id if candidate else None, "person_name": candidate.display_name if candidate else None, "prior_entry_at": prior.isoformat() if prior else None})
            await websocket.send_json({"type": "tracks", "tracks": response})
    except WebSocketDisconnect:
        return
    except ValueError as error:
        await websocket.send_json({"type": "error", "detail": str(error)})
frontend_dist = Path("frontend/dist")
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="dashboard")