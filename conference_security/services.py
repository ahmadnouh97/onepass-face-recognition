from datetime import datetime
from sqlalchemy import desc, select
from sqlalchemy.orm import Session
from fastapi import HTTPException
import cv2
from .config import settings
from .models import AuditLog, ConferenceEvent, Decision, EntryEvent, FaceSample, Person, Station, User
from .recognition import Track, cosine_distance


def audit(session: Session, actor: User | None, action: str, target_type: str, target_id: str, detail: dict | None = None) -> None:
    session.add(AuditLog(actor_id=actor.id if actor else None, action=action, target_type=target_type, target_id=target_id, detail=detail or {}))


def active_event(session: Session) -> ConferenceEvent:
    event = session.scalar(select(ConferenceEvent).where(ConferenceEvent.active.is_(True)).order_by(desc(ConferenceEvent.created_at)))
    if event is None:
        raise HTTPException(status_code=409, detail="No active conference event is configured")
    return event


def person_out(person: Person) -> dict:
    return {"id": person.id, "display_name": person.display_name, "credential_id": person.credential_id, "version": person.version, "sample_count": len(person.samples)}


def nearest_people(session: Session, embedding: list[float], limit: int = 3) -> list[tuple[Person, float]]:
    results: dict[str, tuple[Person, float]] = {}
    for sample in session.scalars(select(FaceSample).join(Person)).all():
        distance = cosine_distance(embedding, sample.embedding)
        current = results.get(sample.person_id)
        if current is None or distance < current[1]:
            results[sample.person_id] = (sample.person, distance)
    return sorted(results.values(), key=lambda result: result[1])[:limit]


def save_sample(track: Track, person_id: str) -> str:
    directory = settings.upload_dir / person_id
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / f"{track.track_id}.jpg"
    if not cv2.imwrite(str(image_path), track.crop):
        raise HTTPException(status_code=500, detail="Could not persist captured face sample")
    return str(image_path)


def enrollment(session: Session, actor: User, station: Station, track: Track, name: str, credential_id: str | None, allow_distinct: bool) -> dict:
    candidates = [(person, distance) for person, distance in nearest_people(session, track.embedding) if distance < settings.recognition_threshold]
    if candidates and not allow_distinct:
        return {"status": "duplicate_review", "candidates": [{"person_id": person.id, "display_name": person.display_name, "distance": distance} for person, distance in candidates]}
    try:
        if credential_id:
            existing = session.scalar(select(Person).where(Person.credential_id == credential_id).with_for_update())
            if existing:
                raise HTTPException(status_code=409, detail="This credential is already assigned to a person")
        person = Person(display_name=name, credential_id=credential_id)
        session.add(person)
        session.flush()
        session.add(FaceSample(person_id=person.id, image_path=save_sample(track, person.id), embedding=track.embedding, quality=track.quality))
        audit(session, actor, "person.enrolled", "person", person.id, {"station_id": station.id, "track_id": track.track_id})
        session.commit()
    except Exception:
        session.rollback()
        raise
    session.refresh(person)
    return {"status": "enrolled", "person": person_out(person), "candidates": []}


def record_entry(session: Session, actor: User, station: Station, person_id: str | None, track_id: str | None, decision: str, confidence: float | None, reason: str | None) -> EntryEvent:
    if decision not in {item.value for item in Decision}:
        raise HTTPException(status_code=422, detail="Unsupported entry decision")
    event = active_event(session)
    try:
        if person_id and session.scalar(select(Person).where(Person.id == person_id).with_for_update()) is None:
            raise HTTPException(status_code=404, detail="Person was not found")
        entry = EntryEvent(conference_event_id=event.id, person_id=person_id, station_id=station.id, operator_id=actor.id, track_id=track_id, decision=decision, confidence=confidence, reason=reason)
        session.add(entry)
        session.flush()
        audit(session, actor, f"entry.{decision}", "entry_event", entry.id, {"person_id": person_id, "station_id": station.id, "track_id": track_id})
        session.commit()
    except Exception:
        session.rollback()
        raise
    session.refresh(entry)
    return entry


def last_entry(session: Session, event_id: str, person_id: str) -> datetime | None:
    return session.scalar(select(EntryEvent.created_at).where(EntryEvent.conference_event_id == event_id, EntryEvent.person_id == person_id, EntryEvent.decision.in_([Decision.CONFIRMED.value, Decision.OVERRIDE.value])).order_by(desc(EntryEvent.created_at)).limit(1))