import uuid
from datetime import datetime, timezone
from enum import StrEnum
from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .db import Base
from .vector import VectorEmbedding


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def uid() -> str:
    return str(uuid.uuid4())


class Role(StrEnum):
    ADMINISTRATOR = "administrator"
    OPERATOR = "operator"


class Decision(StrEnum):
    CONFIRMED = "confirmed"
    DENIED = "denied"
    OVERRIDE = "override"


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(24), default=Role.OPERATOR.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ConferenceEvent(Base):
    __tablename__ = "conference_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String(160), unique=True)
    active: Mapped[bool] = mapped_column(default=True, index=True)
    starts_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ends_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Station(Base):
    __tablename__ = "stations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    enabled: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Person(Base):
    __tablename__ = "people"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    display_name: Mapped[str] = mapped_column(String(160), index=True)
    credential_id: Mapped[str | None] = mapped_column(String(100), unique=True, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    samples: Mapped[list["FaceSample"]] = relationship(back_populates="person", cascade="all, delete-orphan")


class FaceSample(Base):
    __tablename__ = "face_samples"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    person_id: Mapped[str] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    image_path: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(VectorEmbedding().with_variant(JSON, "sqlite"))
    quality: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    person: Mapped[Person] = relationship(back_populates="samples")


class EntryEvent(Base):
    __tablename__ = "entry_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    conference_event_id: Mapped[str] = mapped_column(ForeignKey("conference_events.id"), index=True)
    person_id: Mapped[str | None] = mapped_column(ForeignKey("people.id"), nullable=True, index=True)
    station_id: Mapped[str] = mapped_column(ForeignKey("stations.id"), index=True)
    operator_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    track_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    decision: Mapped[str] = mapped_column(String(24))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    actor_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(100), index=True)
    target_type: Mapped[str] = mapped_column(String(80))
    target_id: Mapped[str] = mapped_column(String(100))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)