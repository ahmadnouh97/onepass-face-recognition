from datetime import datetime
from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    token: str
    role: str
    username: str


class EventCreate(BaseModel):
    name: str = Field(min_length=2, max_length=160)


class PersonOut(BaseModel):
    id: str
    display_name: str
    credential_id: str | None
    version: int
    sample_count: int


class StationCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)


class EnrollmentRequest(BaseModel):
    station_id: str
    track_id: str | None = None
    capture_id: str | None = None
    display_name: str = Field(min_length=2, max_length=160)
    credential_id: str | None = None
    allow_distinct: bool = False


class EnrollmentCaptureRequest(BaseModel):
    station_id: str
    track_id: str


class EnrollmentCaptureOut(BaseModel):
    capture_id: str
    expires_in_seconds: int


class EnrollmentCandidate(BaseModel):
    person_id: str
    display_name: str
    distance: float


class EnrollmentResponse(BaseModel):
    status: str
    person: PersonOut | None = None
    candidates: list[EnrollmentCandidate] = []


class EntryDecisionRequest(BaseModel):
    station_id: str
    track_id: str | None = None
    person_id: str | None = None
    decision: str
    confidence: float | None = None
    reason: str | None = Field(default=None, max_length=500)


class EntryEventOut(BaseModel):
    id: str
    person_id: str | None
    person_name: str | None
    station_id: str
    decision: str
    confidence: float | None
    created_at: datetime
    operator: str


class TrackResult(BaseModel):
    track_id: str
    box: list[int]
    status: str
    confidence: float | None = None
    person_id: str | None = None
    person_name: str | None = None
    prior_entry_at: datetime | None = None


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=12, max_length=128)
    role: str