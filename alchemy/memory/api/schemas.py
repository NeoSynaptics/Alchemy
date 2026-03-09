"""Pydantic request/response schemas for AlchemyMemory API."""

from __future__ import annotations

from pydantic import BaseModel


# --- Requests ---

class MemorySearchRequest(BaseModel):
    query: str
    include_internet: bool = True
    time_range_hours: int | None = None
    max_ltm_results: int = 10
    max_internet_results: int = 5


class TimelineQueryRequest(BaseModel):
    query: str | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    event_types: list[str] = []
    app_names: list[str] = []
    limit: int = 50
    semantic: bool = True


class IngestRequest(BaseModel):
    event_type: str
    summary: str
    source: str = ""
    app_name: str = ""
    raw_text: str = ""
    meta: dict | None = None


# --- Responses ---

class TimelineEventResponse(BaseModel):
    id: int
    ts: float
    event_type: str
    source: str
    summary: str
    app_name: str
    screenshot_url: str | None = None
    score: float = 0.0
    meta: dict = {}


class ContextPackResponse(BaseModel):
    activity: str
    recent: list[str]
    apps: list[str]
    preferences: dict[str, str]
    generated_at: float
    text_summary: str


class ActivityResponse(BaseModel):
    activity: str
    last_classified_at: float


class HealthResponse(BaseModel):
    status: str
    timeline: dict
    vectors: dict
    stm: dict
    activity: str
    storage_path: str


class SearchTaskResponse(BaseModel):
    task_id: str
    status: str = "started"


class BucketResponse(BaseModel):
    bucket_ts: float
    count: int
    types: dict[str, int] = {}


class TagRequest(BaseModel):
    event_ids: list[int]
    tags: list[str]
