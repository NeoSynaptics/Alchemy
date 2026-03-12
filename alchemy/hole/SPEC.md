# AlchemyHole — Device Tunnel

## What It Does

HTTP tunnel that lets phones/tablets push files to the PC over the local network (or Tailscale). Files land in a staging folder where BaratzaMemory batch-ingests them automatically.

**This module ONLY receives and stores files. No classification, no embedding, no AI. Just receive, save, and handoff to BaratzaMemory.**

## How It Works

1. PC runs a lightweight HTTP receiver on a configurable port (default 8100)
2. Phone/tablet sends files via multipart POST with metadata
3. PC saves files + JSON sidecar to `~/neosy_inbox/hole/`
4. BaratzaMemory's batch ingest watcher picks up new files automatically
5. After successful ingest, files are moved to `~/neosy_inbox/hole/processed/`

## Device Pairing

First-time pairing uses a one-time code:

1. User opens Alchemy dashboard → "Pair Device"
2. Server generates a 6-digit pairing code (valid 5 minutes)
3. User enters code on phone app
4. Phone receives a long-lived API key (stored on device)
5. All subsequent uploads use this API key in the `Authorization: Bearer <key>` header
6. Paired devices visible in dashboard, can be revoked

## API Endpoints

### `POST /v1/hole/pair`
Generate a pairing code. Returns `{ "code": "847293", "expires_at": "..." }`.
Called from dashboard only (requires Alchemy auth token).

### `POST /v1/hole/confirm`
Device sends `{ "code": "847293", "device_name": "iPhone 15" }`.
Returns `{ "api_key": "...", "device_id": "..." }`.

### `POST /v1/hole/upload`
Multipart form upload. Headers: `Authorization: Bearer <device_api_key>`.

Fields:
- `file` — the file (any type: image, video, PDF, text)
- `metadata` (optional JSON) — source app, tags, notes, timestamp

Returns `{ "status": "received", "filename": "...", "size_bytes": ... }`.

### `GET /v1/hole/devices`
List paired devices. Dashboard use only.

### `DELETE /v1/hole/devices/{device_id}`
Revoke a paired device.

## Staging Folder Structure

```
~/neosy_inbox/hole/
├── 2026-03-15_iphone_photo_abc.jpg
├── 2026-03-15_iphone_photo_abc.json
├── 2026-03-15_iphone_note_def.txt
├── 2026-03-15_iphone_note_def.json
└── processed/
    └── (moved here after BaratzaMemory ingests)
```

Each `.json` sidecar:
```json
{
  "source_device": "iPhone 15",
  "device_id": "d_a1b2c3",
  "original_filename": "IMG_4521.jpg",
  "content_type": "image/jpeg",
  "size_bytes": 3450000,
  "uploaded_at": "2026-03-15T14:30:00Z",
  "user_tags": ["vacation", "beach"],
  "user_notes": "Sunset photo from Cancún",
  "source_app": "Photos"
}
```

## Akinator-Style Query

When searching for a file received via Hole, the model can ask clarifying questions instead of returning weak results:

1. User asks: "Find that photo I sent from my phone last week"
2. Model sees 47 photos from last week → too many for a good answer
3. Model asks: "Was it from a specific trip or event?"
4. User: "The beach one"
5. Model narrows to 3 results → returns them

This works through BaratzaMemory's search API with an iterative refinement loop. No special Hole code needed — the metadata sidecars provide the filter dimensions (date, device, tags, source app).

## Network Discovery

For local network use (no Tailscale):
- PC broadcasts mDNS service: `_alchemy-hole._tcp.local`
- Phone app discovers PC automatically on same WiFi
- Falls back to manual IP entry

For Tailscale:
- Use Tailscale MagicDNS hostname (e.g., `desktop-pc.tail12345.ts.net:8100`)
- No port forwarding or NAT traversal needed

## Security

- All uploads require a valid device API key (from pairing)
- File size limit: 500MB per upload (configurable)
- Rate limit: 100 uploads per hour per device
- No executable files accepted (`.exe`, `.bat`, `.sh`, `.ps1` blocked)
- Files are virus-scanned if ClamAV is available (optional)

## Module Structure

```
alchemy/hole/
├── __init__.py
├── SPEC.md          (this file)
├── manifest.py      (module manifest)
├── receiver.py      (HTTP upload handler)
├── pairing.py       (device pairing logic)
└── api.py           (FastAPI router)
```

## Settings

```python
class HoleSettings(BaseModel):
    enabled: bool = False
    port: int = 8100
    inbox_path: str = "~/neosy_inbox/hole"
    max_file_size_mb: int = 500
    rate_limit_per_hour: int = 100
```

## Dependencies

- BaratzaMemory batch ingest (for automatic file processing)
- Alchemy security module (for dashboard auth on pairing endpoints)
- No GPU or model requirements
