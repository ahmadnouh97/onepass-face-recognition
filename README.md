# Gatewatch Conference Security

Gatewatch is an on-premises operator console for conferences that need guards to review face-recognition suggestions at multiple entry stations. It tracks multiple faces from a browser camera feed, shows whether a person has entered previously, and records a gate decision only after an operator confirms it.

> Recognition is advisory. Guards make the admission decision. Deploy only with documented consent, policy approval, operator training, and appropriate legal authority for biometric data.

## What is included

- React/TypeScript live gate dashboard with multi-face overlays and selected-face actions.
- FastAPI API with individual operator logins, administrator/operator roles, WebSocket frame processing, held enrollment captures, duplicate review, people management, audit records, and immutable entry events.
- PostgreSQL + pgvector-ready schema and Redis/PostgreSQL Docker Compose services for venue deployment.
- A local SQLite development mode that works without Docker.

## System status and deployment scope

`docker-compose.yml` currently starts the shared PostgreSQL and Redis infrastructure only. The FastAPI service and React dashboard run on the host machine. After building the dashboard, FastAPI serves `frontend/dist` at `http://127.0.0.1:8000`.

The live pipeline uses InsightFace buffalo_l embeddings when the production-recognition dependency group is installed. This standard group uses the portable CPU ONNX Runtime 1.20.1, verified for the supported Windows 10 workstation setup; newer 1.29 builds can fail to initialize their native DLL on that platform. It works on operator workstations without CUDA. Use the production-recognition-gpu group only on a GPU server with compatible CUDA drivers. If the ONNX runtime or model is unavailable, Gatewatch still shows face boxes but marks them Recognition unavailable; it never falls back to pixel-based identity matching. Install and verify the model before enrollment. Do not use recognition as an automatic admission system.

## Local development

1. Install Python dependencies and test tools:

   ```powershell
   uv sync --extra test
   ```

### Windows prerequisite for recognition

InsightFace uses the native ONNX Runtime library. On Windows, install the current **Microsoft Visual C++ v14 Redistributable (x64)** before starting Gatewatch. Use the installer linked from [Microsoft's supported redistributable downloads](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170), then restart the terminal. If the runtime cannot load, the dashboard deliberately reports `Recognition unavailable` instead of producing identity matches.

2. Start the API:

   ```powershell
   uv run python app.py
   ```

3. In another terminal, start the dashboard:

   ```powershell
   cd frontend
   npm install
   npm run dev
   ```

4. Open `http://127.0.0.1:5173`. Initial development credentials are `admin` / `change-me-now`.

Local development uses SQLite by default. The browser transmits JPEG frames only while the gate camera is running. Unknown faces remain in memory unless the guard explicitly enrolls them.

## Run with Docker and shared PostgreSQL

### Prerequisites

- Docker Desktop with Compose v2
- Python 3.11 or 3.12 and `uv`
- Node.js 20 or later and npm

### Configure secrets

1. Copy the template:

   ```powershell
   Copy-Item .env_example .env
   ```

2. In `.env`, choose a long, unique `POSTGRES_PASSWORD`. Set `GATEWATCH_DATABASE_URL` to the same password, and replace `GATEWATCH_JWT_SECRET` and `GATEWATCH_DEMO_ADMIN_PASSWORD`.

   Use an alphanumeric database password or URL-encode reserved characters before placing it in `GATEWATCH_DATABASE_URL`. The Compose variable and the URL must resolve to the same password.
### Environment-variable reference

GATEWATCH_ values are read when the API starts. A process environment variable takes precedence over the .env file, which takes precedence over the default in conference_security/config.py. Restart the API after changing any value.

| Variable | Default | Effect |
| --- | --- | --- |
| POSTGRES_PASSWORD | none | Used only by Docker Compose to set the PostgreSQL account password. It must match the password in GATEWATCH_DATABASE_URL. |
| GATEWATCH_DATABASE_URL | sqlite:///./db/gatewatch.db | SQLAlchemy connection URL. Use the PostgreSQL URL in .env when sharing one database between stations. |
| GATEWATCH_JWT_SECRET | insecure development value | Signs operator sessions. Set a long, unique secret before any non-local deployment; changing it signs out existing operators. |
| GATEWATCH_DEMO_ADMIN_USERNAME / GATEWATCH_DEMO_ADMIN_PASSWORD | admin / change-me-now | Creates the bootstrap administrator only when that username does not already exist. Changing these does not reset an existing password. |
| GATEWATCH_RECOGNITION_THRESHOLD | 0.32 | Maximum cosine distance accepted as a match or possible enrollment duplicate. Lower values are stricter; higher values are looser. Tune only with consented venue test data, because this reference matcher has no universal safe value. |
| GATEWATCH_TOKEN_EXPIRE_MINUTES | 480 | Operator-session lifetime in minutes. |
| GATEWATCH_UPLOAD_DIR | db/gatewatch-media | Local directory where enrolled face sample images are stored. Include it in retention and deletion procedures. |
| GATEWATCH_REDIS_URL | redis://127.0.0.1:6379/0 | Reserved for the future Redis-backed inference queue. The current live pipeline does not consume it. |
| GATEWATCH_FRAME_QUEUE_LIMIT | 8 | Reserved queue size setting. It is not yet applied by the current in-process tracker. |

3. Start the shared services:

   ```powershell
   docker compose up -d
   docker compose ps
   ```

   PostgreSQL is available through `127.0.0.1:5432`; its data persists in the `gatewatch-postgres` Docker volume.

4. Install the PostgreSQL driver and start the application:

   ```powershell
   uv sync --extra postgres --extra production-recognition --extra test
   # GPU server only: replace production-recognition with production-recognition-gpu
   uv run python app.py
   ```

5. Build the dashboard in a second terminal:

   ```powershell
   cd frontend
   npm ci
   npm run build
   cd ..
   ```

6. Open `http://127.0.0.1:8000`. Restart the API after the first dashboard build if it was already running.

The API creates the `vector` extension when it connects to PostgreSQL. On first use, InsightFace may need to retrieve its buffalo_l model; provision and test that model on the on-premises server before the event so live gates do not depend on Internet access. People enrolled by the earlier OpenCV demo matcher must be enrolled again after enabling InsightFace because the old embeddings are incompatible. The application does not expose the API directly on the LAN: `app.py` binds it to loopback. Put an HTTPS reverse proxy on the server in front of `127.0.0.1:8000` to make it available to gate stations.

### Redis security note

The current Compose Redis service runs without a password and publishes port `6379`. The application has a Redis URL setting, but the current live recognition pipeline does not consume Redis yet.

Treat this Redis setup as local-development infrastructure only. Before an on-premises venue deployment, change the Compose service to require a password, update `GATEWATCH_REDIS_URL` to include that password, and restrict or remove the host port mapping. Never expose an unauthenticated Redis port to an untrusted network.

## Venue operations

- Put the dashboard/API behind HTTPS on the private venue network. Browser camera access from a LAN hostname requires HTTPS.
- Assign every guard an individual operator account. Create administrator accounts only for staff who manage users, stations, events, thresholds, retention, or exports.
- Select one active conference event before a station records entry decisions.
- Confirm database backups, proxy certificates, and retention policy before enrollment begins.
- Default bootstrap credentials are for initial setup only. Change the administrator password before opening a gate.

### Start, inspect, and stop services

```powershell
docker compose logs -f postgres
docker compose logs -f redis
docker compose down
```

`docker compose down` preserves database and Redis volumes. `docker compose down -v` permanently deletes those volumes; use it only when intentionally discarding all local conference data.

## Operational behavior

- A single active conference event is required before any entry decision is recorded.
- Browser stations preserve a 1280×720 preview. They send a high-resolution identity frame when a track appears and every 1.2 seconds, while intervening 640-pixel frames update normalized face boxes with OpenCV optical flow. Only one frame is in flight, preventing stale-result backlog on CPU-only operator workstations. Each identity refresh is authoritative: overlays not re-detected are removed instead of drifting on background texture. A GPU server is recommended for a five-station deployment.
- A confirmed recognition records a new immutable entry event. Prior entry is displayed as an allowed re-entry, not an automatic denial.
- Selecting a face creates a short-lived, operator-scoped enrollment capture. The enrollment form remains available if the live track disappears, and the capture is removed after a successful enrollment or its expiry.
- Enrollment checks for likely existing faces. Guards must review a possible duplicate or explicitly declare the person distinct.
- Administrators can delete people; this removes their face samples and creates an audit record. Records otherwise remain until an administrator removes them.
- `db/` is local application data. Existing legacy captures are never imported automatically.

## Verification

```powershell
uv run pytest -q
cd frontend
npm run build
```

To validate the Compose file without starting containers:

```powershell
docker compose config
```
